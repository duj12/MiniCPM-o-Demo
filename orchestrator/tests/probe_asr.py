#!/usr/bin/env python3
"""阶段 0 量测：ASR 服务（Fun-ASR 协议）的行为。

回答：
  1. 连通性与握手是否如文档所述
  2. `2pass-online` 部分结果的延迟（Policy 要用它做实时决策）
  3. `2pass-offline` 最终结果的时间戳字段形状与单位
  4. `turnsense` 消息是否真的会下发、label 取值、字段单位

用法::

    python -m orchestrator.tests.probe_asr                      # 合成音频
    python -m orchestrator.tests.probe_asr --wav path/to.wav    # 真实语音（推荐）
    python -m orchestrator.tests.probe_asr --pcm16 path/to.pcm  # 裸 PCM16 16k

强烈建议喂**真实语音**：合成正弦不会触发 turnsense（那不是语音），
只有真实语音才能验证 turnsense 的 label 语义。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np

SR = 16000
DEFAULT_URL = "ws://192.168.88.101:31366"


@dataclass
class AsrEvent:
    t: float              # 本地收到时刻（相对 t_start）
    mode: str
    text: str
    is_final: bool
    raw: dict


@dataclass
class ProbeResult:
    url: str
    chunk_ms: int
    audio_s: float = 0.0
    events: List[AsrEvent] = field(default_factory=list)
    turnsense: List[AsrEvent] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    t_start: float = 0.0
    first_partial_at: Optional[float] = None
    first_final_at: Optional[float] = None
    done_at: Optional[float] = None

    def summary(self) -> str:
        out = ["=" * 68, f"ASR 量测结果  url={self.url}", "-" * 68]
        out.append(f"  音频时长        : {self.audio_s:.2f}s   发送块 = {self.chunk_ms}ms")
        online = [e for e in self.events if e.mode == "2pass-online"]
        offline = [e for e in self.events if e.mode.startswith("2pass-offline")]
        out.append(f"  2pass-online 帧 : {len(online)}")
        out.append(f"  2pass-offline 帧: {len(offline)}")
        out.append(f"  turnsense 帧    : {len(self.turnsense)}")

        if self.first_partial_at is not None:
            out.append(f"  首个部分结果延迟: {(self.first_partial_at - self.t_start) * 1000:.0f} ms")
        if self.first_final_at is not None:
            out.append(f"  首个最终结果    : {(self.first_final_at - self.t_start) * 1000:.0f} ms")
        if self.done_at is not None:
            out.append(f"  整条流结束      : {(self.done_at - self.t_start) * 1000:.0f} ms")

        if self.turnsense:
            labels = [e.raw.get("label") for e in self.turnsense]
            out.append(f"  turnsense labels: {labels}")
            # 打印一条完整样例，供接口设计参考
            sample = self.turnsense[0].raw
            out.append(f"  turnsense 样例  : {json.dumps(sample, ensure_ascii=False)[:400]}")

        if offline:
            out.append("  最终结果文本:")
            for e in offline[-3:]:
                out.append(f"    {e.text!r}")
            last = offline[-1].raw
            for key in ("timestamp", "vad_segments", "start_time", "end_time",
                        "stamp_sents", "confidence", "index", "slice_type"):
                if key in last:
                    v = last[key]
                    vs = json.dumps(v, ensure_ascii=False) if not isinstance(v, str) else v
                    out.append(f"    {key}: {vs[:200]}")

        if online:
            out.append("  部分结果（最后 3 条）:")
            for e in online[-3:]:
                conf = e.raw.get("confidence", {})
                avg = conf.get("avg") if isinstance(conf, dict) else None
                out.append(f"    {e.text!r}  conf_avg={avg}")

        out.append(f"  错误: {self.errors if self.errors else '无'}")
        out.append("=" * 68)
        return "\n".join(out)


def load_audio(args) -> tuple[np.ndarray, float]:
    """返回 (float32 mono [-1,1], 时长秒)。"""
    if args.wav:
        with wave.open(args.wav, "rb") as w:
            assert w.getframerate() == SR, f"需要 {SR}Hz，得到 {w.getframerate()}"
            n = w.getnframes()
            raw = w.readframes(n)
            x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            if w.getnchannels() == 2:
                x = x.reshape(-1, 2).mean(axis=1)
        return x, len(x) / SR
    if args.pcm16:
        x = np.frombuffer(Path(args.pcm16).read_bytes(), dtype=np.int16).astype(np.float32) / 32768.0
        return x, len(x) / SR
    # 合成：一段"像语音"的调频信号 + 静音尾（触发 VAD 闭合）
    dur = args.synth_seconds
    t = np.arange(int(dur * SR), dtype=np.float32) / SR
    x = 0.3 * np.sin(2 * np.pi * (150 + 80 * np.sin(2 * np.pi * 3 * t)) * t)
    x *= (0.5 + 0.5 * np.sin(2 * np.pi * 2.5 * t))  # 包络，模拟音节
    x = np.concatenate([x, np.zeros(int(1.0 * SR), dtype=np.float32)])  # 1s 尾静音
    return x.astype(np.float32), len(x) / SR


async def run_probe(url: str, x: np.ndarray, duration_s: float, chunk_ms: int,
                    verbose: bool) -> ProbeResult:
    import websockets

    res = ProbeResult(url=url, chunk_ms=chunk_ms, audio_s=duration_s)
    chunk_samples = int(SR * chunk_ms / 1000)
    n_chunks = max(1, len(x) // chunk_samples)

    async with websockets.connect(
        url, subprotocols=["binary"], ping_interval=None, max_size=64 * 1024 * 1024
    ) as ws:
        cfg = {
            "mode": "2pass",
            "chunk_size": [5, 10, 5],
            "chunk_interval": 10,
            "audio_fs": SR,
            "wav_name": "probe",
            "wav_format": "pcm",
            "is_speaking": True,
            "itn": True,
            "vad_tail_sil": 600,
            "vad_max_len": 20000,
            "vad_energy": -100,
            "svs_lang": "auto",
            "spk_diar": False,
            "enable_turnsense": True,
            "enable_timestamp": True,
            "confidence_threshold": 0.8,
            "online_confidence_threshold": 0.6,
        }
        await ws.send(json.dumps(cfg))
        res.t_start = time.perf_counter()

        async def sender() -> None:
            for i in range(n_chunks):
                seg = x[i * chunk_samples:(i + 1) * chunk_samples]
                pcm16 = np.clip(seg * 32767.0, -32768, 32767).astype(np.int16)
                await ws.send(pcm16.tobytes())
                if i == n_chunks - 1:
                    await ws.send(json.dumps({"is_speaking": False}))
                target = res.t_start + (i + 1) * (chunk_samples / SR)
                delay = target - time.perf_counter()
                if delay > 0:
                    await asyncio.sleep(delay)

        async def receiver() -> None:
            while True:
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=30)
                except asyncio.TimeoutError:
                    res.errors.append("30s 无消息，疑似流未闭合")
                    return
                except Exception as exc:  # noqa: BLE001
                    res.errors.append(f"连接异常: {exc}")
                    return
                if isinstance(raw, bytes):
                    res.errors.append(f"收到未预期的 binary 帧 ({len(raw)}B)")
                    continue
                msg = json.loads(raw)
                now = time.perf_counter() - res.t_start
                # mode 字段不一定存在（例如 is_final 收尾帧只带 is_final）
                mode = str(msg.get("mode") or "")
                if mode == "turnsense":
                    res.turnsense.append(
                        AsrEvent(now, mode, msg.get("label", ""), False, msg)
                    )
                    if verbose:
                        print(f"[turnsense] {msg}")
                    continue
                ev = AsrEvent(
                    t=now, mode=mode, text=msg.get("text", ""),
                    is_final=bool(msg.get("is_final")), raw=msg,
                )
                res.events.append(ev)
                if mode == "2pass-online" and res.first_partial_at is None:
                    res.first_partial_at = time.perf_counter()
                if mode.startswith("2pass-offline") and res.first_final_at is None:
                    res.first_final_at = time.perf_counter()
                if verbose:
                    print(f"[{mode}] final={ev.is_final} {ev.text!r}")
                if ev.is_final:
                    res.done_at = time.perf_counter()
                    return

        send_task = asyncio.create_task(sender())
        recv_task = asyncio.create_task(receiver())
        await send_task
        try:
            await asyncio.wait_for(recv_task, timeout=45)
        except asyncio.TimeoutError:
            res.errors.append("发送完毕后 45s 内未收到 is_final")
            recv_task.cancel()

    return res


def main() -> None:
    p = argparse.ArgumentParser(description="ASR 服务量测")
    p.add_argument("--url", default=DEFAULT_URL)
    p.add_argument("--wav", default=None, help="16kHz 单/双声道 wav")
    p.add_argument("--pcm16", default=None, help="裸 PCM16 LE 16k mono")
    p.add_argument("--synth-seconds", type=float, default=5.0)
    p.add_argument("--chunk-ms", type=int, default=100)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    x, dur = load_audio(args)
    print(f"连接 {args.url}  音频={dur:.2f}s  块={args.chunk_ms}ms")
    if args.wav is None and args.pcm16 is None:
        print("[提示] 用的是合成信号 —— turnsense 很可能不触发。"
              "要验证 turnsense 请用 --wav 喂真实语音。")
    try:
        res = asyncio.run(run_probe(args.url, x, dur, args.chunk_ms, args.verbose))
    except Exception as exc:  # noqa: BLE001
        print(f"[FATAL] {type(exc).__name__}: {exc}")
        raise SystemExit(1)
    print(res.summary())


if __name__ == "__main__":
    main()
