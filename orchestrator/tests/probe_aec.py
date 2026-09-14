#!/usr/bin/env python3
"""阶段 0 量测：AEC 服务（speech_frontend /ws/asr_frontend）的长时流行为。

官方参考客户端只演示了「整段 → is_end → stream_end → 关」的批量用法，
我们需要的是**数分钟连续灌流**。本脚本回答四个问题：

  1. 长连接持续灌流是否可行（会不会中途断流 / 报错）
  2. 首窗输出延迟是否 ~600ms，之后是否每 500ms 一窗
  3. 输出总样本数是否等于输入总样本数（抓「直通分支 20% 时间拉伸」缺陷）
  4. 静音 farend 时输出是否异常（决策 3 的前提）

用法::

    python -m orchestrator.tests.probe_aec --duration 60
    python -m orchestrator.tests.probe_aec --duration 60 --farend-mode silence
    python -m orchestrator.tests.probe_aec --duration 60 --farend-mode omit   # 直通，触发 20% 拉伸

不需要本地的 wav 文件：默认用合成信号（正弦扫频 + 噪声）驱动。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np

# 复用 speech_frontend 的帧编解码（协议唯一权威实现，不要自己写）
_SPEECH_FRONTEND = Path(__file__).resolve().parents[3] / "speech_frontend"
if _SPEECH_FRONTEND.is_dir():
    sys.path.insert(0, str(_SPEECH_FRONTEND))
try:
    from webserver.protocol import (  # type: ignore
        ProtocolError,
        encode_frame,
        pack_arrays,
        parse_result_frame,
    )
except ImportError as exc:  # pragma: no cover
    print(f"[FATAL] 无法从 {_SPEECH_FRONTEND} 导入 webserver.protocol: {exc}")
    print("        需要 speech_frontend 仓库在同一层的 code/ 目录下。")
    raise SystemExit(2)

import websockets  # noqa: E402

SR = 16000
CHUNK = 1600  # 100ms —— 官方客户端 main() 里硬编码的值

DEFAULT_URL = "ws://192.168.88.253:30255/ws/asr_frontend"


@dataclass
class WindowStat:
    """一次 result 帧到达的记录。"""

    idx: int
    t_send: float  # 发出对应 chunk 的时刻
    t_recv: float  # 收到 result 的时刻
    n_samples: int


@dataclass
class ProbeResult:
    url: str
    farend_mode: str
    duration_s: float
    chunks_sent: int = 0
    samples_sent: int = 0
    samples_recv: int = 0
    result_frames: int = 0
    windows: List[WindowStat] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    stream_ended_early: bool = False
    first_result_at: Optional[float] = None
    t_start: float = 0.0

    def summary(self) -> str:
        lines = []
        lines.append("=" * 68)
        lines.append(f"AEC 量测结果  url={self.url}")
        lines.append(f"  farend_mode = {self.farend_mode}   duration = {self.duration_s}s")
        lines.append("-" * 68)
        lines.append(f"  发送 chunk      : {self.chunks_sent}")
        lines.append(f"  发送样本数      : {self.samples_sent}")
        lines.append(f"  收到 result 帧  : {self.result_frames}")
        lines.append(f"  收到样本数      : {self.samples_recv}")

        # 关键断言 1：样本守恒（抓 20% 拉伸）
        if self.samples_sent > 0 and not self.stream_ended_early:
            ratio = self.samples_recv / self.samples_sent
            lines.append(f"  输出/输入 样本比: {ratio:.4f}")
            if abs(ratio - 1.0) < 0.02:
                lines.append("    [OK] 样本守恒 —— 没有时间拉伸")
            elif ratio > 1.1:
                lines.append(
                    f"    [!] 输出多出 {(ratio - 1) * 100:.1f}% —— 疑似直通分支的"
                    " 20% 时间拉伸缺陷"
                )
            else:
                lines.append(f"    [?] 比例异常（{ratio:.3f}），可能是尾部未 flush")

        # 关键断言 2：首窗延迟
        if self.first_result_at is not None:
            first_ms = (self.first_result_at - self.t_start) * 1000
            lines.append(f"  首个 result 延迟: {first_ms:.0f} ms")
            if first_ms < 400:
                lines.append("    [OK] 短于 600ms 窗口 —— 服务端可能用了更小的窗口配置")
            elif 450 <= first_ms <= 900:
                lines.append("    [OK] 符合预期（9600 样点 = 600ms 预热）")
            else:
                lines.append("    [?] 与 600ms 预期不符，需查 WS_MIN_INFER_SEC")

        # 关键断言 3：窗口节奏
        if len(self.windows) >= 3:
            sizes = [w.n_samples for w in self.windows]
            gaps = [
                self.windows[i + 1].t_recv - self.windows[i].t_recv
                for i in range(len(self.windows) - 1)
            ]
            lines.append(f"  窗口样本数      : min={min(sizes)} max={max(sizes)} "
                         f"median={int(statistics.median(sizes))}")
            lines.append(f"  窗口间隔        : median={statistics.median(gaps) * 1000:.0f} ms")
            if statistics.median(gaps) > 0.05:
                lines.append("    [OK] 稳定成窗节奏")
            else:
                lines.append("    [!] 窗口来得过密，检查是否在累积")

        if self.stream_ended_early:
            lines.append("  [!] 服务端在预期结束前发出了 stream_end —— 长时流不受支持")
        if self.errors:
            lines.append(f"  错误 ({len(self.errors)}):")
            for e in self.errors[:5]:
                lines.append(f"    - {e}")
        else:
            lines.append("  错误: 无")
        lines.append("=" * 68)
        return "\n".join(lines)


def synth_chunk(n: int, phase: float, sr: int = SR) -> np.ndarray:
    """合成一段近端信号：低频扫频 + 一点噪声，保证有内容可消。"""
    t = (np.arange(n, dtype=np.float32) + phase) / sr
    sweep = 0.25 * np.sin(2 * np.pi * (200 + 300 * np.sin(2 * np.pi * 0.3 * t)) * t)
    noise = 0.01 * np.random.randn(n).astype(np.float32)
    return (sweep + noise).astype(np.float32)


def synth_farend(n: int, phase: float, sr: int = SR) -> np.ndarray:
    """合成一段远端参考（模拟 TTS 播放）：单一稳定音调。"""
    t = (np.arange(n, dtype=np.float32) + phase) / sr
    return (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


async def run_probe(url: str, duration_s: float, farend_mode: str,
                    verbose: bool) -> ProbeResult:
    """farend_mode: 'synthetic'（合成参考） | 'silence'（静音，模拟空闲）
                     | 'omit'（不传 farend，走直通——应触发 20% 拉伸）"""
    res = ProbeResult(url=url, farend_mode=farend_mode, duration_s=duration_s)

    async with websockets.connect(url, max_size=64 * 1024 * 1024) as ws:
        # 1) hello
        hello = {
            "type": "hello",
            "service": "asr_frontend",
            "connection_id": "probe-aec",
            "options": {
                "enable_speech_enhancement": True,
                "enable_aec": True,
                "enable_speech_separation": False,
            },
        }
        await ws.send(json.dumps(hello))
        ready_raw = await asyncio.wait_for(ws.recv(), timeout=20)
        ready = json.loads(ready_raw)
        if ready.get("type") != "ready":
            raise RuntimeError(f"服务端拒绝 hello: {ready!r}")
        print(f"[ready] {ready}")

        res.t_start = time.perf_counter()
        stop_at = res.t_start + duration_s
        n_chunks = int(duration_s * 1000 / (CHUNK / SR * 1000))

        async def sender() -> None:
            phase = 0.0
            for i in range(n_chunks):
                near = synth_chunk(CHUNK, phase)[None, :]  # (1, T)
                if farend_mode == "omit":
                    arrays = [("nearend", near)]
                elif farend_mode == "silence":
                    arrays = [
                        ("nearend", near),
                        ("farend", np.zeros((1, CHUNK), dtype=np.float32)),
                    ]
                else:
                    arrays = [
                        ("nearend", near),
                        ("farend", synth_farend(CHUNK, phase)[None, :]),
                    ]
                slots, payload = pack_arrays(arrays)
                frame = encode_frame(
                    {
                        "type": "chunk",
                        "is_start": i == 0,
                        "is_end": i == n_chunks - 1,
                        "flush_buffer": False,
                        "sampling_rate": SR,
                        "slots": slots,
                    },
                    payload,
                )
                await ws.send(frame)
                res.chunks_sent += 1
                res.samples_sent += CHUNK
                phase += CHUNK
                # 按实时节奏发送（真流式，不是灌满）
                target = res.t_start + (i + 1) * (CHUNK / SR)
                delay = target - time.perf_counter()
                if delay > 0:
                    await asyncio.sleep(delay)

        async def receiver() -> None:
            while True:
                try:
                    raw = await ws.recv()
                except websockets.ConnectionClosed as exc:
                    res.errors.append(f"接收端连接关闭: {exc}")
                    return
                if isinstance(raw, str):
                    msg = json.loads(raw)
                    mtype = msg.get("type")
                    if mtype == "stream_end":
                        if res.chunks_sent < n_chunks:
                            res.stream_ended_early = True
                        return
                    if mtype == "error":
                        res.errors.append(f"服务端 error: {msg}")
                        return
                    if verbose:
                        print(f"[text] {msg}")
                    continue
                # binary result
                try:
                    segments = parse_result_frame(raw)
                except ProtocolError as exc:
                    res.errors.append(f"解析 result 帧失败: {exc}")
                    continue
                now = time.perf_counter()
                if res.first_result_at is None:
                    res.first_result_at = now
                n = sum(int(np.asarray(s).size) for s in segments)
                res.result_frames += 1
                res.samples_recv += n
                res.windows.append(
                    WindowStat(idx=res.result_frames, t_send=now, t_recv=now, n_samples=n)
                )
                if verbose:
                    print(f"[result {res.result_frames}] n_segments={len(segments)} "
                          f"samples={n}")

        send_task = asyncio.create_task(sender())
        recv_task = asyncio.create_task(receiver())
        await send_task
        # 发完后给它一点时间收尾
        try:
            await asyncio.wait_for(recv_task, timeout=15)
        except asyncio.TimeoutError:
            res.errors.append("发送完毕后 15s 内未收到 stream_end")
            recv_task.cancel()

    return res


def main() -> None:
    p = argparse.ArgumentParser(description="AEC 服务长时流量测")
    p.add_argument("--url", default=DEFAULT_URL)
    p.add_argument("--duration", type=float, default=60.0,
                   help="灌流时长（秒）；建议 ≥60 验证长连接稳定性")
    p.add_argument("--farend-mode", default="synthetic",
                   choices=["synthetic", "silence", "omit"],
                   help="synthetic=合成参考（模拟播放）, silence=静音（模拟空闲）, "
                        "omit=不传（直通，应触发 20%% 拉伸）")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    print(f"连接 {args.url}  时长={args.duration}s  farend={args.farend_mode}")
    t0 = time.perf_counter()
    try:
        res = asyncio.run(
            run_probe(args.url, args.duration, args.farend_mode, args.verbose)
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[FATAL] {type(exc).__name__}: {exc}")
        raise SystemExit(1)
    print(f"\n墙钟耗时 {time.perf_counter() - t0:.1f}s")
    print(res.summary())


if __name__ == "__main__":
    main()
