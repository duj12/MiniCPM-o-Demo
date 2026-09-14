#!/usr/bin/env python3
"""阶段 2/3 端到端验证：模拟浏览器客户端跑通完整链路。

链路：模拟 mic 音频 → Orchestrator → AEC → ASR → downstream → TTS
      → 回送浏览器 → playback 回执 → RefTrack

不需要真实浏览器，不需要真实 TTS（默认用 MockTtsClient）。

    # 本地（不连 ASR/TTS，只验证编排骨架）
    python -m orchestrator.tests.test_e2e_local --no-asr --no-omni --mock-tts

    # 在 106 上（连真实 ASR + AEC）
    python -m orchestrator.tests.test_e2e_local --duration 20
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from orchestrator.clock import SR  # noqa: E402
from orchestrator.config import Settings  # noqa: E402
from orchestrator.protocol import (  # noqa: E402
    AudioChunk, MIC_CHUNK, PlaybackReceiptMsg, to_json,
)


async def run_client(host: str, port: int, duration_s: float,
                     use_real_audio: bool, verbose: bool,
                     drain_s: float = 8.0,
                     wav_path: str = None) -> dict:
    import websockets

    url = f"ws://{host}:{port}/v1/orchestrator"
    stats = {"audio_sent": 0, "tts_frames": 0, "tts_audio_bytes": 0,
             "tts_ends": 0, "tts_starts": 0,
             "asr_partials": 0, "asr_finals": 0, "playbacks_sent": 0,
             "ready": False, "errors": []}
    tts_buffers: dict = {}

    async with websockets.connect(url, max_size=64 * 1024 * 1024) as ws:
        await ws.send(json.dumps({
            "type": "session.start",
            "identity": {"probe": True},
            "seeded_delay_samples": 0,
            "system_prompt": "",
        }, ensure_ascii=False))

        async def receiver():
            while True:
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=45)
                except asyncio.TimeoutError:
                    return
                except Exception:
                    return
                msg = json.loads(raw)
                t = msg.get("type")
                if t == "session.ready":
                    stats["ready"] = True
                    if verbose:
                        print(f"[recv] session.ready id={msg.get('session_id')}")
                elif t == "tts.start":
                    tts_buffers[msg["response_id"]] = []
                    stats["tts_starts"] += 1
                    if verbose:
                        print(f"[recv] tts.start '{msg.get('text','')[:30]}'")
                elif t == "tts.audio":
                    import base64
                    b = base64.b64decode(msg["audio_base64"])
                    tts_buffers.setdefault(msg["response_id"], []).append(b)
                    stats["tts_frames"] += 1
                    stats["tts_audio_bytes"] += len(b)
                elif t == "tts.end":
                    rid = msg["response_id"]
                    total = sum(len(b) for b in tts_buffers.get(rid, []))
                    stats["tts_ends"] += 1
                    if verbose:
                        print(f"[recv] tts.end {rid} {total} bytes "
                              f"({total/2/24000:.2f}s)")
                    # 回播放回执（模拟浏览器）
                    now = time.monotonic()
                    await ws.send(to_json(PlaybackReceiptMsg(
                        response_id=rid, phase="started", ctx_time=now, seq=0)))
                    stats["playbacks_sent"] += 1
                elif t == "tts.cancel":
                    if verbose:
                        print(f"[recv] tts.cancel {msg.get('response_id')}")
                elif t == "asr":
                    if msg.get("phase") == "partial":
                        stats["asr_partials"] += 1
                    else:
                        stats["asr_finals"] += 1
                    if verbose:
                        print(f"[recv] asr.{msg.get('phase')}: "
                              f"{msg.get('text','')[:40]!r}")
                elif t == "error":
                    stats["errors"].append(f"{msg.get('code')}: {msg.get('message')}")
                    print(f"[recv] ERROR {msg}")
                elif t == "face.state" and verbose:
                    print(f"[recv] face.state {msg}")

        recv_task = asyncio.create_task(receiver())
        # 等 ready
        for _ in range(200):
            if stats["ready"]:
                break
            await asyncio.sleep(0.05)
        if not stats["ready"]:
            stats["errors"].append("未收到 session.ready")
            recv_task.cancel()
            return stats

        # 发音频
        # ⚠️ 合成信号（正弦/类语音）**不会**产出 2pass-offline 最终结果 ——
        # ASR 的 turnsense 会把它判为 invalid。要验证 final/TTS 闭环必须
        # 用**真实语音**（--wav）。
        src = None
        if wav_path:
            import wave
            with wave.open(wav_path, "rb") as w:
                assert w.getframerate() == SR, f"需要 {SR}Hz"
                raw = w.readframes(w.getnframes())
                src = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                if w.getnchannels() == 2:
                    src = src.reshape(-1, 2).mean(axis=1)
            if verbose:
                print(f"[send] 用真实语音 {wav_path}（{len(src)/SR:.2f}s）")

        t0 = time.monotonic()
        n_chunks = int(duration_s * 1000 / (MIC_CHUNK / SR * 1000))
        for i in range(n_chunks):
            if src is not None:
                seg = src[i * MIC_CHUNK:(i + 1) * MIC_CHUNK]
                if seg.size < MIC_CHUNK:
                    seg = np.pad(seg, (0, MIC_CHUNK - seg.size))
                x = seg.astype(np.float32)
            elif use_real_audio:
                # 类语音：基频 + 共振峰 + 包络（仅供链路连通性验证）
                t = (np.arange(MIC_CHUNK, dtype=np.float32) + i * MIC_CHUNK) / SR
                f0 = 120 + 40 * np.sin(2 * np.pi * 2.5 * t)
                x = 0.3 * np.sin(2 * np.pi * f0 * t)
                x += 0.15 * np.sin(2 * np.pi * 2 * f0 * t)
                env = 0.5 + 0.5 * np.sin(2 * np.pi * 3.0 * t)
                x = (x * env).astype(np.float32)
            else:
                x = np.zeros(MIC_CHUNK, dtype=np.float32)
            await ws.send(to_json(AudioChunk.from_float32(
                x, t_ms=int((time.monotonic() - t0) * 1000))))
            stats["audio_sent"] += MIC_CHUNK
            target = t0 + (i + 1) * (MIC_CHUNK / SR)
            d = target - time.monotonic()
            if d > 0:
                await asyncio.sleep(d)

        # 尾部静音（让 ASR 的 VAD 闭合最后一段）
        for _ in range(15):
            await ws.send(to_json(AudioChunk.from_float32(
                np.zeros(MIC_CHUNK, dtype=np.float32), t_ms=0)))
            await asyncio.sleep(0.1)

        # 触发服务端收尾（drain：发 ASR is_speaking=false → 等最终结果
        # → 推完 OmniLLM 残余），期间保持接收，让 TTS 音频能回来。
        try:
            await ws.send(json.dumps({"type": "session.stop"}))
        except Exception:
            pass
        # 收尾需要时间，且**长度不定**（OmniLLM 的回复时短时长，TTS 合成
        # 耗时随之变化）。固定 sleep 会偶发"没等到" —— 改为轮询等待
        # 稳定条件：已收到至少一次 tts.end 且 N 秒内无新增帧，或超时。
        deadline = time.monotonic() + drain_s
        stable_since = None
        last_frames = -1
        while time.monotonic() < deadline:
            await asyncio.sleep(0.25)
            if stats["tts_frames"] != last_frames:
                last_frames = stats["tts_frames"]
                stable_since = time.monotonic()
                continue
            # 无新帧满 3s，且已收到 tts.end → 收尾完成
            if (stable_since is not None and
                    time.monotonic() - stable_since > 3.0 and
                    stats.get("tts_ends", 0) > 0):
                break
        recv_task.cancel()
        try:
            await recv_task
        except (asyncio.CancelledError, Exception):
            pass
    return stats


async def main_async(args) -> int:
    import uvicorn
    from orchestrator.main import create_app, REGISTRY

    cfg = Settings.from_env()
    cfg.host = "127.0.0.1"
    cfg.port = args.port
    cfg.enable_aec = not args.no_aec
    cfg.enable_asr = not args.no_asr
    cfg.enable_omni = not args.no_omni
    cfg.enable_tts = not args.no_tts
    cfg.mock_tts = args.mock_tts
    cfg.downstream_mode = args.downstream_mode

    app = create_app(cfg)
    config = uvicorn.Config(app, host=cfg.host, port=cfg.port, log_level="warning")
    server = uvicorn.Server(config)
    server_task = asyncio.create_task(server.serve())

    for _ in range(100):
        if server.started:
            break
        await asyncio.sleep(0.05)
    print(f"Orchestrator 启动于 {cfg.host}:{cfg.port}"
          f"（aec={cfg.enable_aec} asr={cfg.enable_asr} "
          f"omni={cfg.enable_omni} tts={cfg.enable_tts} "
          f"mock_tts={getattr(cfg,'mock_tts',False)} "
          f"downstream={cfg.downstream_mode}）")

    try:
        stats = await run_client(cfg.host, cfg.port, args.duration,
                                 args.real_audio, args.verbose, args.drain_s,
                                 args.wav)
    finally:
        server.should_exit = True
        await asyncio.sleep(0.3)
        server_task.cancel()

    print()
    print("=" * 62)
    print("端到端验证结果")
    print("-" * 62)
    print(f"  session.ready    : {stats['ready']}")
    print(f"  发送音频         : {stats['audio_sent']} 采样 "
          f"({stats['audio_sent']/SR:.1f}s)")
    print(f"  ASR 部分结果     : {stats['asr_partials']}")
    print(f"  ASR 最终结果     : {stats['asr_finals']}")
    print(f"  TTS 音频帧       : {stats['tts_frames']} "
          f"({stats['tts_audio_bytes']/2/24000:.2f}s @24k)")
    print(f"  播放回执已发     : {stats['playbacks_sent']}")
    if stats["errors"]:
        print(f"  错误             : {stats['errors']}")
    print("=" * 62)

    if not stats["ready"]:
        print("FAILED: 会话未建立")
        return 1
    if args.expect_asr and stats["asr_finals"] == 0:
        print("FAILED: 期望有 ASR 最终结果但没有")
        return 1
    if args.expect_tts and stats["tts_frames"] == 0:
        print("FAILED: 期望有 TTS 音频但没有")
        return 1
    print("通过")
    return 0


def main() -> None:
    p = argparse.ArgumentParser(description="Orchestrator 端到端验证")
    p.add_argument("--port", type=int, default=8199)
    p.add_argument("--duration", type=float, default=8.0)
    p.add_argument("--real-audio", action="store_true", default=True)
    p.add_argument("--silence", action="store_true",
                   help="发静音（不发语音）")
    p.add_argument("--no-aec", action="store_true")
    p.add_argument("--no-asr", action="store_true")
    p.add_argument("--no-omni", action="store_true")
    p.add_argument("--no-tts", action="store_true")
    p.add_argument("--mock-tts", action="store_true")
    p.add_argument("--downstream-mode", default="asr",
                   choices=["asr", "omni", "echo", "none"])
    p.add_argument("--expect-asr", action="store_true")
    p.add_argument("--expect-tts", action="store_true")
    p.add_argument("--drain-s", type=float, default=8.0,
                   help="尾部静音后等待 ASR 最终结果 + TTS 的时间")
    p.add_argument("--wav", default=None,
                   help="16kHz 真实语音 wav（验证 ASR final / TTS 闭环**必须**用它）")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()
    if args.silence:
        args.real_audio = False
    raise SystemExit(asyncio.run(main_async(args)))


if __name__ == "__main__":
    main()
