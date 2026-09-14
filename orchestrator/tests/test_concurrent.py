#!/usr/bin/env python3
"""阶段 6 并发验证：多会话并行。

**为什么必须测**：AEC 服务（speech_frontend）**每条 WS 连接 = 一个独占
`StreamInference` 实例**，且文档明确说不能靠 ``--workers > 1`` 横向扩
（会复制多份显存）。所以单进程能支撑几路，只能实测。

同时验证：
  · 多路会话互不干扰（各自的时钟、参考轨、AEC 状态独立）
  · 会话数上升时延迟与丢帧的变化
  · 指标聚合正确

    python -m orchestrator.tests.test_concurrent --n 4 --wav path/to.wav
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from orchestrator.clock import SR  # noqa: E402
from orchestrator.protocol import AudioChunk, MIC_CHUNK  # noqa: E402


async def one_session(idx: int, host: str, port: int, duration: float,
                      src: np.ndarray, drain_s: float,
                      jitter: float) -> Dict:
    """一个会话：连上、推流、收结果、断开。"""
    import websockets

    url = f"ws://{host}:{port}/v1/orchestrator"
    st = {
        "idx": idx, "ready": False, "t_ready": None,
        "tts_frames": 0, "tts_audio_bytes": 0, "tts_ends": 0,
        "asr_partials": 0, "asr_finals": 0, "errors": [],
        "t_start": time.monotonic(),
    }
    # 会话错开启动，避免同步冲击（模拟真实用户陆续进入）
    await asyncio.sleep(jitter * idx)

    try:
        async with websockets.connect(url, max_size=64 * 1024 * 1024) as ws:
            await ws.send(json.dumps({
                "type": "session.start",
                "identity": {"worker": idx},
            }))

            async def receiver():
                while True:
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=60)
                    except Exception:
                        return
                    m = json.loads(raw)
                    t = m.get("type")
                    if t == "session.ready":
                        st["ready"] = True
                        st["t_ready"] = time.monotonic() - st["t_start"]
                    elif t == "tts.audio":
                        st["tts_frames"] += 1
                        st["tts_audio_bytes"] += len(m.get("audio_base64", "")) * 3 // 4
                    elif t == "tts.end":
                        st["tts_ends"] += 1
                    elif t == "asr":
                        if m.get("phase") == "partial":
                            st["asr_partials"] += 1
                        else:
                            st["asr_finals"] += 1
                    elif t == "error":
                        st["errors"].append(f"{m.get('code')}: {m.get('message')}")

            rtask = asyncio.create_task(receiver())
            for _ in range(200):
                if st["ready"]:
                    break
                await asyncio.sleep(0.05)
            if not st["ready"]:
                st["errors"].append("未收到 session.ready")
                rtask.cancel()
                return st

            t0 = time.monotonic()
            n = int(duration * 1000 / (MIC_CHUNK / SR * 1000))
            for i in range(n):
                seg = src[i * MIC_CHUNK:(i + 1) * MIC_CHUNK]
                if seg.size < MIC_CHUNK:
                    seg = np.pad(seg, (0, MIC_CHUNK - seg.size))
                await ws.send(json.dumps({
                    "type": "audio",
                    "audio_base64": AudioChunk.from_float32(
                        seg.astype(np.float32)).audio_base64,
                    "t_ms": int((time.monotonic() - t0) * 1000),
                }))
                target = t0 + (i + 1) * (MIC_CHUNK / SR)
                d = target - time.monotonic()
                if d > 0:
                    await asyncio.sleep(d)
            # 尾静音
            for _ in range(10):
                await ws.send(json.dumps({
                    "type": "audio",
                    "audio_base64": AudioChunk.from_float32(
                        np.zeros(MIC_CHUNK, dtype=np.float32)).audio_base64,
                    "t_ms": 0,
                }))
                await asyncio.sleep(0.1)
            try:
                await ws.send(json.dumps({"type": "session.stop"}))
            except Exception:
                pass
            deadline = time.monotonic() + drain_s
            while time.monotonic() < deadline:
                await asyncio.sleep(0.25)
                if st["tts_ends"] > 0 and st["tts_frames"] > 0:
                    # 已有 TTS 且 2s 无新帧 → 提前结束
                    last = st["tts_frames"]
                    await asyncio.sleep(2.0)
                    if st["tts_frames"] == last:
                        break
            rtask.cancel()
    except Exception as exc:  # noqa: BLE001
        st["errors"].append(f"{type(exc).__name__}: {exc}")
    st["t_total"] = time.monotonic() - st["t_start"]
    return st


async def main_async(args) -> int:
    import uvicorn
    from orchestrator.main import create_app, REGISTRY
    from orchestrator.config import Settings

    cfg = Settings.from_env()
    cfg.host = "127.0.0.1"
    cfg.port = args.port
    cfg.enable_aec = not args.no_aec
    cfg.enable_asr = not args.no_asr
    cfg.enable_omni = not args.no_omni
    cfg.enable_tts = not args.no_tts
    cfg.mock_tts = args.mock_tts
    cfg.downstream_mode = args.downstream_mode

    # 载入音频
    import wave
    with wave.open(args.wav, "rb") as w:
        raw = w.readframes(w.getnframes())
        src = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        if w.getnchannels() == 2:
            src = src.reshape(-1, 2).mean(axis=1)

    app = create_app(cfg)
    config = uvicorn.Config(app, host=cfg.host, port=cfg.port, log_level="warning")
    server = uvicorn.Server(config)
    stask = asyncio.create_task(server.serve())
    for _ in range(100):
        if server.started:
            break
        await asyncio.sleep(0.05)
    print(f"Orchestrator @ {cfg.host}:{cfg.port}  并发 {args.n} 路  "
          f"aec={cfg.enable_aec} asr={cfg.enable_asr} omni={cfg.enable_omni} "
          f"tts={cfg.enable_tts} mock={args.mock_tts}")

    t0 = time.monotonic()
    try:
        results = await asyncio.gather(*[
            one_session(i, cfg.host, cfg.port, args.duration, src,
                        args.drain_s, args.jitter)
            for i in range(args.n)
        ])
    finally:
        # 等指标落账
        await asyncio.sleep(1.0)
        api_stats = REGISTRY.stats()
        server.should_exit = True
        await asyncio.sleep(0.3)
        stask.cancel()
    wall = time.monotonic() - t0

    # ---------------- 报告 ----------------
    print()
    print("=" * 74)
    print(f"并发验证结果   N={args.n}   墙钟 {wall:.1f}s")
    print("-" * 74)
    print(f"{'#':>3} {'ready':>7} {'ASR p/f':>10} {'TTS帧':>7} "
          f"{'TTS音频':>9} {'耗时':>7}  错误")
    ok = 0
    for r in results:
        asr = f"{r['asr_partials']}/{r['asr_finals']}"
        tts_s = r["tts_audio_bytes"] / 2 / 24000
        err = "; ".join(r["errors"][:1]) if r["errors"] else ""
        print(f"{r['idx']:>3} {str(r['ready']):>7} {asr:>10} "
              f"{r['tts_frames']:>7} {tts_s:>8.2f}s "
              f"{r.get('t_total',0):>6.1f}s  {err}")
        if r["ready"] and not r["errors"]:
            ok += 1

    reads = [r["t_ready"] for r in results if r["t_ready"]]
    print("-" * 74)
    if reads:
        print(f"  建连耗时: min={min(reads)*1000:.0f}ms "
              f"median={statistics.median(reads)*1000:.0f}ms "
              f"max={max(reads)*1000:.0f}ms")
    print(f"  指标（服务端）: {api_stats.get('active_sessions')} 活跃")
    print("=" * 74)

    fails = []
    if ok != args.n:
        fails.append(f"仅 {ok}/{args.n} 路正常")
    if not args.no_asr and args.downstream_mode in ("asr",):
        bad = [r["idx"] for r in results if r["asr_finals"] == 0]
        if bad:
            fails.append(f"这些会话无 ASR 最终结果: {bad}")
    if not args.mock_tts and args.downstream_mode == "asr":
        bad = [r["idx"] for r in results if r["tts_frames"] == 0]
        if bad:
            fails.append(f"这些会话无 TTS 音频: {bad}")

    if fails:
        print(f"FAILED: {fails}")
        return 1
    print("全部通过")
    return 0


def main() -> None:
    p = argparse.ArgumentParser(description="并发验证")
    p.add_argument("--n", type=int, default=4, help="并发会话数")
    p.add_argument("--port", type=int, default=8197)
    p.add_argument("--duration", type=float, default=8.0)
    p.add_argument("--drain-s", type=float, default=30.0)
    p.add_argument("--jitter", type=float, default=0.5, help="会话启动间隔(秒)")
    p.add_argument("--wav", required=True)
    p.add_argument("--no-aec", action="store_true")
    p.add_argument("--no-asr", action="store_true")
    p.add_argument("--no-omni", action="store_true")
    p.add_argument("--no-tts", action="store_true")
    p.add_argument("--mock-tts", action="store_true")
    p.add_argument("--downstream-mode", default="asr",
                   choices=["asr", "omni", "echo", "none"])
    args = p.parse_args()
    raise SystemExit(asyncio.run(main_async(args)))


if __name__ == "__main__":
    main()
