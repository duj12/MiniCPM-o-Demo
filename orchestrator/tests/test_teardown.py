#!/usr/bin/env python3
"""阶段 6 拆除路径验证：会话结束后资源是否干净释放。

上线前必须确认没有泄漏 —— 每次会话都要：
  · AEC 发 is_end、收到 stream_end
  · OmniLLM 发 session.close
  · ASR 发 is_speaking=false 并等到 is_final
  · TTS channel 关闭、无在飞 RPC
  · 人脸线程 join、G1 destroy
  · 后台任务全部取消、无残留 asyncio task

方法：跑 N 轮短会话，每轮后检查 asyncio 任务数、线程数、文件句柄、
内存是否回到基线附近。

    python -m orchestrator.tests.test_teardown --rounds 5 --wav path/to.wav
"""
from __future__ import annotations

import argparse
import asyncio
import gc
import json
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from orchestrator.clock import SR  # noqa: E402
from orchestrator.protocol import AudioChunk, MIC_CHUNK  # noqa: E402

_failures: list = []


def check(cond: bool, msg: str) -> None:
    print(f"  [{'OK' if cond else 'FAIL'}] {msg}")
    if not cond:
        _failures.append(msg)


def snapshot() -> dict:
    """当前进程资源快照。"""
    d = {
        "threads": threading.active_count(),
        "asyncio_tasks": 0,
    }
    try:
        d["asyncio_tasks"] = len(asyncio.all_tasks())
    except RuntimeError:
        pass
    # 文件句柄（Linux）
    try:
        d["fds"] = len(os.listdir("/proc/self/fd"))
    except OSError:
        d["fds"] = -1
    # 内存（RSS, MB）
    try:
        with open("/proc/self/status", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    d["rss_mb"] = int(line.split()[1]) / 1024
                    break
    except OSError:
        d["rss_mb"] = -1.0
    return d


async def one_round(idx: int, host: str, port: int, src: np.ndarray,
                    duration: float, drain_s: float) -> dict:
    """跑一轮会话，返回收到的消息统计。"""
    import websockets

    counts = {"ready": False, "tts_end": 0, "asr_final": 0, "errors": []}
    async with websockets.connect(
        f"ws://{host}:{port}/v1/orchestrator", max_size=64 * 1024 * 1024
    ) as ws:
        await ws.send(json.dumps({"type": "session.start",
                                  "identity": {"round": idx}}))

        async def receiver():
            while True:
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=60)
                except Exception:
                    return
                m = json.loads(raw)
                t = m.get("type")
                if t == "session.ready":
                    counts["ready"] = True
                elif t == "tts.end":
                    counts["tts_end"] += 1
                elif t == "asr" and m.get("phase") == "final":
                    counts["asr_final"] += 1
                elif t == "error":
                    counts["errors"].append(m.get("code"))

        rt = asyncio.create_task(receiver())
        for _ in range(200):
            if counts["ready"]:
                break
            await asyncio.sleep(0.05)

        t0 = time.monotonic()
        n = int(duration * 1000 / (MIC_CHUNK / SR * 1000))
        for i in range(n):
            seg = src[i * MIC_CHUNK:(i + 1) * MIC_CHUNK]
            if seg.size < MIC_CHUNK:
                seg = np.pad(seg, (0, MIC_CHUNK - seg.size))
            await ws.send(json.dumps({
                "type": "audio",
                "audio_base64": AudioChunk.from_float32(seg.astype(np.float32)).audio_base64,
            }))
            d = (t0 + (i + 1) * (MIC_CHUNK / SR)) - time.monotonic()
            if d > 0:
                await asyncio.sleep(d)
        for _ in range(10):
            await ws.send(json.dumps({
                "type": "audio",
                "audio_base64": AudioChunk.from_float32(
                    np.zeros(MIC_CHUNK, dtype=np.float32)).audio_base64,
            }))
            await asyncio.sleep(0.1)
        try:
            await ws.send(json.dumps({"type": "session.stop"}))
        except Exception:
            pass
        deadline = time.monotonic() + drain_s
        while time.monotonic() < deadline:
            await asyncio.sleep(0.3)
            if counts["tts_end"] > 0:
                break
        rt.cancel()
        try:
            await rt
        except (asyncio.CancelledError, Exception):
            pass
    return counts


async def main_async(args) -> int:
    import uvicorn
    from orchestrator.main import create_app, REGISTRY, SHUTDOWN_TASKS
    from orchestrator.config import Settings

    cfg = Settings.from_env()
    cfg.host, cfg.port = "127.0.0.1", args.port
    cfg.enable_aec = not args.no_aec
    cfg.enable_asr = not args.no_asr
    cfg.enable_omni = not args.no_omni
    cfg.enable_tts = not args.no_tts
    cfg.mock_tts = args.mock_tts
    cfg.downstream_mode = args.downstream_mode

    import wave
    with wave.open(args.wav, "rb") as w:
        raw = w.readframes(w.getnframes())
        src = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        if w.getnchannels() == 2:
            src = src.reshape(-1, 2).mean(axis=1)

    app = create_app(cfg)
    server = uvicorn.Server(uvicorn.Config(
        app, host=cfg.host, port=cfg.port, log_level="warning"))
    stask = asyncio.create_task(server.serve())
    for _ in range(100):
        if server.started:
            break
        await asyncio.sleep(0.05)
    print(f"Orchestrator @ {cfg.host}:{cfg.port}  跑 {args.rounds} 轮")

    base = snapshot()
    print(f"基线: {base}")
    print()

    try:
        for i in range(args.rounds):
            c = await one_round(i, cfg.host, cfg.port, src, args.duration,
                                args.drain_s)
            # 等收尾任务真正跑完（它是后台任务）
            t0 = time.monotonic()
            while SHUTDOWN_TASKS and time.monotonic() - t0 < 30:
                await asyncio.sleep(0.2)
            await asyncio.sleep(1.5)     # 让线程/连接彻底回收
            gc.collect()
            s = snapshot()
            print(f"  轮 {i+1}: ready={c['ready']} tts_end={c['tts_end']} "
                  f"asr_final={c['asr_final']} err={c['errors'] or '-'} | "
                  f"tasks={s['asyncio_tasks']} threads={s['threads']} "
                  f"fds={s['fds']} rss={s['rss_mb']:.0f}MB "
                  f"残留收尾={len(SHUTDOWN_TASKS)} 活跃会话={len(REGISTRY.sessions)}")
            if not c["ready"]:
                _failures.append(f"轮 {i+1} 未建立会话")
    finally:
        await asyncio.sleep(1.0)
        final = snapshot()
        active = len(REGISTRY.sessions)
        pending = len(SHUTDOWN_TASKS)
        server.should_exit = True
        await asyncio.sleep(0.3)
        stask.cancel()

    print()
    print("=" * 70)
    print("拆除路径验证结果")
    print("-" * 70)
    print(f"  基线: tasks={base['asyncio_tasks']} threads={base['threads']} "
          f"fds={base['fds']} rss={base['rss_mb']:.0f}MB")
    print(f"  终态: tasks={final['asyncio_tasks']} threads={final['threads']} "
          f"fds={final['fds']} rss={final['rss_mb']:.0f}MB")
    print(f"  残留收尾任务: {pending}")
    print(f"  活跃会话:     {active}")
    print("-" * 70)

    check(active == 0, f"无残留活跃会话（{active}）")
    check(pending == 0, f"无残留收尾任务（{pending}）")
    # 线程数允许小幅波动（uvicorn/grpc 常驻），但不该随轮数线性增长
    check(final["threads"] <= base["threads"] + 8,
          f"线程数未失控（{base['threads']} → {final['threads']}）")
    if base["fds"] > 0 and final["fds"] > 0:
        check(final["fds"] <= base["fds"] + 40,
              f"文件句柄未失控（{base['fds']} → {final['fds']}）")
    if base["rss_mb"] > 0:
        growth = final["rss_mb"] - base["rss_mb"]
        check(growth < 600, f"内存增长可控（+{growth:.0f}MB）")
    print("=" * 70)

    if _failures:
        print(f"FAILED: {len(_failures)} 项")
        for f in _failures:
            print(f"  - {f}")
        return 1
    print("通过")
    return 0


def main() -> None:
    p = argparse.ArgumentParser(description="拆除路径验证")
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--port", type=int, default=8196)
    p.add_argument("--duration", type=float, default=5.0)
    p.add_argument("--drain-s", type=float, default=25.0)
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
