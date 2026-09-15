#!/usr/bin/env python3
"""校准通道的端到端测试 —— 模拟浏览器，不需要真设备。

覆盖的正是之前踩过的两个坑：
  1. 前端漏发 ``calibrate.start`` → 服务端一直等（表现为"点了没声音"）
  2. **先发 start 后注册 onmessage** → 服务端 5ms 就回消息，消息丢失
     （表现为"前端 20s 超时"）

以及正常路径：收到校准音频 → 回传合成的麦克风信号（含已知延迟的回声）
→ 服务端算出延迟 → 保存。

    python -m orchestrator.tests.test_calibrate_e2e
"""
from __future__ import annotations

import asyncio
import base64
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

SR = 16000
_failures: list = []


def check(cond: bool, msg: str) -> None:
    print(f"  [{'OK' if cond else 'FAIL'}] {msg}")
    if not cond:
        _failures.append(msg)


async def fake_browser(url: str, true_delay_ms: float,
                       lead_ms: float = 200.0) -> dict:
    """模拟浏览器：**先注册 handler 再发 start**，回传带已知延迟的麦克风。"""
    import websockets

    received_play = False
    async with websockets.connect(url, max_size=64 << 20) as ws:
        result_fut: asyncio.Future = asyncio.get_running_loop().create_future()

        async def recv_loop():
            nonlocal received_play
            while True:
                raw = await ws.recv()
                m = json.loads(raw)
                if m.get("type") == "calibrate.play":
                    received_play = True
                    # 收到音频后：构造"麦克风"= 该音频延迟 lead+true 后 + 噪声
                    audio = np.frombuffer(
                        base64.b64decode(m["audio_base64"]), dtype=np.int16
                    ).astype(np.float32) / 32768.0
                    # 降到 16k
                    n_out = int(len(audio) * SR / m["sample_rate"])
                    pos = np.linspace(0, len(audio) - 1, n_out)
                    i0 = np.floor(pos).astype(int)
                    i1 = np.minimum(i0 + 1, len(audio) - 1)
                    fr = (pos - i0).astype(np.float32)
                    a16 = (audio[i0] * (1 - fr) + audio[i1] * fr).astype(np.float32)

                    lead = int(lead_ms * SR / 1000)
                    delay = int(true_delay_ms * SR / 1000)
                    # 采集时长要覆盖「播放 + 最大搜索延迟」——与服务端
                    # ready() 的判据一致（1.5s 播放 + 1.5s 搜索 + 余量）
                    total = lead + delay + len(a16) + int(1.6 * SR)
                    mic = np.random.randn(total).astype(np.float32) * 0.001
                    s = lead + delay
                    mic[s:s + len(a16)] += a16 * 0.5

                    # 按 100ms 分块回传
                    for i in range(0, len(mic), 1600):
                        chunk = mic[i:i + 1600]
                        if chunk.size < 1600:
                            chunk = np.pad(chunk, (0, 1600 - chunk.size))
                        await ws.send(json.dumps({
                            "type": "calibrate.mic",
                            "audio_base64": base64.b64encode(
                                chunk.astype(np.float32).tobytes()).decode(),
                        }))
                        await asyncio.sleep(0.01)
                elif "ok" in m:
                    if not result_fut.done():
                        result_fut.set_result(m)
                    return

        rtask = asyncio.create_task(recv_loop())
        # handler 已注册，现在才发 start
        await ws.send(json.dumps({"type": "calibrate.start",
                                  "client_key": "e2e_test"}))
        try:
            res = await asyncio.wait_for(result_fut, timeout=25)
        finally:
            rtask.cancel()
        res["_received_play"] = received_play
        return res


async def main_async() -> int:
    import uvicorn
    from orchestrator.main import create_app
    from orchestrator.config import Settings
    from orchestrator import delay_store as ds_mod

    cfg = Settings.from_env()
    cfg.host, cfg.port = "127.0.0.1", 8195
    tmp_store = Path(tempfile.gettempdir()) / "orch_delay_e2e.json"
    if tmp_store.exists():
        tmp_store.unlink()
    cfg.delay_store_path = str(tmp_store)

    app = create_app(cfg)
    server = uvicorn.Server(uvicorn.Config(
        app, host=cfg.host, port=cfg.port, log_level="warning"))
    stask = asyncio.create_task(server.serve())
    for _ in range(100):
        if server.started:
            break
        await asyncio.sleep(0.05)

    url = f"ws://{cfg.host}:{cfg.port}/v1/calibrate"
    print("=" * 66)
    print("校准通道端到端测试（模拟浏览器）")
    print("-" * 66)

    try:
        for true_d in (150.0, 284.0):
            print(f"\n--- 真实延迟 {true_d:.0f}ms ---")
            res = await fake_browser(url, true_d)
            check(res.get("_received_play", False),
                  "收到 calibrate.play（先注册 handler 再发 start）")
            if res.get("ok"):
                got = res["delay_ms"]
                err = got - true_d
                check(abs(err) <= 25,
                      f"测得 {got:.0f}ms（真值 {true_d:.0f}ms，误差 {err:+.0f}ms）")
                check(res.get("saved", False), "结果已保存")
            else:
                check(False, f"校准失败: {res.get('error')}")

        # 异常值应被拒绝
        print(f"\n--- 异常值 900ms（应被拒绝，多半是选错峰）---")
        res = await fake_browser(url, 900.0)
        check(not res.get("ok"),
              f"900ms 被拒绝（{res.get('error', '')[:40]}）")
    finally:
        server.should_exit = True
        await asyncio.sleep(0.3)
        stask.cancel()

    print("=" * 66)
    if _failures:
        print(f"FAILED: {len(_failures)} 项")
        for f in _failures:
            print(f"  - {f}")
        return 1
    print("通过")
    return 0


def main() -> None:
    raise SystemExit(asyncio.run(main_async()))


if __name__ == "__main__":
    main()
