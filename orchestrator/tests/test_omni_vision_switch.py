#!/usr/bin/env python3
"""`ORCH_OMNI_VISION` 开关验证（纯离线，不连 omni）。

关掉后**一个视频帧都不发**给 OmniLLM，音频照旧（音频走 `push_audio`，不在
本文件的断言范围）。两条送帧路径都要锁住：

1. 真机路径 `on_video_face` → `_maybe_feed_omni`（1s 抽帧，复用同一份字节）；
2. replay 路径 `on_video_omni`（只有 replay 工具会走）。

⚠️ 全部用例放在**同一个** `asyncio.run` 里：`OrchestratorSession.__init__`
会建 `asyncio.Queue`，而旧版本 Python（3.7/3.8）在循环关闭后再构造会抛
`There is no current event loop`。

    python -m orchestrator.tests.test_omni_vision_switch
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

_failures: list = []


def check(cond: bool, msg: str) -> None:
    print(f"  [{'OK' if cond else 'FAIL'}] {msg}")
    if not cond:
        _failures.append(msg)


class FakeOmni:
    """最小替身：只记 `offer_frame` 次数。"""

    def __init__(self) -> None:
        self.frames = 0

    def offer_frame(self, jpeg: bytes) -> None:
        self.frames += 1


def _mk(vision: bool):
    from orchestrator.session import OrchestratorSession
    sess = OrchestratorSession("t-omni-vision", {"omni_vision": vision})
    sess.omni = FakeOmni()
    return sess


async def test_vision_on_feeds_frames() -> None:
    print("\n[1] omni_vision=True ⇒ 两条路径都送帧")
    sess = _mk(True)
    sess._maybe_feed_omni(b"jpeg-1", 1000)
    check(sess.omni.frames == 1, f"真机路径送 1 帧（实际 {sess.omni.frames}）")
    await sess.on_video_omni(b"jpeg-2", 2000)
    check(sess.omni.frames == 2, f"replay 路径再送 1 帧（实际 {sess.omni.frames}）")


async def test_vision_off_sends_nothing() -> None:
    print("\n[2] omni_vision=False ⇒ 两条路径都**不**送帧")
    sess = _mk(False)
    for t in (1000, 2000, 3000):
        sess._maybe_feed_omni(b"jpeg", t)
    check(sess.omni.frames == 0, f"真机路径一帧都没发（实际 {sess.omni.frames}）")
    await sess.on_video_omni(b"jpeg", 4000)
    check(sess.omni.frames == 0, f"replay 路径也不发（实际 {sess.omni.frames}）")
    check(sess.stats.get("video_omni_frames", 0) == 0,
          "不计入 video_omni_frames（Omni 根本没看到）")


async def test_omni_none_is_safe() -> None:
    print("\n[3] 没建 Omni 连接 ⇒ 两条路径都不炸")
    from orchestrator.session import OrchestratorSession
    sess = OrchestratorSession("t-omni-vision-none", {"omni_vision": True})
    sess.omni = None
    sess._maybe_feed_omni(b"jpeg", 1000)
    await sess.on_video_omni(b"jpeg", 2000)
    check(True, "没有异常")


async def _amain() -> None:
    await test_vision_on_feeds_frames()
    await test_vision_off_sends_nothing()
    await test_omni_none_is_safe()


def main() -> int:
    asyncio.run(_amain())
    print("=" * 68)
    if _failures:
        print(f"FAILED: {len(_failures)} 项")
        for f in _failures:
            print(f"  - {f}")
        return 1
    print("通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
