#!/usr/bin/env python3
"""VLM 描述的**帧新鲜度闸**验证（纯离线，不连 omni）。

2026-10-29 起：距最近一次**真正发出去**的视频帧超过 `omni_desc_stale_s`
（默认 5s）就不触发描述。原因见 `config.py` 的实测证据 —— 没有画面时模型
不会说"看不到"，而是按 8 键格式**编造**一份格式完好的快照（SOP 11_1026 /
08_2_1026 的尾巴上各复现一次），然后被当视觉轮推给 Agent。

本脚本锁住四件事：

1. 有新鲜帧 ⇒ 正常触发；
2. 帧过期 / 从没发过帧 ⇒ **不触发**（`trigger_reply` 一次都不调）；
3. 跳过时**推进心跳计时** —— 否则滚动循环每 250ms 重试，日志被刷屏；
4. `full` 被压住后会**补发**（full 只发一次，不能被闸吃掉）；
5. 不报帧时刻的 client（测试替身/老接口）⇒ **不设闸**（fail-open）。

    python -m orchestrator.tests.test_vlm_frame_guard
"""
from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

_failures: list = []


def check(cond: bool, msg: str) -> None:
    print(f"  [{'OK' if cond else 'FAIL'}] {msg}")
    if not cond:
        _failures.append(msg)


class FakeOmni:
    """最小替身：只记触发次数。

    ``reports_frame_clock=False`` 用来模拟**没有** `last_frame_at` 的老
    client（闸必须 fail-open，否则不关心画面的用例会被误闸）。
    """

    def __init__(self, last_frame_at=None, reports_frame_clock=True):
        self.reqs: list = []
        self.last_frame_at = last_frame_at
        if not reports_frame_clock:
            # 直接把属性摘掉 —— getattr 落到哨兵上 = "不报帧时刻"
            del self.last_frame_at

    async def trigger_reply(self, text: str = "", max_new_tokens=None) -> bool:
        self.reqs.append((text, max_new_tokens))
        return True


def _mk_session(stale_s: float, omni: FakeOmni):
    from orchestrator.session import OrchestratorSession
    sess = OrchestratorSession("t-frame-guard",
                               {"omni_describe": True,
                                "omni_desc_stale_s": stale_s})
    sess.omni = omni
    return sess


def _run(sess, stage: str) -> None:
    """跑一次 `_do_describe` 并把它的任务收干净。"""
    async def _go():
        await sess._do_describe(stage)
        if sess._desc_task is not None:
            await asyncio.gather(sess._desc_task, return_exceptions=True)
    asyncio.run(_go())


def test_fresh_frame_fires() -> None:
    print("\n[1] 有新鲜帧 ⇒ 正常触发")
    omni = FakeOmni(last_frame_at=time.monotonic())
    sess = _mk_session(5.0, omni)
    _run(sess, "turn")
    check(len(omni.reqs) == 1, f"触发 1 次（实际 {len(omni.reqs)}）")
    sess._stop_desc_loop()


def test_stale_frame_skips() -> None:
    print("\n[2] 帧过期 ⇒ 不触发，且推进心跳计时")
    omni = FakeOmni(last_frame_at=time.monotonic() - 30.0)
    sess = _mk_session(5.0, omni)
    sess._desc_last_done_at = 0.0
    t0 = time.monotonic()
    _run(sess, "turn")
    check(len(omni.reqs) == 0, f"一次都没触发（实际 {len(omni.reqs)}）")
    check(sess._desc_last_done_at >= t0,
          "推进了 `_desc_last_done_at`（不推进会每 250ms 重试刷屏）")
    # 连续再调几次，也不该触发
    for _ in range(3):
        _run(sess, "turn")
    check(len(omni.reqs) == 0, f"重复调用仍不触发（实际 {len(omni.reqs)}）")
    sess._stop_desc_loop()


def test_never_sent_frame_skips() -> None:
    print("\n[3] 一帧都没发过（last_frame_at=None）⇒ 按过期处理")
    omni = FakeOmni(last_frame_at=None)
    sess = _mk_session(5.0, omni)
    _run(sess, "turn")
    check(len(omni.reqs) == 0, f"不触发（实际 {len(omni.reqs)}）")
    sess._stop_desc_loop()


def test_full_is_deferred_then_backfilled() -> None:
    print("\n[4] full 被压住 ⇒ 有帧后补发")
    omni = FakeOmni(last_frame_at=None)
    sess = _mk_session(5.0, omni)

    _run(sess, "full")                       # 开场还没帧
    check(len(omni.reqs) == 0, "没帧时 full 不触发")
    check(sess._desc_pending_full is True, "记下了「待补发 full」")

    omni.last_frame_at = time.monotonic()    # 摄像头出图了
    _run(sess, "turn")                       # 下一次触发应是 full
    check(len(omni.reqs) == 1, f"补发 1 次（实际 {len(omni.reqs)}）")
    if omni.reqs:
        from orchestrator.omni.describe import FULL_INSTRUCTION
        check(omni.reqs[0][0] == FULL_INSTRUCTION,
              "补发的是**起点全量**指令，不是每轮指令")
    check(sess._desc_pending_full is False, "补发后清掉待补标记")
    sess._stop_desc_loop()


def test_fail_open_without_frame_clock() -> None:
    print("\n[5] client 不报帧时刻 ⇒ 不设闸（fail-open）")
    omni = FakeOmni(reports_frame_clock=False)
    sess = _mk_session(5.0, omni)
    _run(sess, "turn")
    check(len(omni.reqs) == 1,
          f"照常触发（实际 {len(omni.reqs)}）—— 老接口/测试替身不能被闸掉")
    sess._stop_desc_loop()


def test_gate_disabled_by_zero() -> None:
    print("\n[6] `omni_desc_stale_s=0` ⇒ 关闸（回到旧行为）")
    omni = FakeOmni(last_frame_at=time.monotonic() - 300.0)
    sess = _mk_session(0.0, omni)
    _run(sess, "turn")
    check(len(omni.reqs) == 1, f"照常触发（实际 {len(omni.reqs)}）")
    sess._stop_desc_loop()


def main() -> int:
    test_fresh_frame_fires()
    test_stale_frame_skips()
    test_never_sent_frame_skips()
    test_full_is_deferred_then_backfilled()
    test_fail_open_without_frame_clock()
    test_gate_disabled_by_zero()
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
