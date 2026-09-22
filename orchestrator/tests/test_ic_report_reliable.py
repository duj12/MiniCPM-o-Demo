#!/usr/bin/env python3
"""IC 动作**下发可靠性**验证。

背景（实测踩过的 D16/D18 判题失败）：

  一个 `GREET` 走两条路 —— **播报**（`_dispatch` → `Speak` → TTS）和
  **通知客户端**（`_on_action` → 显示队列 → replay 写 JSONL）。
  两者独立：播报成功了，通知那条路**可能丢**。

  而 `GREET` 这类动作**一瞬就过去**（下一拍就变 `HOLD`），丢了之后：
    · 离线判题只看客户端收到的 `ic` 消息 → 记成失败
    · 喇叭其实响了 → 现象是「有声音、dump 里没 GREET」，极难定位

  早先有两个缺陷叠加：
    ① `_should_report` **在入队之前**就记 `_last_reported_key` —— 丢了不回滚
    ② 显示队列满了就丢，IC 动作与人脸/字幕**抢同一个可丢队列**

本脚本锁住修复后的行为。

    python -m orchestrator.tests.test_ic_report_reliable
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


def _mk_downstream():
    """构造一个不连 IC / 不连 Agent 的 InteractionDownstream。"""
    from orchestrator.interaction.downstream import InteractionDownstream
    d = InteractionDownstream.__new__(InteractionDownstream)
    d.session_id = "unit-test"
    d._last_reported_key = None
    d._last_report_at = 0.0
    d._pending_report = False
    d.REPORT_INTERVAL_S = 10.0
    return d


def test_send_failure_retries() -> None:
    """**送出失败 → 下一拍重试同一条动作**（而不是永远丢掉）。"""
    print("\n[可靠性] 下发失败会重试")
    d = _mk_downstream()
    attempts = []
    fail_first = [True]

    def on_action(atype, sop, text):
        attempts.append(atype)
        if fail_first[0]:
            fail_first[0] = False
            return False            # 第 1 次失败
        return True

    d._on_action = on_action
    for _ in range(5):
        if d._should_report("GREET", "15"):
            ok = d._on_action("GREET", "15", "您好")
            if ok:
                d._mark_reported("GREET", "15")
            else:
                d._pending_report = True

    check(len(attempts) == 2,
          f"失败 1 次 + 成功 1 次 = 共尝试 2 次（实际 {len(attempts)}）")
    check(attempts == ["GREET", "GREET"], "两次尝试的是**同一条**动作")


def test_success_does_not_repeat() -> None:
    """**送出成功后不再重发**（否则 IC/replay 会收到重复的 GREET）。

    ⚠️ 这是修复过程中实际踩到的：第一版把「一瞬就过去的动作」写成
    「永不走心跳节流」，结果成功之后仍每拍重发。
    """
    print("\n[可靠性] 成功之后不重复发")
    d = _mk_downstream()
    attempts = []
    d._on_action = lambda a, s, t: (attempts.append(a), True)[1]

    for _ in range(10):
        if d._should_report("GREET", "15"):
            if d._on_action("GREET", "15", "您好"):
                d._mark_reported("GREET", "15")

    check(len(attempts) == 1, f"只发 1 次（实际 {len(attempts)}）")


def test_steady_state_throttled() -> None:
    """稳态动作仍受心跳节流（不能因为改了重试就退化成每拍全发）。"""
    print("\n[可靠性] 稳态动作仍受心跳节流")
    d = _mk_downstream()
    n = 0
    for _ in range(20):
        if d._should_report("HOLD", None):
            d._mark_reported("HOLD", None)
            n += 1
    check(n == 1, f"20 拍 HOLD 只发 1 次（实际 {n}）")


def test_critical_not_dropped_when_queue_full() -> None:
    """**队列满时，IC 动作不被丢** —— 挤掉普通消息给它让位。"""
    print("\n[可靠性] 队列满时 IC 动作不被丢")
    from orchestrator.protocol import FaceDisplay, IcDisplay
    from orchestrator.session import OrchestratorSession

    s = OrchestratorSession.__new__(OrchestratorSession)
    s.session_id = "unit-test"
    s.stats = {}
    s._display_q = asyncio.Queue(maxsize=5)

    # 塞满：普通消息 + 关键消息
    for i in range(3):
        s._send_display(FaceDisplay(t_ms=i, tracks=[], identity=None, wake=None))
    for _ in range(2):
        s._send_display(IcDisplay(action="HOLD", sop="06"), critical=True)
    check(s._display_q.qsize() == 5, "队列已满（5/5）")

    ok = s._send_display(IcDisplay(action="GREET", sop="15", text="您好"),
                         critical=True)
    check(ok, "满队列下 GREET 仍入队成功")
    check(s.stats.get("display_evicted") == 1,
          "挤掉了 1 条普通消息（display_evicted=1）")

    actions = []
    while not s._display_q.empty():
        m = s._display_q.get_nowait()
        if isinstance(m, IcDisplay):
            actions.append(m.action)
    check("GREET" in actions, f"GREET 在队列里（实际 {actions}）")


def test_normal_message_can_still_drop() -> None:
    """普通消息（人脸/字幕）队列满时**仍然可丢** —— 不能因噎废食。"""
    print("\n[可靠性] 普通消息满时仍可丢（不影响正确性）")
    from orchestrator.protocol import FaceDisplay
    from orchestrator.session import OrchestratorSession

    s = OrchestratorSession.__new__(OrchestratorSession)
    s.session_id = "unit-test"
    s.stats = {}
    s._display_q = asyncio.Queue(maxsize=2)

    for i in range(2):
        s._send_display(FaceDisplay(t_ms=i, tracks=[], identity=None, wake=None))
    ok = s._send_display(FaceDisplay(t_ms=9, tracks=[], identity=None, wake=None))
    check(ok is False, "普通消息入队失败（返回 False）")
    check(s.stats.get("display_dropped") == 1, "计了 display_dropped=1")


def main() -> int:
    print("=" * 68)
    print("IC 动作下发可靠性验证")
    print("-" * 68)
    test_send_failure_retries()
    test_success_does_not_repeat()
    test_steady_state_throttled()
    test_critical_not_dropped_when_queue_full()
    test_normal_message_can_still_drop()
    print("=" * 68)
    if _failures:
        print(f"FAILED: {len(_failures)} 项")
        for f in _failures:
            print("  -", f)
        return 1
    print("通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
