#!/usr/bin/env python3
"""复现 + 回归：**打断后新回复的参考轨乱掉**（barge-in 场景）。

## 现象（用户报告）

上一句 TTS 还没播完就插话，紧接着的新回复语音**回声完全消不掉**、
被 ASR 整段识别成用户说话。

## 根因：四个缺陷叠加

  ① 前端 `PcmPlayer.stop()` 是**空操作**（gain 设 0 又立刻设回 1），
     且 WebAudio 里 `node.start()` 排程过的 buffer **无法取消** ——
     打断后旧句照播；
  ② 前端收到 `tts.start` 时 `nextAt = max(nextAt, now+0.2)`，而 `nextAt`
     还停在**旧句末尾** → 新句被排到旧句之后；
  ③ 服务端把新句落位在"当前时刻 + 提前量"，与②的实际排程不符；
  ④ 下一句开始前**没人打断旧句** —— `_current_response_id` 被直接覆盖，
     旧句的落位记录留在轨上，参考轨与麦克风里的实际回声彻底对不上。

→ mic 里的回声来自**旧句**，farend 写的是**新句**，两者无关，AEC 失效。

本测试覆盖服务端侧能验证的部分：
  · 新句开始时旧句被自动打断（参考轨里旧句未被播出的部分被清）
  · 已播出的部分**保留**（否则那段真实的回声就没有参考了）
  · `truncate` **不误伤**期间插入的其它 response
  · 打断后新句能正常落位、参考轨内容正确

    python -m orchestrator.tests.test_bargein_ref
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from orchestrator.actions.executor import ActionExecutor  # noqa: E402
from orchestrator.audio.ref_track import RefTrack  # noqa: E402
from orchestrator.clock import SR  # noqa: E402
from orchestrator.downstream.interface import Cancel, Speak  # noqa: E402
from orchestrator.session import OrchestratorSession  # noqa: E402

_failures: list = []


def check(cond: bool, msg: str) -> None:
    print(f"  [{'OK' if cond else 'FAIL'}] {msg}")
    if not cond:
        _failures.append(msg)


class FakeTts:
    """按请求的时长返回一段可辨识的音频（不同 response 频率不同）。"""

    def __init__(self) -> None:
        self.n = 0

    async def synthesize(self, text, **kw):
        self.n += 1
        secs = float(text)
        n = int(24000 * secs)
        t = np.arange(n, dtype=np.float64) / 24000.0
        freq = 300 + 200 * self.n          # 每句不同频率，便于区分
        return (np.sin(2 * np.pi * freq * t) * 20000).astype(np.int16)


def _mk_session() -> OrchestratorSession:
    s = OrchestratorSession("bargein", config={"aec_default_delay_ms": 250})
    s.ref_track = RefTrack()
    s.tts = FakeTts()
    s.send_to_client = lambda msg: asyncio.sleep(0)
    return s


def _spec(session: OrchestratorSession, rid: str) -> tuple:
    """某 response 在轨上的 (起点, 终点, 非零采样数)。"""
    chunks = session.ref_track._placed.get(rid)
    if not chunks:
        return (None, None, 0)
    lo = min(c.t0 for c in chunks)
    hi = max(c.t1 for c in chunks)
    seg = session.ref_track.buf.read(lo, hi - lo)
    return (lo, hi, int(np.count_nonzero(np.abs(seg) > 1e-4)))


async def test_new_speak_interrupts_previous() -> None:
    print("== 新回复开始时应自动打断旧句 ==")
    s = _mk_session()
    ex = ActionExecutor(playback_delay_ms=200)

    # 第一句：6s 长（大部分还没播）
    await ex.execute(Speak(text="6"), s)
    rid1 = ex._current_response_id
    lo1, hi1, nz1 = _spec(s, rid1)
    print(f"  第 1 句：落位 [{lo1},{hi1}) = {(hi1-lo1)/SR:.1f}s，非零 {nz1/SR:.1f}s")
    check(nz1 / SR > 5.0, f"第 1 句落位完整（{nz1/SR:.1f}s）")

    # 播了 1s 后用户插话 → 新回复
    s.clock.advance(int(1.0 * SR))
    await ex.execute(Speak(text="3"), s)
    rid2 = ex._current_response_id

    check(rid1 != rid2, "新回复用了新的 response_id")
    lo1b, hi1b, nz1b = _spec(s, rid1)
    print(f"  打断后第 1 句：非零 {nz1b/SR:.1f}s（截断点应在起播点附近）")
    # 第一句应被清掉：因为它刚落位、还没起播（started_at 在未来）
    check(nz1b / SR < 1.0,
          f"第 1 句被清空（剩 {nz1b/SR:.1f}s）—— 未播出的部分不该留在轨上")

    lo2, hi2, nz2 = _spec(s, rid2)
    print(f"  第 2 句：落位 [{lo2},{hi2}) = {(hi2-lo2)/SR:.1f}s，非零 {nz2/SR:.1f}s")
    check(nz2 / SR > 2.5, f"第 2 句参考完整（{nz2/SR:.1f}s）")


async def test_played_part_is_kept() -> None:
    print("== 已播出的部分必须保留（否则那段回声没有参考）==")
    s = _mk_session()
    ex = ActionExecutor(playback_delay_ms=200)

    await ex.execute(Speak(text="6"), s)
    rid1 = ex._current_response_id
    started = s.ref_track.started_at(rid1)
    # 模拟"已经播了 2s"：把会话时钟推到起播点之后 2s。
    # 截断点应为 max(now, started) = now → 保留 [started, now) ≈ 2s。
    s.clock._t = started + int(2.0 * SR)

    ex._interrupt_current(s, reason="test")
    seg = s.ref_track.buf.read(started, 6 * SR)
    kept = np.count_nonzero(np.abs(seg) > 1e-4) / SR
    print(f"  播了 2s 后打断：参考保留 {kept:.2f}s"
          f"（截断点 = max(now, started) = {max(s.clock.now(), started)}）")
    check(1.5 < kept < 2.6,
          f"保留约 2s（得到 {kept:.2f}s）—— 已播部分不能清，"
          "否则那段真实回声没有 farend")


async def test_truncate_does_not_touch_other_responses() -> None:
    print("== truncate 不得误伤期间落位的其它 response ==")
    rt = RefTrack()
    n24 = 24000 * 2                     # 每段 2s @24k
    t = np.arange(n24, dtype=np.float64) / 24000.0
    a = (np.sin(2 * np.pi * 300 * t) * 20000).astype(np.int16)
    b = (np.sin(2 * np.pi * 700 * t) * 20000).astype(np.int16)

    rt.place("A", 0, a, 0)              # A: [0, 32000)
    rt.place("B", 0, b, 40000)          # B 之后才落位: [40000, 72000)

    # 取消 A 的**后半段**：从 16000 起清。
    # 早先的实现清 `[16000, A末尾)` 这段**无差别区间**，会把期间插入的
    # 别的 response 一起清掉；现在按 chunk 求交集，只清 A 自己的部分。
    n = rt.truncate("A", from_sample=16000)
    print(f"  A 清了 {n} 采样；B 落在 [40000,72000)，与 A 不重叠")

    a_tail = np.count_nonzero(np.abs(rt.buf.read(16000, 16000)) > 1e-4)
    b_kept = np.count_nonzero(np.abs(rt.buf.read(40000, 32000)) > 1e-4)

    check(n == 16000, f"A 只清了它自己的后半段（{n} 采样 = {n/SR:.1f}s）")
    check(a_tail == 0, "A 的尾部已清空")
    check(b_kept / SR > 1.9,
          f"B **未被误伤**（保留 {b_kept/SR:.1f}s / 2.0s）")


async def main_async() -> int:
    print("=" * 72)
    print("barge-in 参考轨回归（打断后新回复回声消不掉）")
    print("-" * 72)
    await test_new_speak_interrupts_previous()
    print()
    await test_played_part_is_kept()
    print()
    await test_truncate_does_not_touch_other_responses()
    print("=" * 72)
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
