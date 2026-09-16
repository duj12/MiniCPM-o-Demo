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

## 修法（核心是"参考轨跟随**实际播出**，而不是预测"）

  ① 前端 `PcmPlayer.stop()` 真正掐断每个已排程的 `AudioBufferSourceNode`
     （登记 `_sources` 逐个 `stop(0)`）—— WebAudio 里唯一能取消已 start
     源的办法；
  ② 前端 `stop()` 后置 `_cancelled`，下一句 `beginResponse` 从零重排
     `nextAt`，不再排在旧句之后；
  ③ 服务端开新句前 `await _interrupt_current()`：**发 `tts.cancel`
     让浏览器真停** + 截断参考轨（两件事缺一不可）；
  ④ 截断点 = `max(now, started)`，保留 `[started, now)`；
  ⑤ 前端在 `cancelled` 回执里带上 **`sample_offset`（实测播出采样数）**
     —— 服务端据此用 `RefTrack.resize()` 二次校正，把"预测 vs 实际"的
     偏差抹平。这是用户指出的关键点：回采通道只要严格跟随实际播出，
     打断后它自然就是对的。

本测试覆盖：
  · 新句开始时旧句被自动打断，且**真的发了 `tts.cancel`**
  · 已播出的部分**保留**（否则那段真实的回声就没有参考了）
  · 回执（含 `cancelled`）**不**再按 `clock.now()` 截断（会误伤新句）
  · 前端实测 `sample_offset` 能校正参考轨
  · `truncate` **不误伤**期间落位的其它 response

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
    s.sent: list = []
    s.send_to_client = lambda msg: _record(s, msg)
    # ⚠️ **显式**声明没有 playback_anchor 能力位 —— 只有具备该能力位的
    # 客户端（新前端）才会等 armed 承诺。这里不等，于是落位走预测路径，
    # 本文件所有断言的语义与改动前完全一致（零回归）。
    s.client_caps = set()
    return s


async def _record(session: OrchestratorSession, msg) -> None:
    session.sent.append(msg)


def _cancels(session: OrchestratorSession) -> list:
    return [m for m in session.sent if getattr(m, "type", "") == "tts.cancel"]


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

    # ⚠️ **必须真的通知浏览器停** —— 只截断参考轨是不够的：
    # 前端只在收到 tts.cancel 时才停止已排程的音频，否则旧句照旧播完，
    # 而参考轨已清空 → 还在播的音频没有 farend → ASR 全识别到。
    # 真机现象正是"打断后当前语音不停，然后开始识别正在播的音频"。
    cs = _cancels(s)
    check(len(cs) == 1 and cs[0].response_id == rid1,
          f"打断时给浏览器发了 tts.cancel（response={rid1}）—— "
          f"否则旧句照播（收到 {len(cs)} 条）")

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

    await ex._interrupt_current(s, reason="test")
    seg = s.ref_track.buf.read(started, 6 * SR)
    kept = np.count_nonzero(np.abs(seg) > 1e-4) / SR
    print(f"  播了 2s 后打断：参考保留 {kept:.2f}s"
          f"（截断点 = max(now, started) = {max(s.clock.now(), started)}）")
    check(1.5 < kept < 2.6,
          f"保留约 2s（得到 {kept:.2f}s）—— 已播部分不能清，"
          "否则那段真实回声没有 farend")


async def test_receipts_do_not_damage_new_response() -> None:
    """浏览器回执（含 cancelled）**不得**再截断参考轨。

    ⚠️ 回执里按 `clock.now()` 截断踩过两次：
      · `ended` 也截 → 整段参考被清（见 test_ref_truncate_bug）
      · `cancelled` 也截 → 等回执绕一圈回来时**新句往往已开始落位**，
        再切一刀会把新句的参考切掉一截，新回复的回声又对不上
    截断是服务端在动作发生那一刻就该做完的事，不该等网络回执。
    """
    print("== 回执不得截断参考轨（否则新句被误伤）==")
    s = _mk_session()
    ex = ActionExecutor(playback_delay_ms=200)

    await ex.execute(Speak(text="6"), s)
    rid1 = ex._current_response_id
    s.clock.advance(3 * SR)                 # 第 1 句播了 3s
    await ex.execute(Speak(text="3"), s)    # 打断 + 新句
    rid2 = ex._current_response_id

    lo2 = s.ref_track.started_at(rid2)
    before = np.count_nonzero(
        np.abs(s.ref_track.buf.read(lo2, 3 * SR)) > 1e-4) / SR

    # 旧句的三种回执全部到达（真实时序：cancelled 会先到）
    for ph in ("cancelled", "started", "ended"):
        await s.on_playback_receipt(rid1, ph, 0.0, 0)

    after = np.count_nonzero(
        np.abs(s.ref_track.buf.read(lo2, 3 * SR)) > 1e-4) / SR
    print(f"  新句参考：回执前 {before:.2f}s → 回执后 {after:.2f}s")
    check(after >= before - 0.3,
          f"新句参考未被回执截断（{after:.2f}s ≈ {before:.2f}s）")


async def test_observed_offset_corrects_reference() -> None:
    """前端实测的播出量（`sample_offset`）应校正参考轨。

    这是**根本性**的一步（用户指出的）：参考轨应当严格跟随**实际播出**。
    打断那一刻我们只有预测（`clock.now()`），真实播出量要等浏览器回执
    —— 回执带回 `sample_offset` 后，按"落位起点 + 实测播出量"重新对齐。

    用不重叠的时序（新句排在旧句之后）验证：旧句按实测值收敛，新句完好。
    """
    print("== 按前端实测播出量校正参考轨 ==")
    s = _mk_session()
    ex = ActionExecutor(playback_delay_ms=200)

    await ex.execute(Speak(text="6"), s)
    rid1 = ex._current_response_id
    lo1 = s.ref_track.started_at(rid1)

    # 播到 6.5s（旧句 6.2s 已播完）之后用户才开新句 → 时序不重叠
    s.clock.advance(int(6.5 * SR))
    await ex.execute(Speak(text="3"), s)
    rid2 = ex._current_response_id
    lo2 = s.ref_track.started_at(rid2)

    # 前端实测：实际只播了 4.0s（预测是 6.0s）
    await s.on_playback_receipt(rid1, "cancelled", 0.0, 0,
                                sample_offset=int(4.0 * SR))

    k1 = np.count_nonzero(
        np.abs(s.ref_track.buf.read(lo1, 6 * SR)) > 1e-4) / SR
    k2 = np.count_nonzero(
        np.abs(s.ref_track.buf.read(lo2, 3 * SR)) > 1e-4) / SR
    print(f"  旧句保留 {k1:.2f}s（实测播出 4.0s）；新句保留 {k2:.2f}s / 3.0s")
    check(abs(k1 - 4.0) < 0.5,
          f"旧句参考跟随实测播出量（{k1:.2f}s ≈ 4.0s）—— 不是预测的 6.0s")
    check(k2 > 2.5, f"新句未被误伤（{k2:.2f}s）")


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


async def test_armed_anchor_places_reference_exactly() -> None:
    """有能力位时，参考轨必须落在浏览器**承诺**的时刻上（而不是预测）。

    这是本次修复的核心：预测（`clock.now() + 提前量`）的误差含网络往返、
    浏览器主线程抖动、mic 在途积压，**逐句变化** —— 固定常量 D 吸收不了，
    于是"第一句好、后面失效"。承诺（armed）把它变成精确量。
    """
    print("== armed 承诺 → 参考轨精确落位 ==")
    s = _mk_session()
    s.client_caps = {"playback_anchor"}
    ex = ActionExecutor(playback_delay_ms=200)

    # 造一条 ctx → 会话采样的精确映射（真机上由每个 mic 块自带的
    # ctx_time 拟合而来）：sample = (ctx - 100.0) * 16000
    for i in range(5):
        s.clock.record_anchor(100.0 + i * 0.1, i * 1600, 1)
    s.clock.advance(5 * 1600)              # 假装已经收了 0.5s 音频

    # 浏览器承诺：在 ctx=103.7 起播 → 会话采样 (103.7-100)*16000 = 59200
    EXPECT = 59200
    rid_holder: dict = {}

    async def _speak_then_arm():
        await ex.execute(Speak(text="2"), s)

    task = asyncio.create_task(_speak_then_arm())
    # 等执行器发出 tts.start 并注册等待点，再回承诺
    for _ in range(200):
        if s._armed_evt:
            break
        await asyncio.sleep(0)
    rid = next(iter(s._armed_evt))
    rid_holder["rid"] = rid
    await s.on_playback_receipt(rid, "armed", 0.0, start_ctx=103.7, epoch=1)
    await task

    lo = s.ref_track.started_at(rid)
    print(f"  落位起点 = {lo}（承诺换算 {EXPECT}，预测会是 "
          f"{5*1600 + 3200}）")
    check(lo == EXPECT,
          f"参考轨落在承诺时刻（{lo} == {EXPECT}）—— 不是预测值")
    check(s.anchor_source == "ack", f"anchor_source = ack（得到 {s.anchor_source}）")


async def test_missing_armed_falls_back_to_prediction() -> None:
    """有能力位但承诺没到 → 超时退回预测，且**标出来**（不静默降级）。"""
    print("== 承诺超时 → 退回预测（且标记可见）==")
    s = _mk_session()
    s.client_caps = {"playback_anchor"}
    ex = ActionExecutor(playback_delay_ms=200, arm_timeout_ms=50)
    s.clock.advance(1000)

    await ex.execute(Speak(text="0.2"), s)
    rid = ex._current_response_id
    lo = s.ref_track.started_at(rid)
    pred = 1000 + 3200
    print(f"  落位起点 = {lo}（预测 {pred}），anchor_source={s.anchor_source}")
    check(lo == pred, f"退回预测落位（{lo} == {pred}）")
    check(s.anchor_source == "predicted",
          f"降级被标出（anchor_source={s.anchor_source}）—— 不能静默")


async def main_async() -> int:
    print("=" * 72)
    print("barge-in 参考轨回归（打断后新回复回声消不掉）")
    print("-" * 72)
    await test_new_speak_interrupts_previous()
    print()
    await test_played_part_is_kept()
    print()
    await test_receipts_do_not_damage_new_response()
    print()
    await test_observed_offset_corrects_reference()
    print()
    await test_truncate_does_not_touch_other_responses()
    print()
    await test_armed_anchor_places_reference_exactly()
    print()
    await test_missing_armed_falls_back_to_prediction()
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
