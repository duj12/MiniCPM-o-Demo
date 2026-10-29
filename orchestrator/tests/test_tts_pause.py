#!/usr/bin/env python3
"""软打断：TTS **暂停 / 续播** 的服务端验证（不连服务端）。

IC 的软打断流程是三步：``pause`` → （1s 后判误打断）``speak(prompt=True)``
→ ``resume``。本文件验证编排侧这三步的语义：

  ① ``PauseTts`` **不发 tts.cancel**（那是硬打断），改发 ``tts.pause``；
     整轮内容搬进封存槽，「当前播报」清空
  ② 参考轨按浏览器实测的已播采样数 ``resize``（预测值只在回执超时时兜底）
  ③ ``ResumeTts`` 从暂停点接着播：发 ``tts.resume``，把**剩余 PCM** 重新
     落位到当前时刻
  ④ ``prompt=True`` 的短话术**不能**把封存内容冲掉（否则"误打断"话术一播，
     原来那句就没了）；普通 ``Speak`` 则要把它丢掉
  ⑤ 空转时 pause/resume 必须是**无害的 no-op**

    python -m orchestrator.tests.test_tts_pause
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from orchestrator.actions.executor import ActionExecutor  # noqa: E402
from orchestrator.audio.ref_track import TTS_SR, RefTrack  # noqa: E402
from orchestrator.downstream.interface import (  # noqa: E402
    PauseTts,
    ResumeTts,
    Speak,
)
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
        # ⚠️ 频率**不能**取整百 —— 重采样到 16k 后整周期正弦会有一批采样
        #    恰好落在零点上（500Hz → 每 16 个采样一个零），而下面的
        #    「还有多少内容」是按非零采样数的，会把它们当成静音漏掉。
        freq = 311.7 + 97.3 * self.n
        return (np.sin(2 * np.pi * freq * t) * 20000).astype(np.int16)


class Harness:
    """真 `OrchestratorSession` + 假 TTS + 记账的 send_to_client。"""

    def __init__(self, pause_played_s: float = 2.0,
                 answer_pause: bool = True) -> None:
        self.session = OrchestratorSession(
            "pause", config={"aec_default_delay_ms": 250})
        self.session.ref_track = RefTrack()
        self.session.tts = FakeTts()
        self.session.client_caps = set()      # 不走路 armed 承诺
        self.sent: list = []
        self.pause_played_s = pause_played_s
        self.answer_pause = answer_pause
        self.session.send_to_client = self._send

    async def _send(self, msg) -> None:
        self.sent.append(msg)
        # 模拟浏览器：收到 tts.pause 立刻回报「停在哪了」
        if answer_pause_needed(self, msg):
            await self.session.on_playback_receipt(
                msg.response_id, "paused", 0.0,
                sample_offset=int(self.pause_played_s * 16000))

    def of(self, typ: str) -> list:
        return [m for m in self.sent if getattr(m, "type", "") == typ]

    def ref_len(self, rid: str) -> int:
        """某 response 在参考轨上**还有内容**的时长（16k 采样）。

        ⚠️ 不能数 `_placed` 里的 chunk 跨度：`resize` 只清缓冲、不动 chunk
        记录（区间要留着给之后的 `resize`/打断用）。数跨度会把"已清零"
        也当成"还有"，等于测不出暂停有没有真裁。
        """
        chunks = self.session.ref_track._placed.get(rid)
        if not chunks:
            return 0
        lo = min(c.t0 for c in chunks)
        hi = max(c.t1 for c in chunks)
        seg = self.session.ref_track.buf.read(lo, hi - lo)
        return int(np.count_nonzero(np.abs(seg) > 1e-4))


def answer_pause_needed(h: Harness, msg) -> bool:
    return h.answer_pause and getattr(msg, "type", "") == "tts.pause"


def _mk(pause_played_s: float = 2.0, answer_pause: bool = True) -> Harness:
    return Harness(pause_played_s, answer_pause)


async def test_pause_parks_instead_of_cancelling() -> None:
    print("== 暂停：不发 tts.cancel，改发 tts.pause 并封存 ==")
    h = _mk(pause_played_s=2.0)
    ex = ActionExecutor(playback_delay_ms=200)
    await ex.execute(Speak(text="6"), h.session)
    rid = ex._current_response_id
    check(rid is not None, "先说一句 6s 的")

    await ex.execute(PauseTts(reason="ic_soft_yield"), h.session)
    check(True, "pause 走通了 execute() 的分派")
    check(not h.of("tts.cancel"),
          "**不能**发 tts.cancel（那是硬打断，会把整句扔掉）")
    check(len(h.of("tts.pause")) == 1,
          f"要发一条 tts.pause（收到 {len(h.of('tts.pause'))}）")
    check(ex._current_response_id is None, "暂停后「当前播报」必须清空")
    check(ex._parked is not None and ex._parked["response_id"] == rid,
          "整轮（含 response_id）搬进了封存槽")
    check(ex._parked["played"] == 32000,
          f"封存的是浏览器实测已播量（{ex._parked['played']} 采样）")
    n = h.ref_len(rid)
    check(abs(n - 32000) <= 32,
          f"参考轨保留已播的 2s（实际 {n} 采样 = {n / 16000:.2f}s）—— "
          f"多留的会让 AEC 追不存在的回声，少留的会让真回声没有 farend")


async def test_pause_falls_back_when_no_receipt() -> None:
    print("\n== 暂停：浏览器不回执时用估算兜底（不能卡住） ==")
    h = _mk(answer_pause=False)
    ex = ActionExecutor(playback_delay_ms=200)
    ex.pause_ack_timeout_ms = 50          # 测试里别真等 300ms
    await ex.execute(Speak(text="6"), h.session)
    rid = ex._current_response_id
    await ex.execute(PauseTts(reason="ic_soft_yield"), h.session)
    check(ex._parked is not None, "仍然要封存")
    check(h.ref_len(rid) >= 0, "参考轨仍然有效（估算值可能为 0）")


async def test_resume_continues_from_pause_point() -> None:
    print("\n== 续播：从暂停点接着播 ==")
    h = _mk(pause_played_s=2.0)
    ex = ActionExecutor(playback_delay_ms=200)
    await ex.execute(Speak(text="6"), h.session)
    rid = ex._current_response_id
    parked_pcm_len = int(ex._current_pcm.size)
    await ex.execute(PauseTts(reason="ic_soft_yield"), h.session)

    await ex.execute(ResumeTts(reason="ic_false_yield"), h.session)
    check(True, "resume 走通了 execute() 的分派")
    check(len(h.of("tts.resume")) == 1, "要发一条 tts.resume")
    check(ex._parked is None, "续播后封存槽必须清空")
    check(ex._current_response_id == rid, "续播的是**同一个** response_id")

    n = h.ref_len(rid)
    # 剩余时长换算到 16k 采样（参考轨的基准），别拿 24k 的数直接比
    expect = int(round((parked_pcm_len / TTS_SR - 2.0) * 16000))
    check(abs(n - expect) <= 32,
          f"落位的只剩**剩余**那段（{n} 采样 ≈ {n / 16000:.2f}s，"
          f"期望 {expect / 16000:.2f}s）—— 从 0 重放整句就说明没按暂停点切")


async def test_prompt_speak_keeps_parked() -> None:
    print("\n== 「误打断」话术不能把封存内容冲掉 ==")
    h = _mk(pause_played_s=2.0)
    ex = ActionExecutor(playback_delay_ms=200)
    await ex.execute(Speak(text="6"), h.session)
    rid = ex._current_response_id
    await ex.execute(PauseTts(reason="ic_soft_yield"), h.session)

    await ex.execute(Speak(text="1", prompt=True), h.session)
    check(ex._parked is not None,
          "prompt=True 的 Speak 之后，封存内容**必须还在**")
    check(ex._parked["response_id"] == rid, "封存的还是原来那句")
    check(ex._current_response_id not in (None, rid),
          "话术自己成了「当前播报」（它是独立的一轮）")
    check(not h.of("tts.cancel"), "插入话术也不该发 tts.cancel")


async def test_normal_speak_drops_parked() -> None:
    print("\n== 普通 Speak 取代暂停中的那一轮 ==")
    h = _mk(pause_played_s=2.0)
    ex = ActionExecutor(playback_delay_ms=200)
    await ex.execute(Speak(text="6"), h.session)
    rid = ex._current_response_id
    await ex.execute(PauseTts(reason="ic_soft_yield"), h.session)

    await ex.execute(Speak(text="2"), h.session)
    check(ex._parked is None,
          "普通 Speak 必须丢掉封存内容（否则它会一直挂在内存里）")
    check(rid not in h.session.ref_track._placed,
          "被丢掉的那句要从参考轨上除名（forget）")


async def test_idle_pause_resume_are_noop() -> None:
    print("\n== 空转：没有在播 / 没有封存时都要无害 ==")
    h = _mk()
    ex = ActionExecutor(playback_delay_ms=200)
    check(await ex._pause_tts(h.session, "x") is False,
          "没在播时 pause → False")
    check(await ex._resume_tts(h.session, "x") is False,
          "没封存时 resume → False")
    check(not h.of("tts.pause") and not h.of("tts.resume"),
          "两种空转都**不发**任何消息")
    check(ex._parked is None, "状态没被弄脏")


async def test_drop_parked_on_shutdown() -> None:
    print("\n== 会话收尾：封存内容要能丢弃 ==")
    h = _mk(pause_played_s=2.0)
    ex = ActionExecutor(playback_delay_ms=200)
    await ex.execute(Speak(text="6"), h.session)
    rid = ex._current_response_id
    await ex.execute(PauseTts(reason="ic_soft_yield"), h.session)
    await ex.drop_parked(h.session, reason="session_close")
    check(ex._parked is None, "丢弃后封存槽为空")
    check(rid not in h.session.ref_track._placed, "参考轨上也除名了")
    await ex.drop_parked(h.session, reason="again")
    check(True, "重复丢弃是幂等的")


async def main_async() -> None:
    await test_pause_parks_instead_of_cancelling()
    await test_pause_falls_back_when_no_receipt()
    await test_resume_continues_from_pause_point()
    await test_prompt_speak_keeps_parked()
    await test_normal_speak_drops_parked()
    await test_idle_pause_resume_are_noop()
    await test_drop_parked_on_shutdown()


def main() -> int:
    print("=" * 68)
    print("TTS 暂停 / 续播（软打断）服务端验证")
    print("-" * 68)
    asyncio.run(main_async())
    print()
    if _failures:
        print(f"失败 {len(_failures)} 项:")
        for f in _failures:
            print(f"  · {f}")
        return 1
    print("全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
