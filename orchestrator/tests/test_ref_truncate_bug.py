#!/usr/bin/env python3
"""复现 + 回归：``playback.ended`` 把整段参考轨清空。

## 现象（真机实测 s-6d21f3ab5f3c）

用户听到的是「TTS 明明播了一长段，但 reference 回采通道里只剩开头的
三个字，而且被拉得很长」。服务端日志：

    TTS 完成 response=4082d2f8 文本 28 字 音频 5.85s
    诊断: ref推送=152(非零 6, 4%)  写入区间=[137600,231200] D=4000

写入区间 **正好是 5.85s**（TTS 报的长度），但实际非零只有 **0.6s** ——
写进去之后又被清零了。

## 根因

``on_playback_receipt`` 对 ``phase == "ended"`` 也调了
``truncate(response_id, from_sample=clock.now())``。而 truncate 的语义是
「从**当前会话时刻**往后清空」，它假定音频已经播到那儿了。但：

  · TTS 是**整段一次性**送完并整段落位的（executor._speak_inner）
  · ``tts.end`` 到达浏览器时，浏览器**才刚开始播**（还有 200ms 提前量）

于是"当前位置"远在整段音频之前 → 整段被清掉，只剩到达那一刻的一小截。

**播放是排程好的**（WebAudio 按 nextAt 连续排），``ended`` 只表示"音频
已全部交给播放器"，**不代表已经播完**。只有 ``cancelled`` 才是真的没播、
必须立刻清（否则 AEC 会追一个不存在的回声，比不给参考更糟）。

    python -m orchestrator.tests.test_ref_truncate_bug
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from orchestrator.audio.ref_track import RefTrack  # noqa: E402
from orchestrator.clock import SR  # noqa: E402
from orchestrator.session import OrchestratorSession  # noqa: E402

_failures: list = []


def check(cond: bool, msg: str) -> None:
    print(f"  [{'OK' if cond else 'FAIL'}] {msg}")
    if not cond:
        _failures.append(msg)


def _make_session() -> OrchestratorSession:
    s = OrchestratorSession("truncbug", config={"aec_default_delay_ms": 250})
    s.ref_track = RefTrack()
    s.send_to_client = lambda msg: asyncio.sleep(0)   # 丢弃 UI 消息
    return s


def _place_long_tts(session: OrchestratorSession, seconds: float,
                    at_sample: int) -> tuple:
    """按 executor 的做法整段落位一段 TTS 音频。"""
    n24 = int(24000 * seconds)
    t = np.arange(n24, dtype=np.float64) / 24000.0
    pcm = (np.sin(2 * np.pi * 440 * t) * 20000).astype(np.int16)
    session.clock.advance(at_sample)
    session.ref_track.place("r1", 0, pcm, at_sample)
    return session.ref_track.buf.written_span()


def _nonzero_seconds(session: OrchestratorSession) -> float:
    lo, hi = session.ref_track.buf.written_span()
    if lo is None:
        return 0.0
    raw = session.ref_track.buf.read(lo, hi - lo)
    return float(np.count_nonzero(np.abs(raw) > 1e-4)) / SR


async def test_ended_keeps_reference() -> None:
    print("== playback.ended 不应清空参考轨 ==")
    s = _make_session()
    span = _place_long_tts(s, 5.85, 137600)
    print(f"  TTS 报 5.85s，落位写入区间 = "
          f"{(span[1]-span[0])/SR:.2f}s")

    # tts.end 到达时时钟只前进了一点点（浏览器才刚起播）
    s.clock.advance(1600)
    await s.on_playback_receipt("r1", "ended", 0.0, 0)

    kept = _nonzero_seconds(s)
    check(kept > 5.0,
          f"ended 之后参考仍保留 {kept:.2f}s（>5.0s）—— "
          "早先这里只剩 0.6s")


async def test_cancelled_still_truncates() -> None:
    print("== playback.cancelled 仍必须截断（barge-in 关键）==")
    s = _make_session()
    _place_long_tts(s, 5.85, 137600)
    s.clock.advance(1600)
    await s.on_playback_receipt("r1", "cancelled", 0.0, 0)
    kept = _nonzero_seconds(s)
    check(kept < 0.5,
          f"cancelled 之后参考被截断（剩 {kept:.2f}s < 0.5s）—— "
          "否则 AEC 会追一个不存在的回声")


async def main_async() -> int:
    print("=" * 70)
    print("参考轨截断回归（playback.ended 曾经清空整段参考）")
    print("-" * 70)
    await test_ended_keeps_reference()
    print()
    await test_cancelled_still_truncates()
    print("=" * 70)
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
