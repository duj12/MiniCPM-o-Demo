#!/usr/bin/env python3
"""复现 + 回归：AEC 参考轨的时间基准错乱。

**背景**（真实调用序，见 executor._speak_inner / session.on_playback_receipt）：

  1. 浏览器播放首帧 → 发 ``playback.started``，``ctx_time`` 是
     **浏览器 AudioContext 时间**（例如 812.4 秒）
  2. ``session.on_playback_receipt`` → ``ref_track.set_anchor(812.4, clock.now())``
     把 ``ctx_time`` 与**会话采样位置**绑在一起
  3. ``executor._next_ctx_time()`` 返回 ``clock.seconds() + 0.2`` ——
     这是**会话时钟的秒值**（例如 12.7）
  4. 但它被当作 ``ctx_time`` 传给 ``ref_track.place()``，于是
     ``ctx_to_sample()`` 算出 ``anchor_sample + (12.7 - 812.4) * 16000``
     = **约 -1.28e7 的大负数**

后果：参考写入到远离读取区间的位置，``TrackBuffer.read()`` 因两区间
不相交而**恒返回 0** → AEC 的 farend 永远是静音 → 回声完全不消 →
**你自己的 TTS 播报会被 ASR 识别成用户说话**。

    python -m orchestrator.tests.test_ref_track_bug
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from orchestrator.audio.ref_track import RefTrack  # noqa: E402
from orchestrator.clock import SR  # noqa: E402

_failures: list = []


def check(cond: bool, msg: str) -> None:
    print(f"  [{'OK' if cond else 'BUG'}] {msg}")
    if not cond:
        _failures.append(msg)


def make_pcm(n_sec: float = 1.0) -> np.ndarray:
    n = int(24000 * n_sec)
    t = np.arange(n) / 24000.0
    return (np.sin(2 * np.pi * 440 * t) * 20000).astype(np.int16)


def main() -> int:
    print("=" * 70)
    print("AEC 参考轨时间基准检查")
    print("-" * 70)

    pcm = make_pcm(1.0)

    # ---- 场景 A：正常落位（会话位置）----
    rt = RefTrack()
    session_pos = int(12.5 * SR)
    at = session_pos + int(0.2 * SR)          # +200ms 播放提前量
    rt.place("r1", 0, pcm, at)
    lo, hi = rt.buf.written_span()
    print(f"  会话位置             : {session_pos}  (12.5s)")
    print(f"  落位位置 at_sample   : {at}  (+200ms 播放提前)")
    print(f"  实际写入区间         : [{lo}, {hi}]")
    check(lo is not None and abs(lo - at) < 2, f"写入位置 = at_sample（实际 {lo}）")

    got = rt.read(at, 1600)
    peak = float(np.abs(got).max())
    print(f"  在落位处读出峰值     : {peak:.4f}")
    check(peak > 0.01, "read() 读出 TTS 参考（AEC 的 farend 非静音）")

    # 落位之前应是静音（那些时刻还没开始播）
    before = rt.read(session_pos, 1600)
    check(float(np.abs(before).max()) < 1e-6, "落位之前是静音（尚未播放）")

    # ---- 场景 B：声学延迟补偿方向 ----
    rt2 = RefTrack()
    rt2.delay_samples = 1600                    # D = 100ms
    t_play = at
    rt2.place("r1", 0, pcm, t_play)
    # mic 里的回声出现在 t_play + D 之后 → read() 在 t_play + D 处应有值
    got_d = rt2.read(t_play + 1600, 800)
    check(float(np.abs(got_d).max()) > 0.01,
          f"read(t_play+D) 有值（D={rt2.delay_samples}）—— 补偿方向正确")
    # 而在 t_play 处（未补偿的位置）读，应该已经被推走
    got_0 = rt2.read(t_play, 400)
    check(float(np.abs(got_0).max()) < 0.01,
          "read(t_play) 为空 —— 参考已按 D 后移")

    # ---- 场景 C：raw 读取不受补偿影响（延迟估计要用）----
    raw = rt2.read_raw(t_play, 800)
    check(float(np.abs(raw).max()) > 0.01,
          "read_raw(t_play) 有值 —— 未补偿，供延迟估计使用")

    print("=" * 70)
    if _failures:
        print(f"发现 {len(_failures)} 处问题 —— 参考轨不可用，AEC 的 farend 是静音")
        return 1
    print("通过：参考轨落位正确")
    return 0


if __name__ == "__main__":
    sys.exit(main())
