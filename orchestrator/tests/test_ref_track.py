#!/usr/bin/env python3
"""阶段 1 单元测试：TTS 参考轨 + 声学延迟估计。

核心验证（对应计划里阶段 1 的验收项 ②）：
  **构造已知延迟 D 的合成回声，``AcousticDelayTracker`` 应恢复出 D ±1 采样。**

    python -m orchestrator.tests.test_ref_track
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from orchestrator.audio.ref_track import (  # noqa: E402
    AcousticDelayTracker,
    RefTrack,
)
from orchestrator.clock import SR  # noqa: E402

_failures: list[str] = []


def check(cond: bool, msg: str) -> None:
    if cond:
        print(f"  [OK] {msg}")
    else:
        print(f"  [FAIL] {msg}")
        _failures.append(msg)


def test_ref_track_placement() -> None:
    print("== RefTrack 落位 ==")
    rt = RefTrack()
    rt.set_anchor(ctx_time=100.0, sample_pos=0)

    # 一段 0.5s 的 24k 音频，在 ctx_time=100.2 播放
    n24 = 12000
    t = np.arange(n24, dtype=np.float32) / 24000.0
    pcm = (np.sin(2 * np.pi * 440 * t) * 30000).astype(np.int16)

    chunk = rt.place("r1", 0, pcm, ctx_time=100.2)
    check(chunk.t0 == 3200, f"t0 = 0.2s * 16000 = 3200（得到 {chunk.t0}）")
    check(chunk.t1 - chunk.t0 == 8000, f"0.5s -> 8000 采样（得到 {chunk.t1-chunk.t0}）")

    # 读出来应该非零
    got = rt.read(3200, 8000)
    check(np.abs(got).max() > 0.1, "落位区间读出非零")

    # 落位之前应为零
    before = rt.read(0, 3200)
    check(np.all(before == 0), "落位之前的区间为零")

    # 声学延迟补偿：设 D=1600（100ms）后，落位点应前移
    rt2 = RefTrack()
    rt2.set_anchor(100.0, 0)
    rt2.delay_samples = 1600
    c2 = rt2.place("r1", 0, pcm, ctx_time=100.2)
    check(c2.t0 == 3200 - 1600, f"扣除延迟后 t0 = 1600（得到 {c2.t0}）")


def test_ref_track_truncate() -> None:
    print("== RefTrack 截断（barge-in）==")
    rt = RefTrack()
    rt.set_anchor(100.0, 0)
    n24 = 24000  # 1s
    pcm = (np.sin(2 * np.pi * 440 * np.arange(n24) / 24000.0) * 30000).astype(np.int16)
    rt.place("r1", 0, pcm, ctx_time=100.0)  # 落在 [0, 16000)

    # 用户在 0.5s 处打断：此后的参考应清零
    n_zeroed = rt.truncate("r1", from_ctx_time=100.5)
    check(n_zeroed == 8000, f"清零 8000 采样（得到 {n_zeroed}）")

    head = rt.read(0, 8000)
    tail = rt.read(8000, 8000)
    check(np.abs(head).max() > 0.1, "打断点之前的参考保留")
    check(np.all(tail == 0), "打断点之后的参考被清零")

    # 全清
    rt2 = RefTrack()
    rt2.set_anchor(100.0, 0)
    rt2.place("r2", 0, pcm, ctx_time=100.0)
    rt2.truncate("r2")
    check(np.all(rt2.read(0, 16000) == 0), "truncate(None) 清空整个 response")


def test_delay_tracker() -> None:
    print("== 声学延迟估计（已知 D）==")
    rng = np.random.default_rng(7)

    for true_d in (0, 320, 800, 1600, 3200):
        # 参考：一段类语音的宽带信号
        n = 16384
        ref = rng.standard_normal(n).astype(np.float32) * 0.3
        # 加一点低频让信号更像语音
        t = np.arange(n, dtype=np.float32) / SR
        ref += 0.3 * np.sin(2 * np.pi * 300 * t).astype(np.float32)

        # mic = ref 延迟 true_d 个采样，衰减 0.5，加少量噪声
        mic = np.zeros_like(ref)
        if true_d == 0:
            mic = ref * 0.5
        else:
            mic[true_d:] = ref[:-true_d] * 0.5
        mic += rng.standard_normal(n).astype(np.float32) * 0.01

        tr = AcousticDelayTracker(sr=SR, max_delay_ms=300.0)
        # 多帧喂（模拟流式，中值平滑）
        win = 4096
        for i in range(0, n - win, win):
            tr.estimate(mic[i:i + win], ref[i:i + win])

        got = tr.delay
        ok = abs(got - true_d) <= 1
        check(ok, f"D={true_d} -> 估计 {got}（{'±1 内' if ok else '偏差过大'}）")


def test_delay_tracker_rejects_silence() -> None:
    print("== 延迟估计器拒绝无效输入 ==")
    tr = AcousticDelayTracker(sr=SR)
    n = 4096
    mic = np.random.randn(n).astype(np.float32) * 0.1
    ref = np.zeros(n, dtype=np.float32)

    r = tr.estimate(mic, ref)
    check(r is None, "静音 ref 被拒绝（估出来是垃圾）")
    check(tr.delay == 0, "无效输入不污染 delay")

    # 长度不等
    r2 = tr.estimate(mic, ref[:1000])
    check(r2 is None, "长度不等被拒绝")

    # 太短
    r3 = tr.estimate(mic[:100], ref[:100])
    check(r3 is None, "过短输入被拒绝")


def main() -> None:
    test_ref_track_placement()
    test_ref_track_truncate()
    test_delay_tracker()
    test_delay_tracker_rejects_silence()
    print()
    if _failures:
        print(f"FAILED: {len(_failures)} 项")
        for f in _failures:
            print(f"  - {f}")
        raise SystemExit(1)
    print("全部通过")


if __name__ == "__main__":
    main()
