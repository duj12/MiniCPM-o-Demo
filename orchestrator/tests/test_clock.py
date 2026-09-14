#!/usr/bin/env python3
"""阶段 1 单元测试：时钟、环形缓冲、有状态重采样。

不依赖任何外部服务，纯本地跑：

    python -m orchestrator.tests.test_clock
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from orchestrator.audio.resample import StatefulResampler  # noqa: E402
from orchestrator.clock import SR, AudioFrame, SampleClock, TrackBuffer  # noqa: E402

_failures: list[str] = []


def check(cond: bool, msg: str) -> None:
    if cond:
        print(f"  [OK] {msg}")
    else:
        print(f"  [FAIL] {msg}")
        _failures.append(msg)


def test_clock() -> None:
    print("== SampleClock ==")
    c = SampleClock()
    check(c.now() == 0, "初始为 0")

    c.advance(1600)
    check(c.now() == 1600, "advance 推进正确")
    check(abs(c.seconds() - 0.1) < 1e-9, "seconds() = 0.1")

    x = np.zeros((1, 1600), dtype=np.float32)
    f = c.frame_of(x)
    check(f.t0 == 1600 and f.t1 == 3200, "frame_of 返回正确的 t0/t1")
    check(c.now() == 3200, "frame_of 会推进时钟")

    s = c.silence_frame(800)
    check(s.t0 == 3200 and s.n_samples == 800, "silence_frame 正确")
    check(np.all(s.data == 0), "silence_frame 是全零")

    # AudioFrame 校验
    try:
        AudioFrame(t0=0, data=np.zeros(10, dtype=np.float32))  # 1-D
        check(False, "AudioFrame 应拒绝 1-D 数据")
    except ValueError:
        check(True, "AudioFrame 拒绝 1-D 数据")

    try:
        AudioFrame(t0=0, data=np.zeros((1, 10), dtype=np.float64))
        check(False, "AudioFrame 应拒绝非 float32")
    except ValueError:
        check(True, "AudioFrame 拒绝非 float32")


def test_track_buffer() -> None:
    print("== TrackBuffer ==")
    tb = TrackBuffer(1000)

    r = tb.read(0, 10)
    check(np.all(r == 0), "未写入区域读出为 0")

    tb.write(0, np.ones(100, dtype=np.float32))
    check(np.all(tb.read(0, 100) == 1), "写后读出正确")
    check(np.all(tb.read(100, 50) == 0), "写入区间之外为 0")

    # 部分重叠读
    got = tb.read(50, 100)
    check(np.all(got[:50] == 1) and np.all(got[50:] == 0), "部分重叠读正确")

    # 回绕
    tb2 = TrackBuffer(100)
    tb2.write(80, np.full(40, 7.0, dtype=np.float32))
    got = tb2.read(80, 40)
    check(np.all(got == 7), "跨越 capacity 边界的写入/读取正确")

    # zero_range（取消播放时截断 ref）
    tb3 = TrackBuffer(1000)
    tb3.write(0, np.ones(200, dtype=np.float32))
    tb3.zero_range(100, 200)
    check(np.all(tb3.read(0, 100) == 1) and np.all(tb3.read(100, 100) == 0),
          "zero_range 只清零指定区间")

    # 超长写入：保留最后 capacity 个
    tb4 = TrackBuffer(50)
    tb4.write(0, np.arange(120, dtype=np.float32))
    got = tb4.read(70, 50)
    check(np.allclose(got, np.arange(70, 120)), "超长写入保留尾部")


def test_resampler() -> None:
    print("== StatefulResampler (24000 -> 16000) ==")
    n = 24000
    t = np.arange(n, dtype=np.float32) / 24000.0
    x = np.sin(2 * np.pi * 440 * t).astype(np.float32)

    def expect(k: int) -> float:
        num = k * 3
        idx = num // 2
        frac = (num % 2) / 2.0
        return float(x[idx] * (1 - frac) + x[idx + 1] * frac)

    # 均匀分块
    sr = StatefulResampler(24000, 16000)
    s = np.concatenate([sr.process(x[i:i + 1600]) for i in range(0, n, 1600)])
    check(len(s) == 16000, f"输出长度 16000（得到 {len(s)}）")
    bad = [k for k in range(len(s)) if abs(expect(k) - s[k]) > 1e-5]
    check(not bad, f"均匀分块逐样本一致（{len(bad)} 个偏差）")

    # 不规则分块（模拟网络抖动）
    rng = np.random.default_rng(1)
    sr2 = StatefulResampler(24000, 16000)
    outs, i = [], 0
    while i < n:
        st = int(rng.integers(1, 3000))
        outs.append(sr2.process(x[i:i + st]))
        i += st
    s2 = np.concatenate(outs)
    bad2 = [k for k in range(len(s2)) if abs(expect(k) - s2[k]) > 1e-5]
    check(len(s2) == 16000 and not bad2, "不规则分块逐样本一致")

    # 首块就产不出（块太小）——不应崩，且后续能补齐
    sr3 = StatefulResampler(24000, 16000)
    total = 0
    for i in range(0, n, 1):  # 逐采样喂，最极端
        total += len(sr3.process(x[i:i + 1]))
    check(total == 16000, f"逐采样喂也能产 16000（得到 {total}）")


def main() -> None:
    test_clock()
    test_track_buffer()
    test_resampler()
    print()
    if _failures:
        print(f"FAILED: {len(_failures)} 项")
        for f in _failures:
            print(f"  - {f}")
        raise SystemExit(1)
    print("全部通过")


if __name__ == "__main__":
    main()
