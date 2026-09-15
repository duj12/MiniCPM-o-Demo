#!/usr/bin/env python3
"""校准算法验证：用已知延迟的合成回声，检查能否测准。

不连服务、不需要设备 —— 纯算法验证。

    python -m orchestrator.tests.test_calibrate
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from orchestrator.calibrate import (  # noqa: E402
    CAL_SR,
    CalibrationSession,
    make_chirp,
)

_failures: list = []


def check(cond: bool, msg: str) -> None:
    print(f"  [{'OK' if cond else 'FAIL'}] {msg}")
    if not cond:
        _failures.append(msg)


def simulate(true_delay_ms: float, playback_lead_ms: float = 200.0,
             noise: float = 0.002, gain: float = 0.4,
             sr: int = 16000) -> CalibrationSession:
    """模拟一次校准：mic = 啁啾（延迟 + 提前量，衰减）+ 噪声。

    参考轨的语义：服务端在 T0 送音频，浏览器在 T0+lead 开始播放，
    声音经声学路径再延迟 true_delay 到达麦克风。
    → mic 里的啁啾比"服务端送出时刻"晚 (lead + true_delay)。
    校准算法要报出的是 **true_delay**（扣掉 lead）。
    """
    cal = CalibrationSession(sr=sr)
    chirp = cal.chirp24
    if sr == CAL_SR:
        chirp16 = chirp.astype(np.float32) / 32768.0
    else:
        chirp16 = cal.chirp16

    lead = int(playback_lead_ms * sr / 1000.0)
    delay = int(true_delay_ms * sr / 1000.0)
    total = lead + delay + chirp16.size + int(0.3 * sr)
    mic = np.random.randn(total).astype(np.float32) * noise
    # 回声：从 lead+delay 处开始叠加衰减后的啁啾
    start = lead + delay
    mic[start:start + chirp16.size] += chirp16 * gain

    cal.start()
    # 按 100ms 分块喂（模拟流式）
    step = sr // 10
    for i in range(0, mic.size, step):
        cal.feed(mic[i:i + step])
    return cal


def main() -> int:
    print("=" * 66)
    print("校准算法验证（合成回声，已知真值）")
    print("-" * 66)

    # 啁啾自相关应尖锐
    c = make_chirp().astype(np.float32)
    ac = np.correlate(c[:8000], c[:8000], mode="full")
    ac = np.abs(ac[len(ac) // 2:])
    peak_ratio = ac[0] / np.median(ac[1:]) if ac[0] > 0 else 0
    print(f"  啁啾自相关主峰/中位比 = {peak_ratio:.1f}")
    check(peak_ratio > 5, "啁啾自相关足够尖锐（利于定延迟）")

    print()
    cases = [(50, 200), (150, 200), (284, 200), (400, 200), (600, 200),
             (284, 0), (284, 400)]
    print(f"{'真实延迟':>10} {'播放提前':>10} {'测得':>10} {'误差':>9}  判定")
    print("-" * 66)
    for true_d, lead in cases:
        cal = simulate(true_d, lead)
        res = cal.finish(playback_delay_ms=lead)
        if res is None:
            print(f"{true_d:>8}ms {lead:>8}ms {'—':>10}  {'无结果':>8}  FAIL")
            _failures.append(f"D={true_d} 无结果")
            continue
        got = res["delay_ms"]
        err = got - true_d
        ok = abs(err) <= 12 and res["ok"]
        if not ok:
            _failures.append(f"D={true_d} 误差 {err:.0f}ms")
        print(f"{true_d:>8}ms {lead:>8}ms {got:>9.0f}ms {err:>8.0f}ms  "
              f"{'OK' if ok else 'FAIL'}（峰值比 {res['peak_ratio']}）")

    print("=" * 66)
    if _failures:
        print(f"FAILED: {len(_failures)} 项")
        for f in _failures:
            print(f"  - {f}")
        return 1
    print("全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
