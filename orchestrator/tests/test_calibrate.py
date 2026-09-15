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
    """模拟一次校准：mic = 双段啁啾（起播偏移 + 声学延迟，衰减）+ 噪声。

    时间语义（与生产一致）：
      · 麦克风流起点 = 采集开始
      · 浏览器在 ``lead`` 处**开始出声**（前端上报为 anchor）
      · 声音经声学路径再延迟 ``true_delay`` 到达麦克风
      → mic 里第 i 段啁啾出现在 ``lead + seg_starts[i] + true_delay``。
    校准算法要报出的是 **true_delay**（扣掉 lead 与段内偏移）。
    """
    cal = CalibrationSession(sr=sr)
    sig24, _starts = cal.start()
    sig16 = cal._signal16          # 16k 参考（与 mic 同域）

    lead = int(playback_lead_ms * sr / 1000.0)
    delay = int(true_delay_ms * sr / 1000.0)
    total = lead + delay + sig16.size + int(0.3 * sr)
    mic = np.random.randn(total).astype(np.float32) * noise
    # 回声：整段信号从 lead+delay 处开始叠加（衰减）
    start = lead + delay
    n = min(sig16.size, total - start)
    mic[start:start + n] += sig16[:n] * gain

    cal.set_play_anchor(lead)
    # 按 100ms 分块喂（模拟流式）
    step = sr // 10
    for i in range(0, mic.size, step):
        cal.feed(mic[i:i + step])
    return cal


def test_quiet_device() -> None:
    """弱信号设备（手机小音量）应**通过** —— 信噪比判据的回归护栏。

    ⚠️ 来自真机日志：iPhone 上啁啾段 rms≈0.003、本底 rms≈0.0005
    （信噪比 ~15dB，回声清晰可辨），但早先的**绝对门槛** rms≥0.004
    把它判成"扬声器声音没进麦克风"，校准根本用不了。
    正确判据是信噪比，与设备音量无关。
    """
    print("== 弱信号设备（信噪比判据）==")
    # ⚠️ lead 必须 ≥ 250ms：本底噪声是拿"起播之前"那段估的，太短就估不到、
    # 信噪比判据会退化回绝对门槛。真机实测锚点是 1.24s（先采一会儿才起播）。
    for label, sig, noise in (("真机 iPhone", 0.0030, 0.0005),
                              ("更小音量", 0.0012, 0.0004)):
        cal = CalibrationSession(sr=16000)
        sig24, _ = cal.start()
        sig16 = cal._signal16
        lead, delay = int(1.24 * 16000), int(0.084 * 16000)
        total = lead + delay + sig16.size + int(0.3 * 16000)
        mic = np.random.randn(total).astype(np.float32) * noise
        n = min(sig16.size, total - (lead + delay))
        # 目标：啁啾段的总 rms 达到 sig
        target = sig * sig - noise * noise
        if target > 0:
            g = float(np.sqrt(target / np.mean(sig16[:n] ** 2)))
            mic[lead + delay:lead + delay + n] += sig16[:n] * g
        cal.set_play_anchor(lead)
        for i in range(0, mic.size, 1600):
            cal.feed(mic[i:i + 1600])
        res = cal.finish()
        got = (res or {}).get("delay_ms")
        ok = bool(res and res.get("ok") and abs(got - 84.0) <= 5)
        check(ok, f"{label}：啁啾 rms={sig:.4f} 本底 rms={noise:.4f} → "
                  f"测得 {got if got is None else round(got,1)}ms"
                  f"（SNR={res.get('snr_db') if res else '-'}dB）")


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
        res = cal.finish()
        if res is None or not res.get("ok"):
            why = (res or {}).get("reason") or "无结果"
            print(f"{true_d:>8}ms {lead:>8}ms {'—':>10}  {'失败':>8}  "
                  f"FAIL（{why[:30]}）")
            _failures.append(f"D={true_d} 失败：{why[:40]}")
            continue
        got = res["delay_ms"]
        err = got - true_d
        # ⚠️ 判据收紧到 ±5ms：实测 D 偏 6ms 回声抑制就从 12.6dB 掉到
        # 2.5dB（见 test_aec_live_fidelity 的扫描表）。±12ms 太松，
        # 那种精度在实际链路里等于没校准。
        ok = abs(err) <= 5 and res.get("segments", 0) == 2
        if not ok:
            _failures.append(f"D={true_d} 误差 {err:.0f}ms")
        print(f"{true_d:>8}ms {lead:>8}ms {got:>9.1f}ms {err:>8.1f}ms  "
              f"{'OK' if ok else 'FAIL'}（段数 {res.get('segments')}，"
              f"段间差 {res.get('spread_ms')}ms，峰比 {res['peak_ratio']}）")

    print()
    test_quiet_device()

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
