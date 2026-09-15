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
    

    # 一段 0.5s 的 24k 音频，在 ctx_time=100.2 播放
    n24 = 12000
    t = np.arange(n24, dtype=np.float32) / 24000.0
    pcm = (np.sin(2 * np.pi * 440 * t) * 30000).astype(np.int16)

    chunk = rt.place("r1", 0, pcm, 3200)
    check(chunk.t0 == 3200, f"t0 = 0.2s * 16000 = 3200（得到 {chunk.t0}）")
    check(chunk.t1 - chunk.t0 == 8000, f"0.5s -> 8000 采样（得到 {chunk.t1-chunk.t0}）")

    # 读出来应该非零
    got = rt.read(3200, 8000)
    check(np.abs(got).max() > 0.1, "落位区间读出非零")

    # 落位之前应为零
    before = rt.read(0, 3200)
    check(np.all(before == 0), "落位之前的区间为零")

    # 声学延迟补偿在 **read()** 里做（place 只记录原始落位）：
    # ref[t] = raw[t - D]，即把原始轨后移 D，使其与 mic 里的回声对齐。
    rt2 = RefTrack()
    rt2.delay_samples = 1600
    c2 = rt2.place("r1", 0, pcm, 3200)
    check(c2.t0 == 3200, f"place 记录原始落位 = 3200（得到 {c2.t0}）")
    # 补偿后：在 t_play + D 处能读到
    check(np.abs(rt2.read(3200 + 1600, 800)).max() > 0.1,
          "read(t_play+D) 有值 —— 补偿方向正确")
    # 未补偿的 raw 在 t_play 处仍有值（延迟估计要用）
    check(np.abs(rt2.read_raw(3200, 800)).max() > 0.1,
          "read_raw(t_play) 有值 —— 供延迟估计使用")


def test_ref_track_scale() -> None:
    """参考轨必须与麦克风**同量纲**（[-1,1]）—— 回归护栏。

    ⚠️ TTS 服务返回 int16 PCM（实测幅度 ±18000），而麦克风是 [-1,1]。
    早先 ``place()`` 把 int16 量纲原样存进轨里，送进 AEC 的 farend 比
    nearend 大约 **32768 倍** —— 模型看到的参考大了 5 个数量级，回声估计
    被压到 0，回声完全不消。报告里"farend 峰值 rms=4848"那条看似正常的
    记录就是现场（4848 是 int16 量纲，而 mic 只有 0.01~0.1）。
    """
    print("== 参考轨量纲（必须与 mic 同量纲）==")
    rt = RefTrack()
    n24 = 24000
    t = np.arange(n24, dtype=np.float32) / 24000.0
    pcm = (np.sin(2 * np.pi * 440 * t) * 30000).astype(np.int16)
    rt.place("r1", 0, pcm, 0)
    ref = rt.read(0, 8000)
    peak = float(np.abs(ref).max())
    check(0.5 < peak <= 1.0,
          f"int16 输入被归一化（峰值 {peak:.4f}，应 ~0.92 而非 ~30000）")

    # float32 但仍是 int16 量纲（调用方 astype 了但没归一化）也要处理
    rt2 = RefTrack()
    rt2.place("r2", 0, (np.sin(2 * np.pi * 440 * t) * 30000).astype(np.float32), 0)
    peak2 = float(np.abs(rt2.read(0, 8000)).max())
    check(0.5 < peak2 <= 1.0, f"float32(int16 量纲) 也被归一化（峰值 {peak2:.4f}）")

    # 已归一化的 float32 不能被二次缩放
    rt3 = RefTrack()
    rt3.place("r3", 0, (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32), 0)
    peak3 = float(np.abs(rt3.read(0, 8000)).max())
    check(abs(peak3 - 0.5) < 0.05,
          f"已归一化输入保持不变（峰值 {peak3:.4f}，应 ~0.5）")


def test_ref_track_truncate() -> None:
    print("== RefTrack 截断（barge-in）==")
    rt = RefTrack()
    n24 = 24000  # 1s
    pcm = (np.sin(2 * np.pi * 440 * np.arange(n24) / 24000.0) * 30000).astype(np.int16)
    rt.place("r1", 0, pcm, 0)  # 落在 [0, 16000)

    # 用户在 0.5s 处打断：此后的参考应清零
    n_zeroed = rt.truncate("r1", from_sample=8000)
    check(n_zeroed == 8000, f"清零 8000 采样（得到 {n_zeroed}）")

    head = rt.read(0, 8000)
    tail = rt.read(8000, 8000)
    check(np.abs(head).max() > 0.1, "打断点之前的参考保留")
    check(np.all(tail == 0), "打断点之后的参考被清零")

    # 全清
    rt2 = RefTrack()
    
    rt2.place("r2", 0, pcm, 0)
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


def test_delay_tracker_production_window() -> None:
    """生产链路的窗口几何：``ref`` 比 ``mic`` 长 ``offset``（回归护栏）。

    ⚠️ 这正是线上 `_maybe_update_delay` 的调用形态（读
    ``WINDOW + dmax`` 个参考采样喂 ``WINDOW`` 个 mic 采样）。早先
    ``estimate()`` 要求两者**等长**，于是每次调用都在形状校验处 return
    None —— 自适应延迟估计整条路径从未生效，``D`` 永远停在 seed 值。
    这个用例在那版代码上必然全部失败。

    同时验证 ``n_fft`` 足够大：按 ``2*len(mic)`` 算会循环卷绕，把真峰折到
    错误的 lag 上。
    """
    print("== 声学延迟估计（生产窗口几何 ref = mic + offset）==")
    rng = np.random.default_rng(11)

    WINDOW = SR              # 1s，与 session._maybe_update_delay 一致
    OFFSET = int(0.3 * SR)   # 4800 = 默认 max_delay（300ms）

    # D ∈ {20, 84, 284} ms —— 20ms 是模型容忍窗量级，284ms 是报告里的
    # 真机值，84ms 是"扣掉 200ms 播放提前量后的残差"（假说）
    # 造一段"整轨"信号，再按文档约定切窗：
    #   ref_earlier[i] = full[t0 + i]                 （长 WINDOW + OFFSET）
    #   mic[j]         = full[t0 + OFFSET - D + j]    （长 WINDOW）
    # 于是 mic[j] == ref_earlier[j + OFFSET - D]，与约定一致。
    N_WIN = 4
    for true_d in (320, 1344, 4544):
        n = WINDOW + OFFSET + (N_WIN + 1) * WINDOW
        full = rng.standard_normal(n).astype(np.float32) * 0.3
        t = np.arange(n, dtype=np.float32) / SR
        full += 0.3 * np.sin(2 * np.pi * 300 * t).astype(np.float32)

        tr = AcousticDelayTracker(sr=SR, max_delay_ms=300.0)
        for w in range(N_WIN):
            t0 = w * WINDOW
            r = full[t0: t0 + WINDOW + OFFSET]
            m = full[t0 + OFFSET - true_d: t0 + OFFSET - true_d + WINDOW]
            assert r.shape[0] == m.shape[0] + OFFSET, \
                f"窗口几何错误：mic={m.shape[0]} ref={r.shape[0]}"
            tr.estimate(m, r, offset=OFFSET)

        got = tr.delay
        err = got - true_d
        ok = abs(err) <= 2 and tr.estimates == N_WIN
        check(ok, f"D={true_d} 采样（{true_d/SR*1000:.0f}ms）-> "
                 f"估计 {got}（误差 {err:+d} 采样，"
                 f"{tr.estimates}/{N_WIN} 次成功）")

    # offset 不匹配时必须**报出原因**而不是静默 None
    tr = AcousticDelayTracker(sr=SR, max_delay_ms=300.0)
    m = np.random.randn(WINDOW).astype(np.float32) * 0.1
    r_wrong = np.random.randn(WINDOW).astype(np.float32) * 0.1
    res = tr.estimate(m, r_wrong, offset=OFFSET)
    check(res is None and tr.last_reason is not None,
          f"形状不符被拒绝且给出原因（{tr.last_reason}）")


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
    test_ref_track_scale()
    test_ref_track_truncate()
    test_delay_tracker()
    test_delay_tracker_production_window()
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
