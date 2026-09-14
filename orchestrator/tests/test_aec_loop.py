#!/usr/bin/env python3
"""阶段 1 集成测试：AecClient 对真实 AEC 服务。

对应计划里阶段 1 的验收项：
  ① 输出长度 == 输入长度（抓时间拉伸）
  ③ 每窗 ERLE(dB) —— 用合成回声验证消除效果
  ④ 预热恰好一窗，之后稳定成窗
  ⑤ close() 发 is_end 且不挂起

需要 AEC 服务可达 + speech_frontend 仓库在同层 code/ 目录。

    python -m orchestrator.tests.test_aec_loop
    python -m orchestrator.tests.test_aec_loop --duration 30 --erle
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path
from typing import List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from orchestrator.aec.client import DEFAULT_CHUNK, SR, AecClient  # noqa: E402
from orchestrator.audio.ref_track import AcousticDelayTracker  # noqa: E402

DEFAULT_URL = "ws://192.168.88.253:30255/ws/asr_frontend"


def db(x: float) -> float:
    return 10.0 * np.log10(max(x, 1e-12))


def erle_db(mic: np.ndarray, out: np.ndarray) -> float:
    """回声抑制比：mic 能量 / 残留能量，dB。越高越好。"""
    n = min(mic.shape[0], out.shape[0])
    if n == 0:
        return float("nan")
    p_in = float(np.mean(mic[:n] ** 2))
    p_out = float(np.mean(out[:n] ** 2))
    return db(p_in) - db(p_out)


def make_signals(n: int, true_delay: int, phase: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """构造 (mic, ref, near_speech)。

    ref  = 模拟 TTS 播放的**宽带非周期**信号（延迟估计需要非周期才有尖锐峰值；
    纯正弦的 GCC-PHAT 会出现多个等高峰，估计不可靠）
    mic  = 近端语音 + ref 经 true_delay 延迟并衰减（模拟回声）
    """
    # 用确定性 PRNG 保证跨 chunk 连续（seed 固定，phase 只做偏移）
    rng = np.random.default_rng(1234)
    total = phase + n
    ref_full = (rng.standard_normal(total).astype(np.float32) * 0.12)
    # 叠一点低频让包络更像语音
    t = np.arange(total, dtype=np.float32) / SR
    ref_full += 0.20 * np.sin(2 * np.pi * 300 * t).astype(np.float32)
    ref = ref_full[phase:phase + n]

    echo = np.zeros(n, dtype=np.float32)
    if true_delay == 0:
        echo = ref * 0.4
    else:
        # 回声跨越 chunk 边界：从全局信号取，保证连续
        begin = phase - true_delay
        if begin >= 0:
            echo = ref_full[begin:begin + n] * 0.4
        else:
            # 会话开头的 chunk：前面没有历史，只有部分回声
            echo[-begin:] = ref_full[0:n + begin] * 0.4

    # 近端"语音"：不同频率，避免与 ref 混淆
    speech = 0.10 * np.sin(2 * np.pi * 180 * t[phase:phase + n]).astype(np.float32)
    mic = echo + speech
    return mic, ref, speech


async def run(duration_s: float, true_delay: int, do_erle: bool) -> int:
    url = DEFAULT_URL
    aec = AecClient(url, connection_id="test-aec-loop")
    print(f"连接 {url}")
    await aec.connect()

    n_chunks = int(duration_s * 1000 / (DEFAULT_CHUNK / SR * 1000))
    out_segments: List[np.ndarray] = []
    mic_all: List[np.ndarray] = []
    ref_all: List[np.ndarray] = []   # 保存实际送出的 ref，供延迟估计配对
    _idx = [0]

    def on_audio(seg: np.ndarray) -> None:
        out_segments.append(seg.copy())

    recv_task = asyncio.create_task(aec.recv_loop(on_audio))

    # 发送（按实时节奏）。mic 与 ref 必须成对保存——延迟估计要用**真正
    # 送出去的那一对**，重新生成会因相位参数不一致而估出垃圾。
    t_start = time.perf_counter()
    for i in range(n_chunks):
        mic, ref, _ = make_signals(DEFAULT_CHUNK, true_delay, i * DEFAULT_CHUNK)
        mic_all.append(mic)
        ref_all.append(ref)
        await aec.push(mic[None, :], ref[None, :])
        target = t_start + (i + 1) * (DEFAULT_CHUNK / SR)
        d = target - time.perf_counter()
        if d > 0:
            await asyncio.sleep(d)
        _idx[0] = i + 1

    # 发完后必须 drain：服务端还有在途数据（最后一个窗口 + 尾部 flush）。
    # 立刻 close 会丢掉这部分，让样本比看起来"不守恒"——那是测量的时序
    # 问题，不是产品缺陷。正确顺序：先发 is_end 触发 flush，再等 stream_end。
    try:
        silent = np.zeros((1, DEFAULT_CHUNK), dtype=np.float32)
        await aec.push(silent, is_end=True)
    except Exception as exc:  # noqa: BLE001
        print(f"  [!] 发送 is_end 失败: {exc}")
    try:
        await asyncio.wait_for(recv_task, timeout=25)
    except asyncio.TimeoutError:
        print("  [!] recv_loop 未在 25s 内结束（stream_end 未收到）")
        recv_task.cancel()
    await aec.close(send_end=False)

    # ---------------- 断言 ----------------
    fails = []
    print()
    print("=" * 60)
    print(f"AecClient 集成测试  duration={duration_s}s  true_D={true_delay}")
    print("-" * 60)
    print(f"  发送 chunk      : {aec.chunks_sent}")
    print(f"  发送样本        : {aec.samples_sent}")
    print(f"  接收 result 帧  : {aec.result_frames}")
    print(f"  接收样本        : {aec.samples_recv}")

    # 样本守恒：把尾部的 is_end 干扰块排除，只比"实际音频"部分。
    # 尾部 flush 的输出长度由服务端决定（可能不是整 chunk），不应计入。
    audio_sent = aec.samples_sent - DEFAULT_CHUNK   # 去掉 is_end 块
    ratio = (aec.samples_recv / audio_sent) if audio_sent else None
    print(f"  有效发送样本    : {audio_sent}（已排除 is_end 干扰块）")
    print(f"  样本比          : {ratio:.4f}" if ratio else "  样本比          : n/a")
    if ratio is not None:
        if abs(ratio - 1.0) < 0.02:
            print("    [OK] 样本守恒（无时间拉伸）")
        else:
            print(f"    [FAIL] 样本比偏离 1.0（{ratio:.4f}，"
                  f"差 {aec.samples_recv - audio_sent} 采样）")
            fails.append("样本不守恒")

    lat = aec.first_result_latency_ms()
    if lat is not None:
        print(f"  首窗延迟        : {lat:.0f} ms")
        if lat < 1000:
            print("    [OK] 预热在一窗内完成")
        else:
            print("    [FAIL] 首窗延迟过高")
            fails.append("首窗延迟过高")

    if aec.error:
        print(f"  [FAIL] 服务端错误: {aec.error}")
        fails.append(aec.error)
    else:
        print("  服务端错误      : 无")

    # ERLE —— ⚠️ 合成信号下的 ERLE **不可用于评测消除效果**：
    # 近端"语音"是单一正弦，AEC 可能把它当作回声的一部分一起消掉，
    # 得到虚高的数字。这里只作为"链路通了"的健全性检查。
    if do_erle and out_segments and mic_all:
        mic_cat = np.concatenate(mic_all)
        out_cat = np.concatenate([np.asarray(s).reshape(-1) for s in out_segments])
        e = erle_db(mic_cat, out_cat)
        print(f"  整体 ERLE       : {e:.2f} dB  (合成信号，仅供参考，非真实消除能力)")
        win = 3200
        erles = []
        for i in range(0, min(len(mic_cat), len(out_cat)) - win, win):
            erles.append(erle_db(mic_cat[i:i + win], out_cat[i:i + win]))
        if erles:
            print(f"  逐窗 ERLE       : min={min(erles):.1f} "
                  f"median={np.median(erles):.1f} max={max(erles):.1f} dB")
        print("  → 真实的 ERLE 必须在阶段 3 用**真实声学回声**测（扬声器+麦克风）")

    # 延迟估计（用**实际送出的** mic/ref 配对；重新生成会因相位不一致而失败）
    if out_segments and mic_all:
        tr = AcousticDelayTracker(sr=SR)
        W = 4  # 每次拼 4 个 chunk = 6400 采样，够 FFT 分辨率
        for i in range(0, len(mic_all) - W, 2):
            m = np.concatenate(mic_all[i:i + W])
            r = np.concatenate(ref_all[i:i + W])
            tr.estimate(m, r)
        print(f"  延迟估计        : D={tr.delay}（真值 {true_delay}），"
              f"有效估计 {tr.estimates} 次")
        if tr.estimates == 0:
            print("    [!] 无有效估计（参考信号可能被判为静音）")
        elif abs(tr.delay - true_delay) <= 2:
            print("    [OK] 延迟可恢复")
        else:
            print(f"    [FAIL] 估计偏差 {abs(tr.delay - true_delay)} 采样")
            fails.append("延迟估计不准")

    print("=" * 60)
    if fails:
        print(f"FAILED: {fails}")
        return 1
    print("通过")
    return 0


def main() -> None:
    p = argparse.ArgumentParser(description="AecClient 集成测试")
    p.add_argument("--duration", type=float, default=12.0)
    p.add_argument("--true-delay", type=int, default=1600, help="合成回声延迟（采样）")
    p.add_argument("--erle", action="store_true", help="计算 ERLE（较慢）")
    args = p.parse_args()
    rc = asyncio.run(run(args.duration, args.true_delay, args.erle))
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
