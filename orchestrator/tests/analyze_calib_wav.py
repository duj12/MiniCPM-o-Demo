#!/usr/bin/env python3
"""分析前端导出的校准录音，判定"回声为什么没进麦克风"。

## 为什么需要它

真机上校准失败时，指标只能告诉我们"没收到可辨的回声"，但**分不清**是
下面哪一种 —— 三者的修法完全不同：

  ① **校准音没播出来**（扬声器静音 / 路由到听筒 / 被静音开关拦住）
     → mic 里播放时段与播放前本底**没有差异**
  ② **播出来了但太轻**，被环境噪声盖住
     → 播放时段能量略高，但信噪比 < 6dB
  ③ **播了、也够响，但被设备侧处理掉了**
     （iOS 的 play-and-record 音频会话会启用系统级 VPIO，自带 AEC/降噪，
      在声音进到页面之前就把回声消掉了）
     → mic 里播放时段能量**显著低于**本底，或听得到但互相关无峰

用法（在 106 或本机都行，只需要 numpy）::

    python -m orchestrator.tests.analyze_calib_wav \
        --mic calib-mic-1234.wav --chirp calib-chirp-1234.wav \
        --meta calib-meta-1234.json
"""
from __future__ import annotations

import argparse
import json
import sys
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

SR = 16000
_failures: list = []


def load_wav(path: str) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        raw = w.readframes(w.getnframes())
        x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        if w.getnchannels() == 2:
            x = x.reshape(-1, 2).mean(axis=1)
    return x


def db(x: float) -> float:
    return 20.0 * np.log10(max(float(x), 1e-12))


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x ** 2))) if x.size else 0.0


def main() -> int:
    p = argparse.ArgumentParser(description="分析校准录音")
    p.add_argument("--mic", required=True)
    p.add_argument("--chirp")
    p.add_argument("--meta")
    args = p.parse_args()

    mic = load_wav(args.mic)
    meta = {}
    if args.meta and Path(args.meta).is_file():
        meta = json.loads(Path(args.meta).read_text(encoding="utf-8"))
    anchor = meta.get("anchor_samples")
    if anchor is None:
        print("⚠️ 元数据里没有 anchor_samples —— 用前半段当本底估算")
        anchor = len(mic) // 3

    print("=" * 70)
    print("校准录音分析")
    print("-" * 70)
    print(f"  麦克风: {len(mic)/SR:.2f}s  峰值 {np.abs(mic).max():.4f}  "
          f"整体 rms {rms(mic):.5f}")
    print(f"  锚点  : {anchor/SR:.3f}s（播放开始于此刻）")
    if meta.get("result"):
        r = meta["result"]
        print(f"  服务端: ok={r.get('ok')} "
              f"sig_rms={r.get('sig_rms')} noise_rms={r.get('noise_rms')} "
              f"snr={r.get('snr_db')}dB")
    print()

    # 本底 = 播放开始**之前**的麦克风
    pre = mic[:max(0, min(anchor, len(mic)))]
    noise = rms(pre)

    # 播放时段 = 锚点之后、播放信号长度内
    chirp = load_wav(args.chirp) if args.chirp else None
    play_len = len(chirp) if chirp is not None else int(1.7 * SR)
    play = mic[anchor:anchor + play_len]
    sig = rms(play)

    snr = db(sig / noise) if noise > 1e-9 else float("inf")
    print(f"  播放前本底 rms : {noise:.5f}")
    print(f"  播放时段 rms   : {sig:.5f}")
    print(f"  信噪比         : {snr:.1f} dB")
    print()

    # ---- 与播放信号做互相关（最硬的证据：mic 里到底有没有那声音） ----
    peak_ratio = None
    if chirp is not None:
        n = min(len(play), len(chirp))
        if n > 4096:
            a = play[:n] - play[:n].mean()
            b = chirp[:n] - chirp[:n].mean()
            n_fft = 1
            while n_fft < 2 * n:
                n_fft <<= 1
            X = np.fft.rfft(a, n_fft)
            Y = np.fft.rfft(b, n_fft)
            R = X * np.conj(Y)
            mag = np.abs(R)
            mag[mag < 1e-10] = 1e-10
            cc = np.abs(np.fft.irfft(R / mag, n_fft))
            # 只看"延迟为正"的一半（mic 里的声音只会晚于播放信号）
            seg = cc[: n]
            k = int(np.argmax(seg))
            tmp = seg.copy()
            tmp[max(0, k - 40):k + 41] = 0
            peak_ratio = float(seg[k] / tmp.max()) if tmp.max() > 0 else 0.0
            print(f"  与播放信号互相关：峰位于 {k} 采样（{k/SR*1000:.0f}ms）"
                  f"，峰比 {peak_ratio:.2f}")

    # ---- 分段能量包络：看播放时段是不是**没变化** ----
    win = SR // 10
    n_seg = len(mic) // win
    if n_seg >= 4:
        env = np.sqrt(np.mean(mic[:n_seg * win].reshape(n_seg, win) ** 2,
                              axis=1) + 1e-12)
        pre_seg = env[: max(1, anchor // win)]
        play_seg = env[anchor // win: (anchor + play_len) // win]
        if pre_seg.size and play_seg.size:
            print(f"  包络：播放前中位 {db(np.median(pre_seg)):.1f}dB → "
                  f"播放中位 {db(np.median(play_seg)):.1f}dB"
                  f"（差 {db(np.median(play_seg)/max(np.median(pre_seg),1e-12)):+.1f}dB）")

    # ---- 判定 ----
    #
    # ⚠️ **先看互相关峰比，再看能量**。峰比是"mic 里到底有没有这段已知
    # 波形"的**直接**证据；而能量只反映响度。反过来的话，一个"声音确实
    # 进来了、只是比环境噪声轻"的案例会被误判成"设备侧消掉了" ——
    # 实测踩过：峰比 39.77（回声清清楚楚）却因为 sig<noise 被归到 C 类。
    print()
    print("=" * 70)
    pr = peak_ratio if peak_ratio is not None else float("nan")
    if peak_ratio is not None and peak_ratio >= 3.0:
        print(f"→ 麦克风里**确实有**校准音（峰比 {pr:.1f}），回声路径是通的。")
        if snr < 6:
            print(f"  只是它比环境噪声轻（信噪比 {snr:.1f}dB）—— 调大音量即可。")
        else:
            print("  若服务端仍判失败，问题在服务端的分析逻辑，"
                  "请把这两个 wav 发出来。")
    elif noise > 1e-9 and sig < noise:
        print(f"→ ⚠️ 播放时段能量**低于**本底，且与播放信号无相关"
              f"（峰比 {pr:.2f}）—— 回声在进到页面之前就被消掉了。")
        print("  iOS Safari 在『同时播放+录音』时会启用系统级 VPIO")
        print("  （Voice Processing I/O，自带 AEC/降噪/AGC）：它把扬声器的")
        print("  声音当成回声，在我们拿到麦克风数据**之前**就消掉了。")
        print("  **这意味着 iOS 上无法用页面测量算法 AEC 的回声路径** ——")
        print("  设备侧已经在替我们消了。要么接受用浏览器原生 AEC，")
        print("  要么换 Android/桌面端做算法 AEC 的验证与使用。")
    elif snr < 6:
        print(f"→ 播放时段能量略高于本底，但信噪比只有 {snr:.1f}dB"
              "（峰比也不足）。")
        print("  复核：① 手机是否在静音档 ② 音量是否已调到最大"
              " ③ 是否戴了耳机（耳机没有回声路径）")
        print("  ④ 是否把手机压在软表面上（扬声器被堵住）")
    else:
        print(f"→ 信噪比 {snr:.1f}dB 够，但互相关峰比只有 {pr:.2f}（<3）。")
        print("  可能是混响/多次反射把波形涂抹了。听一下 WAV 确认。")

    print()
    print("提示：两个 WAV 直接用播放器听一遍最直观 ——")
    print("  · 能清楚听到『啾』声 → 问题在设备侧处理或分析逻辑")
    print("  · 几乎听不到      → 音量/静音/耳机问题")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
