#!/usr/bin/env python3
"""离线测声学延迟 D —— 每台设备测一次，写进配置。

## 为什么要离线测

AEC 的容忍窗只有约 **±5ms**（``test_aec_live_fidelity.py``：D 偏 6ms，
回声抑制从 12.6dB 掉到 2.5dB）。而 D 现在只剩「扬声器→麦克风的物理延迟 +
设备音频 I/O 缓冲」—— 网络往返、浏览器主线程抖动、mic 在途积压**都不再
计入**，因为参考轨的落位时刻改由浏览器承诺（见 ``clock.ctx_to_sample``）。
既然是纯设备常量，就没有理由在运行时去猜它：一次测准，写进配置。

（原先的运行时自适应已废弃：实测在真机上收敛不了，且一旦被噪声假峰钉死
就会永久失效，见 ``tools/delay_estimate.py`` 里的事故记录。）

## 怎么用

```bash
# ① 跑一轮真实会话，开着三/四路转储
ORCH_DUMP_AUDIO=/tmp/orchdump/s python -m orchestrator.main --port 8100
#    ...在页面上选「算法服务 AEC」，外放、别戴耳机，说几轮...

# ② 用 dump 算 D
python -m orchestrator.tests.measure_delay \
    --mic /tmp/orchdump/s-xxxx-mic.wav \
    --raw /tmp/orchdump/s-xxxx-raw.wav \
    --verify-url ws://192.168.88.253:30255/ws/asr_frontend

# ③ 把打印出来的 ORCH_AEC_DEFAULT_DELAY_MS 写进服务端环境变量
```

⚠️ **必须用 ``-raw.wav``（未做 D 补偿的原始参考轨）**。``-ref.wav`` 是
喂给 AEC 的那一份（已补偿），拿它互相关只能得到**残差**，不是绝对 D。

⚠️ **dump 必须是本次改动之后重新采的**。改动前参考轨的落位误差是**变化**的，
单个 D 吸收不了 —— 脚本会报出很大的散度，这本身就是有用的诊断（见输出
末尾的提示）。
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import wave
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from orchestrator.clock import SR  # noqa: E402
from orchestrator.tools.delay_estimate import AcousticDelayTracker  # noqa: E402

# 一票否决的门槛，与 tracker 的 MIN_CORROBORATION / MAX_SPREAD_SAMPLES 同源
MIN_WINDOWS = 3
MAX_SPREAD_SAMPLES = 160          # 10ms @16k —— 容忍窗同量级
DEFAULT_WINDOW_S = 1.0
DEFAULT_DMAX_MS = 300.0
# 判定"在播放"的包络门槛。参考轨是归一化的 [-1,1]，静音区的 RMS ~1e-9
# （TrackBuffer 未写入区读出为精确 0），所以这个门槛可以很低。
PLAY_RMS_GATE = 1e-3
MIN_REGION_S = 0.3


# ---------------------------------------------------------------------- #
#  读盘
# ---------------------------------------------------------------------- #

def load_wav(path: str) -> np.ndarray:
    """读单声道 16k wav → float32。"""
    with wave.open(path, "rb") as w:
        sr = w.getframerate()
        ch = w.getnchannels()
        raw = w.readframes(w.getnframes())
    if sr != SR:
        raise SystemExit(f"{path}: 采样率 {sr} != {SR}，dump 应当都是 16k")
    x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if ch == 2:
        x = x.reshape(-1, 2).mean(axis=1)
    return x


def playback_regions(raw: np.ndarray, win: int = 320,
                     gate: float = PLAY_RMS_GATE) -> List[Tuple[int, int]]:
    """找出参考轨**确实有内容**的连续区间（20ms 包络 + 门槛）。

    只在播放区间内取窗：空闲时 mic 里没有回声，互相关必然是垃圾。
    """
    n = len(raw) // win
    if n < 2:
        return []
    env = np.sqrt(np.mean(raw[:n * win].reshape(n, win) ** 2, axis=1))
    active = env > gate
    regions: List[Tuple[int, int]] = []
    start = None
    for i, a in enumerate(active):
        if a and start is None:
            start = i
        elif not a and start is not None:
            regions.append((start * win, i * win))
            start = None
    if start is not None:
        regions.append((start * win, n * win))
    # 太短的段丢掉（不足以做互相关，也大概率是噪声/残留）
    return [(a, b) for a, b in regions if (b - a) / SR >= MIN_REGION_S]


# ---------------------------------------------------------------------- #
#  估计
# ---------------------------------------------------------------------- #

def estimate_D(mic: np.ndarray, raw: np.ndarray, window_s: float,
               max_delay_ms: float) -> dict:
    """在播放区间内逐窗估计 D，返回中位数与一致性统计。"""
    W = int(window_s * SR)
    regions = playback_regions(raw)
    ests: List[int] = []
    ratios: List[float] = []
    skipped: List[str] = []

    for lo, hi in regions:
        # 窗口只需完整落在播放区间内。
        #
        # ⚠️ **不要**在开头再留一个 dmax 的余量 —— 那是 `offset > 0` 那种
        # "参考窗比 mic 窗长"的几何才需要的。这里用 `offset=0` 的等长窗：
        # 回声就在窗口**内部**（mic[j] ≈ ref[j-D]，D < W 时重叠段长
        # W-D，对 1s 窗绰绰有余），开头留白只会白白吃掉一个窗口
        # （实测：2s 的播报从能出 2 个窗变成一个）。
        t = lo
        while t + W <= hi:
            m = mic[t:t + W]
            r = raw[t:t + W]
            if m.shape[0] == W and r.shape[0] == W:
                tr = AcousticDelayTracker(sr=SR, max_delay_ms=max_delay_ms)
                got = tr.estimate(m, r, offset=0)
                if got is not None:
                    ests.append(got)
                    ratios.append(tr.last_ratio)
                else:
                    skipped.append(tr.last_reason or "未知")
            t += W

    out = {
        "regions": regions,
        "n_windows": len(ests),
        "estimates": ests,
        "peak_ratios": ratios,
        "skipped": skipped,
    }
    if ests:
        out["median"] = int(np.median(ests))
        out["spread"] = int(max(ests) - min(ests))
    return out


def verdict(res: dict) -> Tuple[bool, str]:
    """给出"能不能信这个 D"的判断。"""
    if res["n_windows"] < MIN_WINDOWS:
        return False, (f"有效窗口只有 {res['n_windows']} 个（需 ≥{MIN_WINDOWS}）"
                       f"—— 会话里播报太少，或参考轨没送到 AEC。")
    if res["spread"] > MAX_SPREAD_SAMPLES:
        return False, (
            f"估计散度 {res['spread']} 采样（{res['spread']/SR*1000:.1f}ms）"
            f" > {MAX_SPREAD_SAMPLES} —— **D 不是一个常量**。\n"
            f"      最常见的原因：这份 dump 是**修复之前**采的，参考轨落位"
            f"含变化的网络/主线程抖动，单个常量 D 吸收不了。\n"
            f"      请用修复后的代码重新采一轮 dump（外放、别戴耳机、"
            f"算法服务 AEC、说两三句）。"
        )
    return True, ""


# ---------------------------------------------------------------------- #
#  用真实 AEC 验证测出来的 D
# ---------------------------------------------------------------------- #

async def verify_with_aec(url: str, mic: np.ndarray, raw: np.ndarray,
                          d_samples: int, span_ms: int = 10) -> dict:
    """在 D±span 三个值上重跑真实 AEC，看抑制峰是否落在 D 上。

    这一步把"**测出的** D"变成"**验证过的** D"。只看互相关峰值比是不够的
    —— 那只能说明两个波形像，不能说明 AEC 会满意。
    """
    from orchestrator.tests.test_aec_live_fidelity import (
        echo_suppression, run_aec,
    )
    step = int(span_ms / 1000.0 * SR)
    results = {}
    for label, d in (("D-Δ", max(0, d_samples - step)),
                     ("D", d_samples),
                     ("D+Δ", d_samples + step)):
        # ref[t] = raw[t - D] —— 与 RefTrack.read() 逐字一致
        farend = np.zeros_like(raw)
        if d < len(raw):
            farend[d:] = raw[:len(raw) - d]
        near = np.zeros_like(raw)      # 这里 mic 已含近端，无法分离，用全零
        out = await run_aec(url, mic, farend, f"md_{label}")
        n = min(len(mic), len(out))
        _, supp, _ = echo_suppression(mic[:n], out[:n], near[:n])
        results[label] = {"delay_samples": d,
                          "delay_ms": d / SR * 1000.0,
                          "suppression_db": float(supp)}
    return results


# ---------------------------------------------------------------------- #

def main() -> None:
    p = argparse.ArgumentParser(
        description="离线测声学延迟 D（用 ORCH_DUMP_AUDIO 的 dump）")
    p.add_argument("--mic", required=True,
                   help="<前缀>-<sid>-mic.wav（原始麦克风，含回声）")
    p.add_argument("--raw", required=True,
                   help="<前缀>-<sid>-raw.wav（**未补偿**的原始参考轨）")
    p.add_argument("--window-s", type=float, default=DEFAULT_WINDOW_S)
    p.add_argument("--dmax-ms", type=float, default=DEFAULT_DMAX_MS)
    p.add_argument("--verify-url", default=None,
                   help="AEC 服务地址；给了就在 D±10ms 上重跑真实 AEC 验证")
    p.add_argument("--write-store", action="store_true",
                   help="把结果写进 DelayStore（按 --client-key 分组）")
    p.add_argument("--client-key", default=None)
    args = p.parse_args()

    mic = load_wav(args.mic)
    raw = load_wav(args.raw)
    n = min(len(mic), len(raw))
    mic, raw = mic[:n], raw[:n]
    print(f"mic: {Path(args.mic).name}  {len(mic)/SR:.1f}s")
    print(f"raw: {Path(args.raw).name}  {len(raw)/SR:.1f}s"
          f"（未补偿的原始参考轨）")

    res = estimate_D(mic, raw, args.window_s, args.dmax_ms)
    print()
    print("=" * 70)
    print("播放区间")
    print("-" * 70)
    for lo, hi in res["regions"]:
        print(f"  [{lo/SR:7.2f}s, {hi/SR:7.2f}s)  长 {(hi-lo)/SR:.2f}s")
    if not res["regions"]:
        print("  （没有检测到播放区间 —— 参考轨整段是静音？）")

    print()
    print("逐窗估计（采样 @16k）")
    print("-" * 70)
    if res["estimates"]:
        print("  " + " ".join(str(e) for e in res["estimates"]))
        print(f"  中位数 = {res['median']} 采样 = "
              f"{res['median']/SR*1000:.1f} ms")
        print(f"  散度   = {res['spread']} 采样 = "
              f"{res['spread']/SR*1000:.1f} ms")
        print(f"  平均峰比 = {np.mean(res['peak_ratios']):.2f}"
              f"（<1.5 说明回声弱，估计不可信）")
    else:
        print("  （无）")
    for r in res["skipped"][:5]:
        print(f"  跳过：{r}")

    ok, why = verdict(res)
    print()
    print("=" * 70)
    if not ok:
        print(f"⚠️ 无法给出可信的 D：\n      {why}")
        raise SystemExit(1)

    d_ms = res["median"] / SR * 1000.0
    print(f"D = {res['median']} 采样 = {d_ms:.1f} ms"
          f"（{res['n_windows']} 个一致窗口）")

    if args.verify_url:
        print()
        print("=" * 70)
        print("用真实 AEC 验证（抑制峰应落在 D 上）")
        print("-" * 70)
        vr = asyncio.run(verify_with_aec(
            args.verify_url, mic, raw, res["median"]))
        best = None
        for label in ("D-Δ", "D", "D+Δ"):
            v = vr[label]
            print(f"  {label:4s} D={v['delay_ms']:6.1f}ms  "
                  f"回声抑制 {v['suppression_db']:5.1f} dB")
            if best is None or v["suppression_db"] > vr[best]["suppression_db"]:
                best = label
        if best == "D":
            print("  → 峰值确实在 D 上 ✓")
        else:
            print(f"  ⚠️ 峰值在 {best} 上而不是 D —— 建议改用该值，"
                  f"或再采一轮 dump 复测")

    print()
    print("=" * 70)
    print(f"把下面这行加进服务端环境变量：\n")
    print(f"    ORCH_AEC_DEFAULT_DELAY_MS={d_ms:.1f}")

    if args.write_store:
        if not args.client_key:
            raise SystemExit("--write-store 需要 --client-key")
        from orchestrator.config import Settings
        from orchestrator.delay_store import DelayStore
        store = DelayStore(Settings.from_env().delay_store_path)
        store.put(args.client_key, d_ms, n_samples=res["n_windows"])
        print(f"\n已写入 DelayStore：client={args.client_key}  D={d_ms:.1f}ms")


if __name__ == "__main__":
    main()
