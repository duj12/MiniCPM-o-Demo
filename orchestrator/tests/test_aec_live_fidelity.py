#!/usr/bin/env python3
"""保真验证：**线上链路重建的 AEC 输出 ≡ 离线模拟实验的输出**。

## 这个脚本回答什么问题

离线实验（``gen_aec_report_assets.py``）里 AEC 输出干净、ASR 不误识别；
而网页真机调用却"算法 AEC 没起作用"。差别不在模型，在**调用方给它的
nearend/farend 是否样本级对齐**。

本脚本用**与线上完全相同的代码路径**重建这对信号，因此它的输出就是
"网页真机调用应当产出的东西"：

  · 参考轨落位：``RefTrack.place(at_sample = t_send + playback_delay_ms)``
    —— 与 ``actions/executor.py`` 逐字一致
  · 参考轨读取：``RefTrack.read(t)`` → 内部做 ``ref[t] = raw[t - D]``
  · 麦克风：``mic[t] = raw_play[t - D_round] · gain + near[t]``
    其中 ``D_round = 声学延迟``，``raw_play`` 是**实际播出**的信号
  · 送进真实 AEC 服务（``ws://…/ws/asr_frontend``）

## 三件事一起验证

1. **扫 D**：ERLE 随 D 的变化曲线 —— 把"错位 = 失效"从推断变成数字。
   预期 D 对齐时 9~11dB，错位 166ms（250 默认值 vs 84ms 真值）掉到 1~2dB。
2. **自动收敛**：跑**修好的** ``AcousticDelayTracker``，断言它从同一批
   信号里恢复出最优 D —— 证明线上能自己找到这个值，不必靠猜。
3. **与 golden 对比**：同一对 (far, near)，一路走"离线完美对齐"（golden），
   一路走上面的线上语义重建，断言两者 ERLE 接近，并把 wav 落盘供试听。

    python -m orchestrator.tests.test_aec_live_fidelity \
        --far  assets/ref_audio/ref_minicpm_signature.wav \
        --near assets/ref_audio/ref_en_dlc_1.wav \
        --out  /tmp/aec_live
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import sys
import wave
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from orchestrator.aec.client import DEFAULT_CHUNK, SR, AecClient  # noqa: E402
from orchestrator.audio.ref_track import (  # noqa: E402
    AcousticDelayTracker,
    RefTrack,
)

DEFAULT_URL = "ws://192.168.88.253:30255/ws/asr_frontend"
PLAYBACK_DELAY_MS = 200        # 与 config.playback_delay_ms / 前端 nextAt 一致
TTS_SR = 24000                 # TTS 输出采样率（参考轨落位前的原始率）

_failures: List[str] = []


def check(cond: bool, msg: str) -> None:
    print(f"  [{'OK' if cond else 'FAIL'}] {msg}")
    if not cond:
        _failures.append(msg)


def db(x: float) -> float:
    return 10.0 * np.log10(max(float(x), 1e-12))


def energy(x: np.ndarray) -> float:
    return float(np.mean(x ** 2)) if x.size else 0.0


def _to_24k(x16: np.ndarray) -> np.ndarray:
    """16k → 24k int16（喂给 ``RefTrack.place()`` 的 TTS 格式）。

    ``place()`` 内部会再降回 16k，于是轨上内容与 ``x16`` 一致、时长也
    一致 —— 这是 mic 与 ref 能对齐的前提。
    """
    n_out = int(len(x16) * TTS_SR / SR)
    pos = np.linspace(0, len(x16) - 1, n_out)
    i0 = np.floor(pos).astype(int)
    i1 = np.minimum(i0 + 1, len(x16) - 1)
    fr = (pos - i0).astype(np.float32)
    y = x16[i0] * (1 - fr) + x16[i1] * fr
    return np.clip(y * 32767.0, -32768, 32767).astype(np.int16)


def load_wav(path: str) -> np.ndarray:
    """读 wav 并重采样到 16kHz float32。"""
    with wave.open(path, "rb") as w:
        src_sr = w.getframerate()
        raw = w.readframes(w.getnframes())
        x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        if w.getnchannels() == 2:
            x = x.reshape(-1, 2).mean(axis=1)
    if src_sr != SR:
        n_out = int(len(x) * SR / src_sr)
        pos = np.linspace(0, len(x) - 1, n_out)
        i0 = np.floor(pos).astype(int)
        i1 = np.minimum(i0 + 1, len(x) - 1)
        fr = (pos - i0).astype(np.float32)
        x = (x[i0] * (1 - fr) + x[i1] * fr).astype(np.float32)
    return x


def write_wav(path: Path, x: np.ndarray) -> None:
    data = np.clip(x * 32767.0, -32768, 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(data.tobytes())


# ---------------------------------------------------------------------- #
#  线上链路的信号重建
# ---------------------------------------------------------------------- #

class LivePath:
    """用**与线上相同的代码**重建 (mic, farend) 对。

    时间轴是会话采样时钟。约定：

      · t=0 时参考轨收到第一块 TTS 音频（``t_send``）
      · 参考轨把它落位到 ``t_send + playback_delay``（executor 的做法）
      · 浏览器在实际起播时刻 ``t_send + playback_delay + jitter`` 出声，
        声音经声学路径延迟 ``acoustic`` 到达麦克风
      · 于是 **真实往返 D_round = jitter + acoustic**，
        这正是 ``read()`` 需要补偿的量
    """

    def __init__(self, speech16: np.ndarray, near16: np.ndarray,
                 jitter_ms: float = 0.0, acoustic_ms: float = 84.0,
                 echo_gain: float = 0.5, noise: float = 1e-4) -> None:
        self.speech16 = speech16
        self.near16 = near16
        self.jitter = int(jitter_ms * SR / 1000)
        self.acoustic = int(acoustic_ms * SR / 1000)
        self.echo_gain = echo_gain
        self.noise = noise
        self.rng = np.random.default_rng(3)

    @property
    def d_round(self) -> int:
        """`read()` 需要补偿的真实往返延迟（采样）。"""
        return self.jitter + self.acoustic

    def build(self, d_comp: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """按补偿量 ``d_comp`` 构造 (mic, farend, near)。

        与线上逐字一致：参考轨落位 → ``read()``（含 D 补偿）→ 作为 farend。
        """
        # ① 参考轨落位：与 actions/executor.py 的 at_sample 算法一致。
        #    这里 t_send=0，故 at_sample = playback_delay_samples。
        #
        # ⚠️ TTS 输出是 24kHz，``RefTrack.place()`` 内部会把它重采样到
        # 16kHz（3:2）。麦克风里的回声是**播出信号**在 16k 会话时钟上的
        # 样子 —— 与 place 之后的轨内容必须一致，否则 mic 与 ref 讲的
        # 是两段不同的波形（时长都对不上），测出来的 ERLE 毫无意义。
        # 所以这里把 16k 的 far 上采到 24k 再 place，落回 16k 后内容与
        # far 一致、时长也一致。
        rt = RefTrack()
        t_send = 0
        at_sample = t_send + int(PLAYBACK_DELAY_MS * SR / 1000.0)
        pcm24 = _to_24k(self.speech16)
        rt.place("r1", 0, pcm24, at_sample)
        rt.delay_samples = d_comp

        total = at_sample + len(self.speech16) + self.d_round + SR
        farend = rt.read(0, total)

        # ② 麦克风：原始轨（未补偿）延迟 D_round + 近端语音 + 噪声。
        #    raw_play[t] = speech16[t - at_sample]
        mic = np.zeros(total, dtype=np.float32)
        s = at_sample + self.d_round
        n = min(len(self.speech16), total - s)
        mic[s:s + n] = self.speech16[:n] * self.echo_gain
        # 近端在麦克风上独立出现（与回声不相关）
        m = min(len(self.near16), total)
        mic[:m] += self.near16[:m]
        mic += self.rng.standard_normal(total).astype(np.float32) * self.noise

        return (mic.astype(np.float32), farend.astype(np.float32),
                self.near16[:total] if len(self.near16) >= total
                else np.pad(self.near16, (0, total - len(self.near16))))


def build_golden(speech16: np.ndarray, near16: np.ndarray,
                 echo_gain: float = 0.5, noise: float = 1e-4) -> Tuple:
    """离线实验的做法：**完美对齐**给 AEC。

    mic = far 延迟 D 衰减 + near，farend 与 mic 里的回声分量样本级对齐。
    这就是 ``gen_aec_report_assets.py`` 里 AEC 表现良好的条件。
    """
    rng = np.random.default_rng(3)
    total = len(speech16) + SR
    near = (near16[:total] if len(near16) >= total
            else np.pad(near16, (0, total - len(near16))))
    mic = np.zeros(total, dtype=np.float32)
    n = min(len(speech16), total)
    mic[:n] = speech16[:n] * echo_gain
    mic += near
    mic += rng.standard_normal(total).astype(np.float32) * noise
    farend = np.pad(speech16, (0, total - len(speech16)))
    return mic.astype(np.float32), farend.astype(np.float32), near


# ---------------------------------------------------------------------- #
#  送进真实 AEC 服务
# ---------------------------------------------------------------------- #

async def run_aec(url: str, mic: np.ndarray, farend: np.ndarray,
                  tag: str) -> np.ndarray:
    _ALLOW = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
                 "0123456789_-")
    safe = "".join(c if c in _ALLOW else "_" for c in tag)[:40]
    aec = AecClient(url, connection_id=f"fid_{safe}")
    await aec.connect()
    outs: List[np.ndarray] = []
    recv = asyncio.create_task(aec.recv_loop(
        lambda s: outs.append(np.asarray(s).reshape(-1).copy())))
    for i in range(0, len(mic), DEFAULT_CHUNK):
        seg = mic[i:i + DEFAULT_CHUNK]
        ref = farend[i:i + DEFAULT_CHUNK]
        if seg.size < DEFAULT_CHUNK:
            seg = np.pad(seg, (0, DEFAULT_CHUNK - seg.size))
            ref = np.pad(ref, (0, DEFAULT_CHUNK - ref.size))
        await aec.push(seg[None, :].astype(np.float32),
                       ref[None, :].astype(np.float32))
        await asyncio.sleep(DEFAULT_CHUNK / SR)
    await aec.push(np.zeros((1, DEFAULT_CHUNK), dtype=np.float32),
                   np.zeros((1, DEFAULT_CHUNK), dtype=np.float32), is_end=True)
    try:
        await asyncio.wait_for(recv, timeout=30)
    except asyncio.TimeoutError:
        recv.cancel()
    await aec.close(send_end=False)
    return np.concatenate(outs) if outs else np.zeros(0, dtype=np.float32)


def echo_suppression(mic: np.ndarray, out: np.ndarray,
                     near: np.ndarray) -> Tuple[float, float, float]:
    """返回 ``(ERLE总, 回声抑制, 近端保留)``，全部 dB。

    · ERLE总     = mic 总能量 / 输出总能量（含近端，故偏低）
    · 回声抑制   = **相对回声分量**的抑制 —— 看这个
    · 近端保留   = 输出能量 / 纯近端能量（~0 表示近端没被削）
    """
    n = min(len(mic), len(out), len(near))
    if n == 0:
        return (float("nan"),) * 3
    e_in, e_out, e_near = (energy(mic[:n]), energy(out[:n]), energy(near[:n]))
    total = db(e_in) - db(e_out)
    # 回声分量能量 = 输入能量 − 近端能量（二者不相关，能量可减）
    e_echo = max(e_in - e_near, 1e-12)
    # 输出里的残留 ≈ 输出能量 − 近端能量（AEC 保近端，故减得合理）
    e_resid = max(e_out - e_near, 1e-12)
    supp = db(e_echo) - db(e_resid)
    keep = db(e_out) - db(e_near) if e_near > 0 else float("nan")
    return (total, supp, keep)


def envelope(x: np.ndarray, win: int = 1600) -> np.ndarray:
    """分块 RMS 包络 —— 用于比较两路输出的"形状"是否一致。"""
    n = len(x) // win
    if n < 2:
        return np.zeros(0, dtype=np.float32)
    return np.sqrt(np.mean(x[:n * win].reshape(n, win) ** 2, axis=1) + 1e-12)


# ---------------------------------------------------------------------- #

async def main_async(args) -> int:
    far = load_wav(args.far)
    near = load_wav(args.near)
    if args.seconds:
        far = far[: int(args.seconds * SR)]
        near = near[: int(args.seconds * SR)]
    print(f"远端（模拟 TTS）: {Path(args.far).name}  {len(far)/SR:.2f}s")
    print(f"近端（模拟用户）: {Path(args.near).name}  {len(near)/SR:.2f}s")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    # ---------------- 1) 离线 golden ----------------
    print()
    print("=" * 78)
    print("① 离线 golden（完美对齐 —— 离线实验的条件）")
    print("-" * 78)
    mic_g, ref_g, near_g = build_golden(far, near)
    out_g = await run_aec(args.url, mic_g, ref_g, "golden")
    g_total, g_supp, g_keep = echo_suppression(mic_g, out_g, near_g)
    print(f"  ERLE总={g_total:.1f}dB  回声抑制={g_supp:.1f}dB  "
          f"近端保留={g_keep:+.1f}dB")
    write_wav(out_dir / "10_golden_mic.wav", mic_g)
    write_wav(out_dir / "11_golden_out.wav", out_g)
    # 线上路径各变量的初值（在下面的 D 扫描循环里赋值）
    live_aligned: Tuple[float, float, float] = (float("nan"),) * 3
    live_mis: Tuple[float, float, float] = (float("nan"),) * 3
    out_live_aligned: Optional[np.ndarray] = None
    rows.append(("golden", 0, g_total, g_supp, g_keep))

    # ---------------- 2) 线上语义重建 + 扫 D ----------------
    print()
    print("=" * 78)
    print("② 线上链路重建：扫补偿量 D，看 ERLE 曲线")
    print("-" * 78)
    print(f"  真实往返 D_round = jitter({args.jitter_ms:.0f}ms) + "
          f"声学({args.acoustic_ms:.0f}ms) = "
          f"{int((args.jitter_ms + args.acoustic_ms) * SR / 1000)} 采样")
    print()
    print(f"  {'补偿 D':>10} {'ERLE总':>9} {'回声抑制':>10} {'近端保留':>10}")
    print("  " + "-" * 44)

    live = LivePath(far, near, jitter_ms=args.jitter_ms,
                    acoustic_ms=args.acoustic_ms)
    d_round = live.d_round
    d_true_ms = d_round / SR * 1000.0

    best = (-1e9, None)
    for d_ms in args.sweep:
        d_comp = int(d_ms * SR / 1000.0)
        mic_l, ref_l, near_l = live.build(d_comp)
        out_l = await run_aec(args.url, mic_l, ref_l, f"live{d_ms}")
        tot, supp, keep = echo_suppression(mic_l, out_l, near_l)
        mark = "  ← 真值" if abs(d_ms - d_true_ms) < 5 else ""
        print(f"  {d_ms:>8.0f}ms {tot:>8.1f}dB {supp:>9.1f}dB "
              f"{keep:>9.1f}dB{mark}")
        rows.append((f"live_D{d_ms:.0f}", d_ms, tot, supp, keep))
        if supp > best[0]:
            best = (supp, d_ms)
        if abs(d_ms - d_true_ms) < 5:
            write_wav(out_dir / "20_live_mic.wav", mic_l)
            write_wav(out_dir / "21_live_out_aligned.wav", out_l)
            live_aligned = (tot, supp, keep)
            out_live_aligned = out_l
        if abs(d_ms - args.misaligned_ms) < 5:
            write_wav(out_dir / "22_live_out_misaligned.wav", out_l)
            live_mis = (tot, supp, keep)

    print("  " + "-" * 44)
    print(f"  最优补偿 {best[1]:.0f}ms → 回声抑制 {best[0]:.1f}dB")

    # ---------------- 3) 自动收敛验证 ----------------
    print()
    print("=" * 78)
    print("③ 自适应延迟估计能否自动找到该补偿量")
    print("-" * 78)
    mic_a, ref_raw_a, _ = live.build(0)     # 用**未补偿**的 raw 做估计
    tracker = AcousticDelayTracker(sr=SR, max_delay_ms=args.max_delay_ms)
    dmax = int(tracker.max_delay)
    WINDOW = SR
    # 与 session._maybe_update_delay 完全相同的取窗方式
    t = dmax + WINDOW
    while t + WINDOW <= len(mic_a):
        m = mic_a[t - WINDOW:t]
        r = ref_raw_a[t - WINDOW - dmax:t]
        tracker.estimate(m, r, offset=dmax)
        t += WINDOW
    est_ms = tracker.delay / SR * 1000.0
    print(f"  估计值 {tracker.delay} 采样（{est_ms:.0f}ms），"
          f"成功 {tracker.estimates} 次")
    print(f"  真实往返 {d_round} 采样（{d_true_ms:.0f}ms）")
    err = tracker.delay - d_round
    check(abs(err) <= 2, f"自适应估计误差 {err:+d} 采样（≤2）")

    # ---------------- 4) 保真对比 ----------------
    print()
    print("=" * 78)
    print("④ 线上重建 vs 离线 golden")
    print("-" * 78)
    # 判据是**单向**的：要求线上不比离线 golden 差。
    # 线上略好是正常的（golden 的合成回声落在 D=0，而线上是真实 D_round
    # 对齐 —— 见扫描表，D=0 那一档本来就略低）。写 abs() 会把"更好"
    # 误判成失败。
    gap = live_aligned[1] - g_supp
    check(gap >= -2.0,
          f"线上回声抑制不低于 golden 2dB 以上"
          f"（golden {g_supp:.1f} → 线上 {live_aligned[1]:.1f}，"
          f"差 {gap:+.1f}dB）")
    mis_drop = live_aligned[1] - live_mis[1]
    check(mis_drop > 5.0,
          f"错位 {args.misaligned_ms:.0f}ms 时抑制下降 {mis_drop:.1f}dB "
          f"—— 证实「错位 = 失效」")

    # 输出包络相关度：两路输出的"形状"应一致（都只应剩近端语音）。
    # 用互相关比形状而不是逐样本比 —— 两条路径的近端落位差一个
    # playback_delay，逐样本比必然对不上。
    eg = envelope(out_g)
    el = envelope(out_live_aligned if out_live_aligned is not None
                  else np.zeros(0, dtype=np.float32))
    n = min(len(eg), len(el))
    if n >= 8:
        a = eg[:n] - eg[:n].mean()
        b = el[:n] - el[:n].mean()
        denom = float(np.sqrt((a ** 2).sum() * (b ** 2).sum()))
        corr = float((a * b).sum() / denom) if denom > 0 else 0.0
        check(corr > 0.7, f"输出包络相关度 {corr:.3f} > 0.7（形状一致）")
    print(f"  试听：{out_dir}/10-11 为离线 golden，"
          f"20-22 为线上重建")

    # ---------------- 落盘 CSV ----------------
    csv_path = out_dir / "live_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["case", "delay_ms", "erle_total_db",
                    "echo_suppression_db", "near_keep_db"])
        for r in rows:
            w.writerow([r[0], f"{r[1]:.0f}", f"{r[2]:.2f}",
                        f"{r[3]:.2f}", f"{r[4]:.2f}"])
    print(f"\n  指标表: {csv_path}")

    print()
    print("=" * 78)
    if _failures:
        print(f"FAILED: {len(_failures)} 项")
        for f in _failures:
            print(f"  - {f}")
        return 1
    print("通过 —— 线上链路重建的输出与离线 golden 等价")
    return 0


def main() -> None:
    here = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser(description="AEC 线上链路保真验证")
    p.add_argument("--url", default=DEFAULT_URL)
    p.add_argument("--far", default=str(here / "assets/ref_audio/"
                                        "ref_minicpm_signature.wav"))
    p.add_argument("--near", default=str(here / "assets/ref_audio/"
                                         "ref_en_dlc_1.wav"))
    p.add_argument("--out", default="/tmp/aec_live")
    p.add_argument("--seconds", type=float, default=8.0)
    p.add_argument("--jitter-ms", type=float, default=0.0,
                   help="浏览器起播相对约定的额外偏差")
    p.add_argument("--acoustic-ms", type=float, default=84.0,
                   help="纯声学路径延迟（报告推断的残差量级）")
    p.add_argument("--misaligned-ms", type=float, default=250.0,
                   help="用来演示「错位=失效」的补偿量（旧默认值）")
    p.add_argument("--max-delay-ms", type=float, default=300.0)
    # 加密在容忍窗边界附近（实测断崖落在 84~120ms 之间）——
    # 这条边界直接决定"冷启动默认值差多少还救得回来"
    p.add_argument("--sweep", type=float, nargs="+",
                   default=[0, 40, 84, 90, 95, 100, 105, 110, 120, 170,
                            250, 284])
    args = p.parse_args()
    raise SystemExit(asyncio.run(main_async(args)))


if __name__ == "__main__":
    main()
