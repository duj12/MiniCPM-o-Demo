#!/usr/bin/env python3
"""AEC 关键实验：**预对齐**是否能让它工作。

## 背景：之前两个实验的结论都不可靠

1. `test_aec_align.py` 扫延迟时，far 与 near **用的是同一段语音**
   （``far = speech.copy()``）。于是 mic 里除了回声没有独立的近端内容 ——
   消掉回声就等于消掉一切。当时报的"近端保留 -49dB / 过度抑制"是
   **实验设计的假象**，不是模型缺陷。

2. 由此推出的"云端 AEC 要求样本级对齐、真机 284ms 必然失效"也只对了一半：
   它说明**不预对齐**时模型自身不估计时延（窗口 <20ms），但**没测**
   预对齐之后会怎样。

## 本实验

用**两段互相独立的语音**：far 一段、near 另一段。构造
``mic = far 延迟 D_path 衰减 + near``，然后对比三种送法：

  A. 不预对齐：直接送 (mic, far)         —— 模型要自己找时延
  B. 预对齐：  送 (mic, ref) 其中 ref 已按 D 前移 —— **生产链路走这条**
  C. 错位预对齐：ref 按 D+50ms 前移       —— 检验对齐精度的敏感度

看两个量：
  · **ERLE**：回声被消掉多少 dB（相对 mic 里的回声分量）
  · **近端保真**：near 的能量是否保留（这次 near 与 far 独立，有意义）

    python -m orchestrator.tests.test_aec_prealign \
        --far assets/ref_audio/ref_minicpm_signature.wav \
        --near assets/ref_audio/ref_en_dlc_1.wav
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from orchestrator.aec.client import DEFAULT_CHUNK, SR, AecClient  # noqa: E402

DEFAULT_URL = "ws://192.168.88.253:30255/ws/asr_frontend"
_ALLOW = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_")


def db(x: float) -> float:
    return 10.0 * np.log10(max(float(x), 1e-12))


def energy(x: np.ndarray) -> float:
    return float(np.mean(x ** 2)) if x.size else 0.0


def load_wav(path: str) -> np.ndarray:
    """读 wav 并重采样到 16kHz（源文件可能是 44.1k/48k）。"""
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
        print(f"  （{Path(path).name}: {src_sr}Hz → {SR}Hz）")
    return x


async def push_pair(url: str, mic: np.ndarray, ref: np.ndarray,
                    tag: str) -> np.ndarray:
    """把 (mic, ref) 成对送进 AEC，返回输出。"""
    aec = AecClient(url, connection_id=f"pa_{tag}")
    await aec.connect()
    outs = []
    recv = asyncio.create_task(aec.recv_loop(
        lambda s: outs.append(np.asarray(s).reshape(-1).copy())))
    for i in range(0, len(mic), DEFAULT_CHUNK):
        m = mic[i:i + DEFAULT_CHUNK]
        r = ref[i:i + DEFAULT_CHUNK]
        if m.size < DEFAULT_CHUNK:
            m = np.pad(m, (0, DEFAULT_CHUNK - m.size))
            r = np.pad(r, (0, DEFAULT_CHUNK - r.size))
        await aec.push(m[None, :].astype(np.float32), r[None, :].astype(np.float32))
        await asyncio.sleep(DEFAULT_CHUNK / SR)
    z = np.zeros((1, DEFAULT_CHUNK), dtype=np.float32)
    await aec.push(z, z, is_end=True)
    try:
        await asyncio.wait_for(recv, timeout=20)
    except asyncio.TimeoutError:
        recv.cancel()
    await aec.close(send_end=False)
    return np.concatenate(outs) if outs else np.zeros(0, dtype=np.float32)


def build(far: np.ndarray, near: np.ndarray, d: int, gain: float):
    """返回 (mic, echo, near)。mic = echo(delay=d) + near。"""
    n = min(len(far), len(near))
    far, near = far[:n], near[:n]
    echo = np.zeros(n, dtype=np.float32)
    if d <= 0:
        echo = far * gain
    else:
        echo[d:] = far[:n - d] * gain
    mic = (echo + near).astype(np.float32)
    return mic, echo, near


def shift(x: np.ndarray, d: int) -> np.ndarray:
    """把 x 前移 d（即 ref[t] = x[t-d] 的效果）：用于预对齐。"""
    if d == 0:
        return x.copy()
    out = np.zeros_like(x)
    if d > 0:
        out[:-d] = x[d:]
    else:
        out[-d:] = x[:d]
    return out


def report(label: str, mic: np.ndarray, out: np.ndarray,
           echo: np.ndarray, near: np.ndarray) -> tuple:
    n = min(len(mic), len(out))
    if n == 0:
        print(f"{label:<34} 无输出")
        return None, None
    e_mic, e_out = energy(mic[:n]), energy(out[:n])
    e_echo, e_near = energy(echo[:n]), energy(near[:n])
    erle = db(e_mic) - db(e_out)                     # 总能量抑制
    # 相对回声分量的抑制（更准确：输出里的剩余应接近 near 的量）
    residual_db = db(max(e_out - e_near, 1e-12))     # 扣掉近端后的剩余
    echo_supp = db(e_echo) - residual_db             # 回声被压了多少
    keep = db(e_out) - db(e_near)                    # 近端保留（0 = 完好）
    print(f"{label:<34} ERLE(总)={erle:>6.1f}dB  回声抑制={echo_supp:>6.1f}dB  "
          f"近端保留={keep:>6.1f}dB")
    return erle, keep


async def main_async(args) -> int:
    far = load_wav(args.far)
    near = load_wav(args.near)
    sec = args.seconds
    far, near = far[:int(sec * SR)], near[:int(sec * SR)]
    print(f"far : {Path(args.far).name}  {len(far)/SR:.1f}s  能量 {db(energy(far)):.1f}dB")
    print(f"near: {Path(args.near).name}  {len(near)/SR:.1f}s  能量 {db(energy(near)):.1f}dB")
    print()
    print("=" * 84)
    print("预对齐实验（far 与 near 是**不同**的语音，近端保留才有意义）")
    print("-" * 84)

    for d_ms in (0, 100, 284, 500):
        d = int(d_ms * SR / 1000)
        mic, echo, near_a = build(far, near, d, args.gain)
        print(f"\n--- 真实延迟 D_path = {d_ms}ms（{d} 采样）---")

        # A. 不预对齐（模型要自己找时延）
        out_a = await push_pair(args.url, mic, far, f"raw{d_ms}")
        report("A. 不预对齐 (mic, far)", mic, out_a, echo, near_a)

        # B. 预对齐（生产链路走这条）
        out_b = await push_pair(args.url, mic, shift(far, d), f"pre{d_ms}")
        report("B. 预对齐 (mic, ref=far前移D) ★", mic, out_b, echo, near_a)

        # C. 预对齐但偏 50ms
        dd = d + int(0.05 * SR)
        out_c = await push_pair(args.url, mic, shift(far, dd), f"err{d_ms}")
        report("C. 预对齐偏差 +50ms", mic, out_c, echo, near_a)

    print()
    print("=" * 84)
    print("解读：")
    print("  · B 行「回声抑制」高 → **预对齐可行**，问题只在我们的时延估计精度")
    print("  · B 行「近端保留」接近 0dB → 近端没被误伤（之前的 -49dB 是实验假象）")
    print("  · C 行若明显变差 → 对时延估计精度要求高，需要持续跟踪而非单次估计")
    return 0


def main() -> None:
    p = argparse.ArgumentParser(description="AEC 预对齐实验")
    p.add_argument("--url", default=DEFAULT_URL)
    p.add_argument("--far", required=True, help="模拟 TTS 播放的信号")
    p.add_argument("--near", required=True, help="模拟用户说话（需与 far 不同）")
    p.add_argument("--seconds", type=float, default=6.0)
    p.add_argument("--gain", type=float, default=0.5, help="回声衰减")
    args = p.parse_args()
    raise SystemExit(asyncio.run(main_async(args)))


if __name__ == "__main__":
    main()
