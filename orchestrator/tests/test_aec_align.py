#!/usr/bin/env python3
"""AEC 对齐能力细粒度扫描。

上一个实验（test_aec_effect.py）得到一个反直觉的结果：
  · D=0     → ERLE 53.5dB（但近端也被消掉了）
  · D=20ms  → ERLE 5.6dB
  · D=100ms → ERLE 0.8dB（完全失效）

说明 AEC 对 nearend/farend 的**样本级对齐**极度敏感。本脚本扫一遍延迟，
找出它的容忍窗口 —— 这直接决定我们的参考轨需要多准。

    python -m orchestrator.tests.test_aec_align --wav assets/ref_audio/ref_minicpm_signature.wav
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
_ALLOW = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-")


def db(x: float) -> float:
    return 10.0 * np.log10(max(float(x), 1e-12))


def energy(x: np.ndarray) -> float:
    return float(np.mean(x ** 2)) if x.size else 0.0


def load_wav(path: str) -> np.ndarray:
    with wave.open(path, "rb") as w:
        assert w.getframerate() == SR
        raw = w.readframes(w.getnframes())
        x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        if w.getnchannels() == 2:
            x = x.reshape(-1, 2).mean(axis=1)
    return x


async def run_case(url: str, mic: np.ndarray, far: np.ndarray, tag: str) -> np.ndarray:
    aec = AecClient(url, connection_id=f"align_{tag}")
    await aec.connect()
    outs = []
    recv = asyncio.create_task(aec.recv_loop(lambda s: outs.append(
        np.asarray(s).reshape(-1).copy())))
    for i in range(0, len(mic), DEFAULT_CHUNK):
        seg = mic[i:i + DEFAULT_CHUNK]
        ref = far[i:i + DEFAULT_CHUNK]
        if seg.size < DEFAULT_CHUNK:
            seg = np.pad(seg, (0, DEFAULT_CHUNK - seg.size))
            ref = np.pad(ref, (0, DEFAULT_CHUNK - ref.size))
        await aec.push(seg[None, :].astype(np.float32),
                       ref[None, :].astype(np.float32))
        await asyncio.sleep(DEFAULT_CHUNK / SR)
    z = np.zeros((1, DEFAULT_CHUNK), dtype=np.float32)
    await aec.push(z, z, is_end=True)
    try:
        await asyncio.wait_for(recv, timeout=20)
    except asyncio.TimeoutError:
        recv.cancel()
    await aec.close(send_end=False)
    return np.concatenate(outs) if outs else np.zeros(0, dtype=np.float32)


async def main_async(args) -> int:
    speech = load_wav(args.wav)[: int(args.seconds * SR)]
    e_near = energy(speech)
    print(f"近端语音能量 = {db(e_near):.1f} dB，时长 {len(speech)/SR:.1f}s")
    print()
    print("=" * 76)
    print("AEC 对齐容忍窗口扫描（far = 语音本身，mic = far 延迟D*0.5 + 语音）")
    print("-" * 76)
    print(f"{'延迟D':>8} {'ERLE':>9} {'近端保留':>10}   说明")
    print("-" * 76)

    for d_ms in (0, 1, 2, 5, 10, 20, 40, 80, 160, 320):
        d = int(d_ms * SR / 1000)
        far = speech.copy()
        echo = np.zeros_like(far)
        if d == 0:
            echo = far * 0.5
        else:
            echo[d:] = far[:-d] * 0.5
        mic = echo + speech

        out = await run_case(args.url, mic, far, f"d{d_ms}")
        n = min(len(mic), len(out))
        if n == 0:
            print(f"{d_ms:>6}ms {'—':>9}")
            continue
        erle = db(energy(mic[:n])) - db(energy(out[:n]))
        keep = db(energy(out[:n])) - db(e_near)
        note = ""
        if erle > 15 and keep < -12:
            note = "**近端也被消掉**（过度抑制）"
        elif erle > 15:
            note = "有效消除"
        elif erle > 6:
            note = "部分消除"
        else:
            note = "无消除"
        print(f"{d_ms:>6}ms {erle:>8.1f}dB {keep:>9.1f}dB   {note}")

    print("=" * 76)
    print()
    print("解读：")
    print("  · 若只有 D=0 有消除 → AEC 不做时延估计，要求调用方**样本级对齐**")
    print("  · 若 D=0 时近端保留极低 → 该模型是'回声+近端一起抑制'，需配合 SE 或调参")
    print("  · 容忍窗口 = 我们参考轨的精度要求")
    return 0


def main() -> None:
    p = argparse.ArgumentParser(description="AEC 对齐窗口扫描")
    p.add_argument("--url", default=DEFAULT_URL)
    p.add_argument("--wav", required=True)
    p.add_argument("--seconds", type=float, default=6.0)
    args = p.parse_args()
    raise SystemExit(asyncio.run(main_async(args)))


if __name__ == "__main__":
    main()
