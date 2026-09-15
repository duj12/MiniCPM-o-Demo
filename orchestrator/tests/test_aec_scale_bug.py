#!/usr/bin/env python3
"""查证：参考轨**量纲错误**是否会让 ERLE 虚高、同时把近端也消掉。

## 背景

真机日志里出现过 ``erle_db = 32.9`` —— 比离线实测的理论上限（对齐时
12.6dB）还高得多。这很可疑：ERLE 虚高通常意味着 **AEC 输出被压成了
静音**，那样"回声消除得很好"是假象，用户自己的声音也一起没了。

怀疑对象是参考轨的量纲 bug：``RefTrack.place()`` 早先没归一化，把 TTS 的
int16 PCM（±32768）原样存进轨里，而麦克风是 [-1,1] —— 送进 AEC 的 farend
比 nearend 大 **约 32768 倍**。

本脚本用同一对信号跑两次 AEC，只改 farend 的量纲：

  A. farend 归一化（[-1,1]，正确）
  B. farend 未归一化（int16 量纲，即 bug 复现）

对比两个量：
  · **ERLE**：mic 总能量 / 输出总能量（高 ≠ 好）
  · **近端保真**：输出能量 / 纯近端能量。**这个才是关键** —— 若它掉到
    极低（如 -40dB），说明近端语音被一起消掉了，那 ERLE 再高也是假的。

    python -m orchestrator.tests.test_aec_scale_bug --wav <far.wav> --near <near.wav>
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

_failures: list = []


def check(cond: bool, msg: str) -> None:
    print(f"  [{'OK' if cond else 'FAIL'}] {msg}")
    if not cond:
        _failures.append(msg)


def db(x: float) -> float:
    return 10.0 * np.log10(max(float(x), 1e-12))


def energy(x: np.ndarray) -> float:
    return float(np.mean(x ** 2)) if x.size else 0.0


def load_wav(path: str) -> np.ndarray:
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


async def run_aec(url: str, mic: np.ndarray, ref: np.ndarray,
                  tag: str) -> np.ndarray:
    _ALLOW = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
                 "0123456789_-")
    safe = "".join(c if c in _ALLOW else "_" for c in tag)[:40]
    aec = AecClient(url, connection_id=f"scl_{safe}")
    await aec.connect()
    outs: list = []
    recv = asyncio.create_task(aec.recv_loop(
        lambda s: outs.append(np.asarray(s).reshape(-1).copy())))
    for i in range(0, len(mic), DEFAULT_CHUNK):
        seg = mic[i:i + DEFAULT_CHUNK]
        r = ref[i:i + DEFAULT_CHUNK]
        if seg.size < DEFAULT_CHUNK:
            seg = np.pad(seg, (0, DEFAULT_CHUNK - seg.size))
            r = np.pad(r, (0, DEFAULT_CHUNK - r.size))
        await aec.push(seg[None, :].astype(np.float32),
                       r[None, :].astype(np.float32))
        await asyncio.sleep(DEFAULT_CHUNK / SR)
    await aec.push(np.zeros((1, DEFAULT_CHUNK), dtype=np.float32),
                   np.zeros((1, DEFAULT_CHUNK), dtype=np.float32), is_end=True)
    try:
        await asyncio.wait_for(recv, timeout=30)
    except asyncio.TimeoutError:
        recv.cancel()
    await aec.close(send_end=False)
    return np.concatenate(outs) if outs else np.zeros(0, dtype=np.float32)


async def main_async(args) -> int:
    far = load_wav(args.far)[: int(args.seconds * SR)]
    near = load_wav(args.near)
    if len(near) < len(far):
        near = np.pad(near, (0, len(far) - len(near)))
    near = near[: len(far)]

    # 完美对齐的 mic（离线实验条件）：回声 + 近端
    mic = far * 0.5 + near
    print(f"远端 {Path(args.far).name}  {len(far)/SR:.2f}s")
    print(f"近端 {Path(args.near).name}")
    print(f"mic 能量 {db(energy(mic)):.1f}dB，"
          f"其中回声 {db(energy(far*0.5)):.1f}dB / "
          f"近端 {db(energy(near)):.1f}dB")
    print()

    print("=" * 74)
    print("参考轨量纲影响（同一 mic，只改 farend 量纲）")
    print("-" * 74)

    cases = [
        ("A 归一化 [-1,1]（正确）", far),
        ("B int16 量纲（bug 复现）", far * 32768.0),
        ("C 放大 100×", far * 100.0),
    ]
    res = {}
    for label, ref in cases:
        out = await run_aec(args.url, mic, ref, label.split()[0])
        n = min(len(mic), len(out))
        if n == 0:
            print(f"  {label:<26} — 无输出")
            continue
        e_in, e_out = energy(mic[:n]), energy(out[:n])
        e_near = energy(near[:n])
        erle = db(e_in) - db(e_out)
        # 近端保真：输出能量 / 纯近端能量。~0dB 说明近端完好；
        # 大幅负值说明近端被一起消掉了（ERLE 再高也是假的）
        keep = db(e_out) - db(e_near) if e_near > 0 else float("nan")
        res[label[0]] = (erle, keep)
        flag = ""
        if keep < -20:
            flag = "  ← ⚠️ 近端被消光，ERLE 是假象"
        print(f"  {label:<26} ERLE {erle:>6.1f}dB   近端保真 {keep:>7.1f}dB{flag}")

    print("=" * 74)
    print()
    if "A" in res and "B" in res:
        erle_a, keep_a = res["A"]
        erle_b, keep_b = res["B"]
        # 归一的必要性：至少不能比正确量纲差
        check(keep_a > -12, f"正确量纲下近端保留（{keep_a:+.1f}dB > -12dB）")
        check(keep_b >= -12,
              f"量纲错误也未把近端消光（{keep_b:+.1f}dB）—— "
              "说明它不是「回声消不掉」的主因")
        print()
        if keep_a > -12 and keep_b > -12 and abs(erle_a - erle_b) < 4:
            print("→ **假设不成立**：量纲错误对 ERLE / 近端保真的影响都很小。")
            print("  归一化仍然是正确做法（量纲不该靠模型兜底），但它**不是**")
            print("  「回声消不掉」的原因 —— 真机 erle_db=32.9 另有出处，")
            print("  需转向其他方向排查（见下方提示）。")
        elif keep_b < -20:
            print("→ 量纲错误导致近端被消光（ERLE 再高也是假象）")

    print()
    if _failures:
        print(f"FAILED: {len(_failures)} 项")
        for f in _failures:
            print(f"  - {f}")
        return 1
    print("通过")
    return 0


def main() -> None:
    here = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser(description="查证参考轨量纲对 AEC 的影响")
    p.add_argument("--url", default=DEFAULT_URL)
    p.add_argument("--far", default=str(here / "assets/ref_audio/"
                                        "ref_minicpm_signature.wav"))
    p.add_argument("--near", default=str(here / "assets/ref_audio/"
                                         "ref_en_dlc_1.wav"))
    p.add_argument("--seconds", type=float, default=6.0)
    args = p.parse_args()
    raise SystemExit(asyncio.run(main_async(args)))


if __name__ == "__main__":
    main()
