#!/usr/bin/env python3
"""AEC 有效性的离线对照实验。

⚠️ **本脚本的结论已作废，请用 `gen_aec_report_assets.py`。**

作废原因：本脚本构造的 far 与 near **是同一段语音**（``far = speech.copy()``），
于是 mic 里除了回声没有独立的近端内容 —— "消掉回声"就等于"消掉一切"。
它报出的「近端被过度抑制 -49dB」是**实验设计的假象**，不是模型缺陷。

用**两段不同语音**复测（`gen_aec_report_assets.py`）后，近端保留
+0.7~+1.5 dB，AEC 并不会吞掉近端语音。

本脚本保留作历史记录，不建议再据此下结论。
"""

实验设计：
  1. 取一段真实语音 ``speech``
  2. 构造 far（扬声器信号）与 mic = far 经延迟 D 衰减 + speech（近端）
  3. 送进 AEC 服务（nearend=mic, farend=far）
  4. 对比输出与输入的 ERLE，以及**近端语音是否被保留**

两个必看的量：
  · **ERLE**：回声被消掉多少 dB。太低（<6dB）说明 AEC 没干活。
  · **近端保真**：近端语音的能量是否还在。如果 ERLE 很高但近端也没了，
    那是「过度抑制」——听感上同样不可用。

    python -m orchestrator.tests.test_aec_effect --wav assets/ref_audio/ref_minicpm_signature.wav
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
from orchestrator.audio.ref_track import AcousticDelayTracker  # noqa: E402

DEFAULT_URL = "ws://192.168.88.253:30255/ws/asr_frontend"


def db(x: float) -> float:
    return 10.0 * np.log10(max(float(x), 1e-12))


def energy(x: np.ndarray) -> float:
    return float(np.mean(x ** 2)) if x.size else 0.0


def load_wav(path: str) -> np.ndarray:
    with wave.open(path, "rb") as w:
        assert w.getframerate() == SR, f"需要 {SR}Hz，得到 {w.getframerate()}"
        raw = w.readframes(w.getnframes())
        x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        if w.getnchannels() == 2:
            x = x.reshape(-1, 2).mean(axis=1)
    return x


def build_case(speech: np.ndarray, delay: int, echo_gain: float
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回 (mic, far, speech_aligned)。

    ``far`` 是扬声器信号（用语音本身模拟 TTS 播放）。
    ``mic`` = far 延迟 delay 并衰减 echo_gain（回声） + speech（近端）。
    """
    far = speech.copy()
    echo = np.zeros_like(far)
    if delay <= 0:
        echo = far * echo_gain
    else:
        echo[delay:] = far[:-delay] * echo_gain
    mic = echo + speech
    return mic, far, speech


async def run_case(url: str, mic: np.ndarray, far: np.ndarray,
                   label: str) -> np.ndarray:
    """把一对信号送进 AEC，返回清洗后的输出。"""
    # connection_id 只允许 [A-Za-z0-9_-]。注意 str.isalnum() 对中文返回
    # True，必须用显式 ASCII 白名单。
    _ALLOW = set("abcdefghijklmnopqrstuvwxyz"
                 "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-")
    safe = "".join(c if c in _ALLOW else "_" for c in label)[:40]
    aec = AecClient(url, connection_id=f"probe_{safe}")
    await aec.connect()
    outs = []

    def on_audio(seg):
        outs.append(np.asarray(seg).reshape(-1).copy())

    recv = asyncio.create_task(aec.recv_loop(on_audio))
    n = len(mic)
    for i in range(0, n, DEFAULT_CHUNK):
        seg = mic[i:i + DEFAULT_CHUNK]
        ref = far[i:i + DEFAULT_CHUNK]
        if seg.size < DEFAULT_CHUNK:
            seg = np.pad(seg, (0, DEFAULT_CHUNK - seg.size))
            ref = np.pad(ref, (0, DEFAULT_CHUNK - ref.size))
        await aec.push(seg[None, :].astype(np.float32),
                       ref[None, :].astype(np.float32))
        await asyncio.sleep(DEFAULT_CHUNK / SR)
    # is_end 触发尾部 flush
    await aec.push(np.zeros((1, DEFAULT_CHUNK), dtype=np.float32),
                   np.zeros((1, DEFAULT_CHUNK), dtype=np.float32), is_end=True)
    try:
        await asyncio.wait_for(recv, timeout=20)
    except asyncio.TimeoutError:
        recv.cancel()
    await aec.close(send_end=False)
    if not outs:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(outs)


async def main_async(args) -> int:
    speech = load_wav(args.wav)
    if args.seconds:
        speech = speech[: int(args.seconds * SR)]
    print(f"近端语音: {args.wav}  {len(speech)/SR:.2f}s  "
          f"能量={db(energy(speech)):.1f} dB")

    print()
    print("=" * 78)
    print("AEC 有效性对照实验")
    print("-" * 78)
    print(f"{'场景':<26} {'输入能量':>9} {'输出能量':>9} {'ERLE':>8} "
          f"{'近端保留':>9}  判定")
    print("-" * 78)

    results = []
    cases = [
        ("无回声(纯近端)", 0, 0.0),
        ("回声 D=100ms 增益0.3", int(0.1 * SR), 0.3),
        ("回声 D=100ms 增益0.6", int(0.1 * SR), 0.6),
        ("回声 D=20ms  增益0.6", int(0.02 * SR), 0.6),
        ("回声 D=0     增益0.6", 0, 0.6),
    ]
    for label, delay, gain in cases:
        mic, far, near = build_case(speech, delay, gain)
        out = await run_case(args.url, mic, far, label.replace(" ", ""))

        n = min(len(mic), len(out))
        if n == 0:
            print(f"{label:<26} — 无输出")
            results.append((label, None, None))
            continue
        e_in, e_out = energy(mic[:n]), energy(out[:n])
        erle = db(e_in) - db(e_out)
        # 近端保留：输出能量 / 纯近端能量。~1 说明近端没被削
        keep = db(e_out) - db(energy(near[:n]))
        verdict = "OK" if erle > 6 else ("弱" if erle > 2 else "无消除")
        if gain > 0 and keep < -12:
            verdict += " / 近端被过度抑制"
        print(f"{label:<26} {db(e_in):>8.1f}dB {db(e_out):>8.1f}dB "
              f"{erle:>7.1f}dB {keep:>8.1f}dB  {verdict}")
        results.append((label, erle, keep))

    print("=" * 78)
    print()
    # 判定
    echo_cases = [r for r in results if r[1] is not None and "纯近端" not in r[0]]
    if not echo_cases:
        print("FAILED: 没有有效的回声场景")
        return 1
    best = max(r[1] for r in echo_cases)
    print(f"最强消除: {best:.1f} dB")
    if best < 6:
        print("→ **AEC 几乎没有消除作用** —— 问题在 AEC 服务或 farend 本身，"
              "不在我们的参考轨")
        return 1
    if best < 12:
        print("→ 消除偏弱。可能原因：延迟补偿不准 / 线性回声路径被破坏"
              "（浏览器 AGC 未关）")
        return 1
    print("→ AEC 工作正常。若真机仍有回声，检查前端是否真的关了 AGC")
    return 0


def main() -> None:
    p = argparse.ArgumentParser(description="AEC 有效性对照实验")
    p.add_argument("--url", default=DEFAULT_URL)
    p.add_argument("--wav", required=True)
    p.add_argument("--seconds", type=float, default=8.0)
    args = p.parse_args()
    raise SystemExit(asyncio.run(main_async(args)))


if __name__ == "__main__":
    main()
