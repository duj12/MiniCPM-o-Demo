#!/usr/bin/env python3
"""生成 AEC 问题报告用的音频样本 + 实测数据表。

产出到 ``--out`` 目录（默认 /tmp/aec_report）：
  · 01_near_only.wav    近端语音（用户说话，无回声）—— 干净参照
  · 02_far_reference.wav 远端参考（模拟 TTS 播放）
  · 03_mic_input.wav    麦克风输入 = 回声(D=284ms) + 近端
  · 04_aec_output.wav   AEC 输出 —— **回声仍在**（这就是问题现象）
  · 05_case_D0_input.wav / 06_case_D0_output.wav
                        最好情况（D=0 完美对齐）的输入/输出
  · metrics.csv         各场景的 ERLE / 回声抑制 / 近端保留

在 106 上跑（需要 AEC 服务）::

    python -m orchestrator.tests.gen_aec_report_assets \
        --far assets/ref_audio/ref_minicpm_signature.wav \
        --near assets/ref_audio/ref_en_dlc_1.wav \
        --out /tmp/aec_report
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import struct
import sys
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from orchestrator.aec.client import DEFAULT_CHUNK, SR, AecClient  # noqa: E402

DEFAULT_URL = "ws://192.168.88.253:30255/ws/asr_frontend"


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


def write_wav(path: Path, x: np.ndarray) -> None:
    """写 16kHz 单声道 int16 wav。"""
    data = np.clip(x * 32767.0, -32768, 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(data.tobytes())


async def push_pair(url: str, mic: np.ndarray, ref: np.ndarray,
                    tag: str) -> np.ndarray:
    aec = AecClient(url, connection_id=f"rep_{tag}")
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
        await aec.push(m[None, :].astype(np.float32),
                       r[None, :].astype(np.float32))
        await asyncio.sleep(DEFAULT_CHUNK / SR)
    z = np.zeros((1, DEFAULT_CHUNK), dtype=np.float32)
    await aec.push(z, z, is_end=True)
    try:
        await asyncio.wait_for(recv, timeout=20)
    except asyncio.TimeoutError:
        recv.cancel()
    await aec.close(send_end=False)
    return np.concatenate(outs) if outs else np.zeros(0, dtype=np.float32)


def build_case(far: np.ndarray, near: np.ndarray, d: int, gain: float):
    """构造 (mic, echo, near)，其中 ``mic[t] = far[t-d]*gain + near[t]``。

    即回声是 far **延迟 d** —— 这决定了预对齐该往哪个方向移。
    """
    n = min(len(far), len(near))
    far, near = far[:n], near[:n]
    echo = np.zeros(n, dtype=np.float32)
    if d <= 0:
        echo = far * gain
    else:
        echo[d:] = far[:n - d] * gain
    return (echo + near).astype(np.float32), echo, near


def delay_by(x: np.ndarray, d: int) -> np.ndarray:
    """把 x **延后** d 个采样：``out[t] = x[t-d]``。

    ⚠️ 这是预对齐的正确方向。因为 ``mic[t]`` 里的回声来自 ``far[t-d]``，
    要让它与 mic 对齐，ref 必须**延后** d，而不是提前。
    （早先写成提前，得出"预对齐有害"的错误结论。）
    """
    if d == 0:
        return x.copy()
    out = np.zeros_like(x)
    if d > 0:
        out[d:] = x[:len(x) - d]
    else:
        out[:d] = x[-d:]
    return out


def metrics(mic: np.ndarray, out: np.ndarray, echo: np.ndarray,
            near: np.ndarray) -> dict:
    n = min(len(mic), len(out), len(echo), len(near))
    if n == 0:
        return {}
    e_mic, e_out = energy(mic[:n]), energy(out[:n])
    e_echo, e_near = energy(echo[:n]), energy(near[:n])
    residual = max(e_out - e_near, 1e-12)
    return {
        "mic_db": round(db(e_mic), 1),
        "out_db": round(db(e_out), 1),
        "near_db": round(db(e_near), 1),
        "echo_db": round(db(e_echo), 1),
        "erle_db": round(db(e_mic) - db(e_out), 1),
        "echo_suppression_db": round(db(e_echo) - db(residual), 1),
        "near_preserved_db": round(db(e_out) - db(e_near), 1),
    }


async def main_async(args) -> int:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    far = load_wav(args.far)[: int(args.seconds * SR)]
    near = load_wav(args.near)[: int(args.seconds * SR)]
    # 两者等长对齐
    n = min(len(far), len(near))
    far, near = far[:n], near[:n]

    print(f"far : {Path(args.far).name}  {len(far)/SR:.2f}s")
    print(f"near: {Path(args.near).name}  {len(near)/SR:.2f}s")
    print(f"输出目录: {out_dir}")
    print()

    # 干净参照
    write_wav(out_dir / "01_near_only.wav", near)
    write_wav(out_dir / "02_far_reference.wav", far)

    rows = []
    # 场景一：真机实测延迟 284ms（问题现场）
    for label, d_ms, write_io in (
        ("D=0ms(完美对齐)", 0, True),
        ("D=100ms", 100, False),
        ("D=284ms(真机实测)", 284, True),
        ("D=500ms", 500, False),
    ):
        d = int(d_ms * SR / 1000)
        mic, echo, near_a = build_case(far, near, d, args.gain)
        print(f"[{label}] 送 AEC ...", flush=True)
        out = await push_pair(args.url, mic, far, f"d{d_ms}")
        m = metrics(mic, out, echo, near_a)
        rows.append({"case": label, "prealigned": "no", **m})
        print(f"   ERLE={m['erle_db']}dB  回声抑制={m['echo_suppression_db']}dB "
              f"近端保留={m['near_preserved_db']}dB")

        if write_io:
            mk = "03" if d_ms == 284 else "05"
            ok = "04" if d_ms == 284 else "06"
            write_wav(out_dir / f"{mk}_{'mic_input' if d_ms==284 else 'case_D0_input'}.wav", mic)
            write_wav(out_dir / f"{ok}_{'aec_output' if d_ms==284 else 'case_D0_output'}.wav", out)

    # 场景二：预对齐 —— 验证"调用方能否通过对齐来适配 AEC"
    # ref 延后 D，使 ref[t] = far[t-D]，与 mic 里的回声分量对齐。
    print()
    print("[预对齐对照] 把 ref 延后 D 后再送（与 mic 里的回声对齐）...", flush=True)
    for d_ms in (0, 100, 284, 500):
        d = int(d_ms * SR / 1000)
        mic, echo, near_a = build_case(far, near, d, args.gain)
        out = await push_pair(args.url, mic, delay_by(far, d), f"pre{d_ms}")
        m = metrics(mic, out, echo, near_a)
        rows.append({"case": f"D={d_ms}ms", "prealigned": "yes", **m})
        print(f"   D={d_ms}ms 预对齐 → ERLE={m['erle_db']}dB "
              f"回声抑制={m['echo_suppression_db']}dB "
              f"近端保留={m['near_preserved_db']}dB")
        # 生产链路 = D=284ms + 预对齐。这是**最关键**的样本 ——
        # 11.4dB 抑制下回声仍可辨识，正是用户听到的现象。
        if d_ms == 284:
            write_wav(out_dir / "07_prod_input.wav", mic)
            write_wav(out_dir / "08_prod_aec_output.wav", out)
            print("   → 已写生产链路样本 07/08")

    # 数据表
    csv_path = out_dir / "metrics.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print()
    print(f"数据表: {csv_path}")
    print(f"音频: {len(list(out_dir.glob('*.wav')))} 个 wav")
    return 0


def main() -> None:
    p = argparse.ArgumentParser(description="生成 AEC 报告素材")
    p.add_argument("--url", default=DEFAULT_URL)
    p.add_argument("--far", required=True)
    p.add_argument("--near", required=True)
    p.add_argument("--out", default="/tmp/aec_report")
    p.add_argument("--seconds", type=float, default=5.0)
    p.add_argument("--gain", type=float, default=0.5)
    args = p.parse_args()
    raise SystemExit(asyncio.run(main_async(args)))


if __name__ == "__main__":
    main()
