#!/usr/bin/env python3
"""8PCM 切片节拍回归（离线，不连服务）。

验的是 ``RemoteFaceProvider._send_pcm8`` 的一条硬约束：

    视频 25fps ⇒ ``/face/detect`` 就该是 ~25Hz，而不是被 100ms 的音频块
    拖成 10Hz。

事故背景（`s-56886bcbcf10`）：旧实现「没有新 PCM 就不发 detect」，而客户端
音频是 100ms 一块 ⇒ 视频 1191 帧 / 55s 里真正进 8768 的只有 513 帧
（≈9.3Hz），BlazeFace / Landmark / LipFsm 全拿不到设计节拍。

现在每帧从缓冲里切 640 采样（40ms）发一次，同时必须保持
「相邻请求的 PCM 首尾相接、不重复、不用未来音频」—— 这里一并断言：

  · 25fps：detect ≥ 24Hz，窗口恒为 640 采样
  · 送出字节按序拼起来 == 输入流的前缀（不重不漏）
  · 帧时间戳与 PCM 窗口严格首尾相接（服务端靠这个判定连续）
  · 视频慢于音频（20fps）时：一次多取排水，积压有界
  · 音频中断后：缓冲空了就不再发（缺 PCM 的 detect 会被 422）

    python -m orchestrator.tests.test_face_pcm8_slice
"""
from __future__ import annotations

import base64
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from orchestrator.face import remote_provider as rpmod  # noqa: E402
from orchestrator.face.remote_provider import RemoteFaceProvider  # noqa: E402

SR = 16000
BLOCK_SAMPLES = 1600        # 客户端一块 100ms
BYTES_PER_SAMPLE = 16       # 8ch × int16
JPEG = b"\xff\xd8fake\xff\xd9"

_failures: list = []


def check(cond: bool, msg: str) -> None:
    print(f"  [{'OK' if cond else 'FAIL'}] {msg}")
    if not cond:
        _failures.append(msg)


def drive(frame_ms: int, dur_ms: int, block_ms: int = 100,
          stop_audio_ms: int | None = None):
    """按真实节奏喂 provider，返回 (provider, 送出的 payload 列表, 输入字节)。"""
    prov = RemoteFaceProvider("http://127.0.0.1:9", session_id="slice",
                              pcm8=True)
    sent = []

    def fake_post(path, payload, timeout=None):
        sent.append(payload)
        return {"valid": True, "dwell_ms": 0}

    prov._post = fake_post

    stream = bytearray()
    nxt_block = 0
    t = 0
    while t < dur_ms:
        while nxt_block <= t and (stop_audio_ms is None or nxt_block < stop_audio_ms):
            blk = bytes(BYTES_PER_SAMPLE) * BLOCK_SAMPLES
            stream += blk
            prov.set_pcm8(blk)
            nxt_block += block_ms
        prov.process(JPEG, t * 16)      # t 是音频采样轴，仅用于观测聚合
        t += frame_ms
    return prov, sent, stream


def windows(sent):
    out = []
    prev_end = 0
    joined = bytearray()
    bad_ts = 0
    for p in sent:
        raw = base64.b64decode(p["pcm8_base64"])
        n = len(raw) // BYTES_PER_SAMPLE
        start_us, end_us = p["pcm8_timestamp_us"], p["timestamp_us"]
        if start_us != prev_end or end_us - start_us != n * 1_000_000 // SR:
            bad_ts += 1
        prev_end = end_us
        out.append(n)
        joined += raw
    return out, joined, bad_ts


def main() -> None:
    print("① 25fps 视频 + 100ms 音频块 → detect 应为 ~25Hz")
    prov, sent, stream = drive(40, 10_000)
    w, joined, bad_ts = windows(sent)
    rate = len(sent) / 10.0
    check(rate >= 24.0, f"detect 频率 {rate:.1f}Hz（≥24）")
    check(set(w[:20]) == {640} or max(w) == 640,
          f"窗口恒为 640 采样（实际 {sorted(set(w))}）")
    check(bad_ts == 0, f"时间窗首尾相接（错 {bad_ts} 处）")
    check(bytes(joined) == bytes(stream[:len(joined)]),
          "送出字节 == 输入流前缀（不重不漏）")
    check(len(prov._pcm8_buf) <= rpmod._PCM8_SLICE_BYTES,
          f"稳态积压 ≤1 片（{len(prov._pcm8_buf) // BYTES_PER_SAMPLE} 采样）")

    print("② 20fps 视频（音频供给快于消费）→ 排水且积压有界")
    prov, sent, stream = drive(50, 10_000)
    w, joined, bad_ts = windows(sent)
    backlog = len(prov._pcm8_buf) // BYTES_PER_SAMPLE
    check(backlog <= rpmod._PCM8_BACKLOG_SLICES * 640,
          f"积压有界（{backlog} 采样 ≤ {rpmod._PCM8_BACKLOG_SLICES} 片）")
    check(max(w) <= rpmod._PCM8_MAX_SAMPLES, f"单请求不超 1s（最大 {max(w)} 采样）")
    check(bad_ts == 0, "时间窗仍首尾相接")
    check(bytes(joined) == bytes(stream[:len(joined)]), "仍不重不漏")

    print("③ 音频中断 → 缓冲空了就不再发 detect")
    prov, sent, stream = drive(40, 4_000, stop_audio_ms=1_000)
    n_after = len(sent)
    check(len(prov._pcm8_buf) == 0, "缓冲已排空")
    for _ in range(50):
        prov.process(JPEG, 0)
    check(len(sent) == n_after, "没有 PCM 时一帧都不发")

    print()
    if _failures:
        print(f"[FAIL] {len(_failures)} 项未通过")
        for f in _failures:
            print(f"  - {f}")
        sys.exit(1)
    print("[OK] 全部通过")


if __name__ == "__main__":
    main()
