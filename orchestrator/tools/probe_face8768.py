"""8768（8PCM 门控）契约探针：抽一帧真实 MJPEG，配 8 声道 PCM 打两次 detect。

用法：
    python probe_face8768.py <face.mjpeg> [base_url]
"""
from __future__ import annotations

import base64
import json
import sys
import urllib.error
import urllib.request

import numpy as np

SR = 16000
NCH = 8
BLOCK = 1600          # 100ms


def first_jpeg(path: str) -> bytes:
    raw = open(path, "rb").read()
    i = raw.find(b"\xff\xd8")
    j = raw.find(b"\xff\xd9", i)
    return raw[i:j + 2]


def make_pcm(seed: int, amp: int = 200) -> np.ndarray:
    """8 声道交错 int16：ch2-5 放一点噪声（模拟近场麦），ch6-7 静音。"""
    rng = np.random.default_rng(seed)
    x = np.zeros((BLOCK, NCH), dtype=np.int16)
    x[:, 2:6] = rng.integers(-amp, amp, size=(BLOCK, 4))
    return x.reshape(-1)


def post(url: str, payload: dict) -> tuple[int, str]:
    body = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, r.read().decode()[:600]
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()[:600]


def main() -> None:
    jpeg_path = sys.argv[1]
    base = (sys.argv[2] if len(sys.argv) > 2
            else "http://192.168.89.106:8768")
    jpeg_b64 = base64.b64encode(first_jpeg(jpeg_path)).decode()
    for i in range(3):
        start_us = i * 100_000
        end_us = (i + 1) * 100_000
        payload = {
            "session_id": "probe-8pcm",
            "timestamp_us": end_us,
            "frame_jpeg_b64": jpeg_b64,
            "pcm8_base64": base64.b64encode(make_pcm(i).tobytes()).decode(),
            "pcm8_timestamp_us": start_us,
        }
        code, text = post(f"{base}/face/detect", payload)
        print(f"--- block {i}: pcm8@{start_us}us frame@{end_us}us -> {code}")
        print(text)
    print("--- close")
    print(post(f"{base}/face/close", {"session_id": "probe-8pcm"}))


if __name__ == "__main__":
    main()
