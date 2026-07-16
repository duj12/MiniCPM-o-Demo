"""
Test video inference through Gateway protocol.
Usage (on remote):
  cd /data/megastore/Projects/DuJing/code/MiniCPM-o-Demo
  /home/dujing/miniconda3/envs/py310/bin/python tests/test_video_inference.py \
    --video /data/megastore/Projects/DuJing/code/MiniCPM-o-Demo/data/input/88.mp4
"""

import asyncio
import json
import time
import argparse
import ssl
import base64

import websockets

GATEWAY = "wss://192.168.89.106:8006/v1/realtime"

SSL_CTX = ssl.create_default_context()
SSL_CTX.check_hostname = False
SSL_CTX.verify_mode = ssl.CERT_NONE

MAX_WS = 128 * 1024 * 1024


async def test_video(video_path: str, timeout_s: float = 120):
    """Send a video file for inference and print the response."""
    print(f"[test] Reading video: {video_path}")
    with open(video_path, "rb") as f:
        video_data = f.read()
    video_b64 = base64.b64encode(video_data).decode()
    print(f"[test] Video size: {len(video_data)} bytes, base64: {len(video_b64)} chars")

    async with websockets.connect(GATEWAY, max_size=MAX_WS, open_timeout=10, ssl=SSL_CTX) as ws:
        # Wait for queue_done
        msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=10))
        print(f"[test] Received: {msg.get('type')}")
        assert msg.get("type") == "session.queue_done"

        # Send session.init
        await ws.send(json.dumps({"type": "session.init", "payload": {"mode": "turn_based"}}))

        # Receive session.created
        msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=10))
        session_id = msg.get("session_id", "?")
        print(f"[test] Session created: {session_id}")

        # Send input.append with video
        t0 = time.perf_counter()
        await ws.send(json.dumps({
            "type": "input.append",
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "请描述这个视频中发生了什么"},
                            {"type": "video", "data": video_b64, "name": "test.mp4", "duration": 10},
                        ],
                    }
                ],
                "streaming": False,
                "tts": {"enabled": False},
                "use_tts_template": False,
                "omni_mode": True,
                "image": {"max_slice_nums": 1},
            },
        }, ensure_ascii=False))
        print(f"[test] input.append sent, waiting for response...")

        text = ""
        error = None
        while True:
            msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=timeout_s))
            t = msg.get("type")
            if t == "response.output.delta" and msg.get("kind") == "text":
                text += msg.get("text", "")
            elif t == "response.done":
                elapsed = (time.perf_counter() - t0) * 1000
                print(f"[test] Response done in {elapsed:.0f}ms")
                break
            elif t == "session.closed":
                reason = msg.get("reason", "")
                diagnostic = msg.get("diagnostic", {})
                error = diagnostic.get("message") or reason or "session.closed"
                print(f"[test] Session closed: reason={reason}, diagnostic={diagnostic}")
                break
            elif t == "error":
                error = msg.get("error", {}).get("message", str(msg))
                print(f"[test] Error: {error}")
                break

        # Close session
        await ws.send(json.dumps({"type": "session.close", "reason": "done"}))

        if error:
            print(f"\n❌ FAILED: {error}")
            return False
        else:
            print(f"\n✅ SUCCESS: {text[:200]}")
            return True


async def main():
    parser = argparse.ArgumentParser(description="Test video inference")
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--timeout", type=float, default=120, help="Timeout per video (s)")
    args = parser.parse_args()

    success = await test_video(args.video, args.timeout)
    exit(0 if success else 1)
 

if __name__ == "__main__":
    asyncio.run(main())
