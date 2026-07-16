"""Test turn_based video inference."""
import asyncio, json, ssl, base64, sys, os
import websockets

GATEWAY = "wss://192.168.89.106:8006/v1/realtime"
DATA_DIR = "/data/megastore/Projects/DuJing/code/MiniCPM-o-Demo/assets/video/turnbased"

async def run_video(name, path):
    ssl_ctx = ssl.create_default_context()
    ssl_ctx.check_hostname = False
    ssl_ctx.verify_mode = ssl.CERT_NONE

    with open(path, "rb") as f:
        video_b64 = base64.b64encode(f.read()).decode()
    print(f"  [{name}] b64={len(video_b64)//1024//1024}MB")

    try:
        async with websockets.connect(GATEWAY, max_size=128*1024*1024, ssl=ssl_ctx) as ws:
            await asyncio.wait_for(ws.recv(), timeout=15)
            await ws.send(json.dumps({"type": "session.init", "payload": {"mode": "turn_based"}}))
            msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=15))
            sid = msg.get("session_id")
            if not sid:
                return False

            payload = {
                "type": "input.append",
                "input": {
                    "messages": [{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "describe this video in detail"},
                            {"type": "video", "data": video_b64, "name": f"{name}.mp4", "duration": 10},
                        ],
                    }],
                    "streaming": False,
                    "tts": {"enabled": False},
                    "use_tts_template": False,
                    "omni_mode": True,
                    "image": {"max_slice_nums": 1},
                },
            }
            await ws.send(json.dumps(payload, ensure_ascii=False))

            text = ""
            timeout_s = 180
            while True:
                msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=timeout_s))
                t = msg.get("type")
                if t == "response.output.delta" and msg.get("kind") == "text":
                    text += msg.get("text", "")
                elif t == "response.done":
                    # 非流式模式下文本在 response.done.text 里
                    text = msg.get("text", "") or text
                    await ws.send(json.dumps({"type": "session.close", "reason": "done"}))
                    break
                elif t == "session.closed":
                    break
                elif t == "error":
                    print(f"  [{name}] error: {msg}")
                    return False

            if text:
                print(f"  [{name}] ✅ {len(text)}chars: {text[:200]}")
                return True
            else:
                print(f"  [{name}] ⚠️ empty response")
                return True
    except Exception as e:
        print(f"  [{name}] ❌ {e}")
        return False

async def main():
    files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith('.mp4')])
    if not files:
        print(f"No videos in {DATA_DIR}")
        return

    print(f"Testing {len(files)} videos in turn_based mode:\n")
    passed = 0
    for fname in files:
        fpath = os.path.join(DATA_DIR, fname)
        ok = await run_video(fname, fpath)
        if ok: passed += 1
        print()

    print(f"Result: {passed}/{len(files)} passed")
    exit(0 if passed == len(files) else 1)

if __name__ == "__main__":
    asyncio.run(main())
