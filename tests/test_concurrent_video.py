"""Test two concurrent video inference sessions."""
import asyncio, json, ssl, base64, time
import websockets

async def run_video(video_path, name):
    ssl_ctx = ssl.create_default_context()
    ssl_ctx.check_hostname = False
    ssl_ctx.verify_mode = ssl.CERT_NONE
    with open(video_path, "rb") as f:
        video_b64 = base64.b64encode(f.read()).decode()

    async with websockets.connect(
        "wss://192.168.89.106:8006/v1/realtime",
        max_size=128*1024*1024, ssl=ssl_ctx
    ) as ws:
        msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=10))
        await ws.send(json.dumps({"type": "session.init", "payload": {"mode": "turn_based"}}))
        msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=10))

        t0 = time.perf_counter()
        payload = {
            "type": "input.append",
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "describe this video"},
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
        }
        await ws.send(json.dumps(payload, ensure_ascii=False))

        text = ""
        while True:
            msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=180))
            t = msg.get("type")
            if t == "response.output.delta" and msg.get("kind") == "text":
                text += msg.get("text", "")
            elif t == "response.done":
                await ws.send(json.dumps({"type": "session.close", "reason": "done"}))
                elapsed = (time.perf_counter() - t0) * 1000
                print(f"{name}: OK in {elapsed:.0f}ms")
                return True
            elif t == "session.closed":
                print(f"{name}: FAILED reason={msg.get('reason')}")
                return False

async def main():
    t0 = time.perf_counter()
    results = await asyncio.gather(
        run_video("/data/megastore/Projects/DuJing/code/MiniCPM-o-Demo/data/input/88.mp4", "88.mp4"),
        run_video("/data/megastore/Projects/DuJing/code/MiniCPM-o-Demo/data/input/121.mp4", "121.mp4"),
        return_exceptions=True,
    )
    elapsed = (time.perf_counter() - t0) * 1000

    ok = 0
    for r in results:
        if isinstance(r, Exception):
            print(f"Exception: {r}")
        elif r:
            ok += 1
    print(f"Result: {ok}/2 passed in {elapsed:.0f}ms")
    exit(0 if ok == 2 else 1)

if __name__ == "__main__":
    asyncio.run(main())
