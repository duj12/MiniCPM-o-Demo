"""Test if WS read timeout kills a session after 10s delay."""
import asyncio, json, ssl
import websockets

async def main():
    ssl_ctx = ssl.create_default_context()
    ssl_ctx.check_hostname = False
    ssl_ctx.verify_mode = ssl.CERT_NONE

    async with websockets.connect(
        "wss://192.168.89.106:8006/v1/realtime",
        max_size=128*1024*1024, ssl=ssl_ctx
    ) as ws:
        msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=10))
        print("queue_done received:", msg.get("type"))

        await ws.send(json.dumps({"type": "session.init", "payload": {"mode": "turn_based"}}))
        msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=10))
        sid = msg.get("session_id", "?")
        print("session created:", sid)

        # delay 10 seconds before sending input
        print("waiting 10s before sending input...")
        await asyncio.sleep(10)

        payload = {
            "type": "input.append",
            "input": {
                "messages": [{"role": "user", "content": "Hello"}],
                "streaming": False, "tts": {"enabled": False}, "use_tts_template": False,
            },
        }
        await ws.send(json.dumps(payload, ensure_ascii=False))
        print("input sent, waiting for response...")

        while True:
            msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=30))
            t = msg.get("type")
            if t == "response.output.delta" and msg.get("kind") == "text":
                print("delta:", msg.get("text", ""))
            elif t == "response.done":
                print("response done!")
                break
            elif t == "session.closed":
                print("session closed:", msg.get("reason"))
                return False
    return True

ok = asyncio.run(main())
print("PASS" if ok else "FAIL")
exit(0 if ok else 1)
