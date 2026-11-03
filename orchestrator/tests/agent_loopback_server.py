"""Optional real Agent server harness; never imports a production model/runtime."""
import asyncio
import importlib.util
import json
from pathlib import Path
import sys

source = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(source / "src"))
spec = importlib.util.spec_from_file_location("agent_test_fakes", source / "tests" / "voice_fakes.py")
fakes = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fakes)
from api.grpc_service import make_server
from agent_platform.voice import bridge
bridge._SETTLE_MIN_SECONDS = 0


class Runtime(fakes.FakeRuntime):
    async def process_message_stream(self, user_message, **kwargs):
        self.calls.append(user_message)
        self.session_ids.append(kwargs.get("session_id"))
        self.mcp_servers_calls.append(kwargs.get("mcp_servers"))
        try:
            if user_message.startswith("error"):
                yield "text", "这个旧片段必须丢弃。"
                raise RuntimeError("controlled model failure")
            if user_message.startswith("slow"):
                yield "tool_ack", {"name": "query", "text": "正在查询。"}
                await asyncio.sleep(0.05)
            if user_message.startswith("blocked"):
                yield "tool_ack", {"name": "query", "text": "正在处理。"}
                await asyncio.Event().wait()
            yield "text", "正文:" + user_message + "。"
            yield "done", {"response": "正文:" + user_message + "。", "session_ended": False}
            self.finished += 1
        except asyncio.CancelledError:
            self.cancelled += 1
            raise


async def main():
    runtime = Runtime()
    server, port = make_server(runtime, port=0, fallback_delay=0, ms_per_char=0)
    await server.start()
    print(json.dumps({"port": port}), flush=True)
    try:
        while True:
            command = await asyncio.to_thread(sys.stdin.readline)
            if not command or command.strip() == "quit":
                break
            if command.strip() == "stats":
                print(json.dumps({"calls": runtime.calls, "starts": runtime.start_calls,
                                  "deleted": runtime.sessions.deleted,
                                  "active": list(runtime.sessions.active),
                                  "cancelled": runtime.cancelled,
                                  "state_updates": runtime.state_updates}), flush=True)
    finally:
        await server.stop(0)


asyncio.run(main())
