#!/usr/bin/env python3
"""端到端验证：InteractionCore 的 ExpressionSink 真能控制编排服务的播报/停止。

回答的问题：Policy 判出 GREET/UTTER 时，模板真的播了吗？判 YIELD/END 时，
**正在播的内容真的被掐断了吗**（包括 Agent 直接调 /v1/speak 排的那条）？

两端都要真起：编排服务（8100）+ InteractionCore（50051）。本脚本自己扮演
浏览器连上去，所以能观测到「服务端是否真的下发了 tts.audio / tts.cancel」。

    python -m orchestrator.tests.test_ic_expression
"""
from __future__ import annotations

import argparse
import asyncio
import json
import ssl
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

_failures: list = []


def check(cond: bool, msg: str) -> None:
    print(f"  [{'OK' if cond else 'FAIL'}] {msg}")
    if not cond:
        _failures.append(msg)


async def main() -> None:
    p = argparse.ArgumentParser(description="IC ExpressionSink 端到端验证")
    p.add_argument("--orch", default="192.168.89.106:8100")
    p.add_argument("--ic", default="127.0.0.1:50051")
    p.add_argument("--text", default="测试阻止播报，这是一段比较长的文本，"
                                    "用来确保有足够的时间在播放中途把它掐断。")
    args = p.parse_args()

    import websockets

    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    url = f"wss://{args.orch}/v1/orchestrator"
    events: list = []

    async with websockets.connect(url, ssl=ctx, max_size=32 * 1024 * 1024) as ws:
        await ws.send(json.dumps({
            "type": "session.start",
            "aec_mode": "off",
            "identity": {"page": "ic-expression-test"},
            "ic": {"enabled": True, "grpc": args.ic},
        }))
        # 等 session.ready
        sid = None
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline and sid is None:
            raw = await asyncio.wait_for(ws.recv(), timeout=20)
            m = json.loads(raw) if isinstance(raw, str) else {}
            events.append(m)
            if m.get("type") == "session.ready":
                sid = m.get("session_id")
        check(sid is not None, f"会话建立（sid={sid}）")
        if sid is None:
            return

        # ---- ① 直接从 IC 侧调 ExpressionSink.play_template ----
        print("\n① IC.play_template() → 应该真的播出来")
        await asyncio.to_thread(_ic_play, args.ic, args.text)
        got_audio = await _wait_for(ws, events, "tts.audio", timeout=25)
        check(got_audio, "编排服务下发了 tts.audio（模板真的去合成/播了）")

        # ---- ② 从 IC 侧调 stop()，应当掐断 ----
        print("\n② IC.stop() → 应该立刻掐断正在播的内容")
        time.sleep(1.0)          # 让它真的开始播一点
        await asyncio.to_thread(_ic_stop, args.ic)
        got_cancel = await _wait_for(ws, events, "tts.cancel", timeout=10)
        check(got_cancel, "编排服务下发了 tts.cancel（播报被掐断）")

        # ---- ③ Agent 播的内容也该被 IC 掐断（不问来源）----
        print("\n③ Agent 经 /v1/speak 排的播报，IC.stop() 也该掐断")
        import urllib.request
        body = json.dumps({"text": "这是假装 Agent 发的回复，"
                                   "它同样应该被 IC 的停止命令掐断。"})
        req = urllib.request.Request(
            f"https://{args.orch}/v1/speak",
            data=body.encode("utf-8"),
            headers={"Content-Type": "application/json", "X-Session-Id": sid},
            method="POST")
        try:
            with urllib.request.urlopen(req, timeout=10, context=ctx):
                pass
        except Exception as exc:  # noqa: BLE001
            print(f"    （Agent 播报请求失败：{exc}）")
        await _wait_for(ws, events, "tts.audio", timeout=25)
        events.clear()
        time.sleep(1.0)
        await asyncio.to_thread(_ic_stop, args.ic)
        got_cancel2 = await _wait_for(ws, events, "tts.cancel", timeout=10)
        check(got_cancel2, "Agent 排的播报也被掐断（不问来源）")

        await ws.send(json.dumps({"type": "session.stop"}))

    print("\n" + "=" * 60)
    types = {}
    for m in events:
        types[m.get("type")] = types.get(m.get("type"), 0) + 1
    print(f"  本轮收到消息: {types}")
    if _failures:
        print(f"FAILED: {len(_failures)} 项")
        for f in _failures:
            print("  -", f)
        raise SystemExit(1)
    print("通过")


def _ic_play(ic_target: str, text: str) -> None:
    from interaction import ExpressionSink
    # 106 的编排服务是**自签证书**。生产应传 ca_file=<cert.pem>；
    # 这里为了脚本能在任意机器跑，显式关校验。
    s = ExpressionSink("https://192.168.89.106:8100", timeout=5.0,
                       verify_ssl=False)
    s.play_template(text)
    s.flush(3.0)
    s.close()
    print(f"    IC play_template: plays={s.plays} failures={s.failures}")


def _ic_stop(ic_target: str) -> None:
    from interaction import ExpressionSink
    s = ExpressionSink("https://192.168.89.106:8100", timeout=5.0,
                       verify_ssl=False)
    s.stop()
    s.close()
    print(f"    IC stop: stops={s.stops} failures={s.failures}")


async def _wait_for(ws, events, mtype: str, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if any(m.get("type") == mtype for m in events):
            return True
        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=1.0)
        except asyncio.TimeoutError:
            continue
        m = json.loads(raw) if isinstance(raw, str) else {}
        events.append(m)
    return any(m.get("type") == mtype for m in events)


if __name__ == "__main__":
    asyncio.run(main())
