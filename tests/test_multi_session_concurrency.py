"""
MiniCPM-o 多路并发测试 — 通过 Gateway 协议验证单卡多 session 推理

测试场景:
  - N 路单工并发
  - 连续创建/销毁 session，验证线程标志不会互相干扰

用法:
  cd /data/megastore/Projects/DuJing/code/MiniCPM-o-Demo
  /home/dujing/miniconda3/envs/py310/bin/python tests/test_multi_session_concurrency.py \
      --gateway wss://192.168.89.106:8006/v1/realtime \
      --simplex 4 --timeout 120
"""

import asyncio
import json
import time
import argparse
import ssl
import sys
from dataclasses import dataclass
from typing import Optional

import websockets

GATEWAY = "wss://192.168.89.106:8006/v1/realtime"
MAX_WS = 128 * 1024 * 1024

# Self-signed cert from docker deployment
SSL_CTX = ssl.create_default_context()
SSL_CTX.check_hostname = False
SSL_CTX.verify_mode = ssl.CERT_NONE


@dataclass
class SessionResult:
    session_id: str
    index: int
    ok: bool = False
    error: str = ""
    duration_ms: float = 0.0
    text: str = ""


async def _handshake(ws, init_payload) -> str:
    """Gateway 握手: wait queue_done → session.init → session.created"""
    while True:
        msg = json.loads(await ws.recv())
        t = msg.get("type")
        if t in ("session.queue_done", "queue_done"):
            await ws.send(json.dumps({"type": "session.init", "payload": init_payload}))
        elif t == "session.created":
            return msg.get("session_id")
        elif t == "error":
            raise RuntimeError(f"handshake error: {msg}")


async def run_simplex_session(idx: int, gateway_url: str, timeout_s: float = 60) -> SessionResult:
    """一路单工会话：发一条文本消息，等 response.done"""
    try:
        async with websockets.connect(gateway_url, max_size=MAX_WS, open_timeout=10, ssl=SSL_CTX) as ws:
            sid = await _handshake(ws, {"mode": "turn_based"})
            t0 = time.perf_counter()
            await ws.send(json.dumps({
                "type": "input.append",
                "input": {
                    "messages": [{"role": "user", "content": f"请用一句话回答：{idx}加{idx}等于几？"}],
                    "streaming": False,
                    "tts": {"enabled": False},
                    "use_tts_template": False,
                },
            }, ensure_ascii=False))
            text = ""
            while True:
                msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=timeout_s))
                t = msg.get("type")
                if t == "response.output.delta" and msg.get("kind") == "text":
                    text += msg.get("text", "")
                elif t == "response.done":
                    await ws.send(json.dumps({"type": "session.close", "reason": "done"}))
                    duration = (time.perf_counter() - t0) * 1000
                    return SessionResult(sid, idx, ok=True, text=text.strip(), duration_ms=duration)
                elif t in ("session.closed", "error"):
                    return SessionResult(sid, idx, error=msg.get("reason", str(msg)),
                                         duration_ms=(time.perf_counter() - t0) * 1000)
    except Exception as e:
        return SessionResult("", idx, error=str(e))


async def main():
    parser = argparse.ArgumentParser(description="多路并发测试")
    parser.add_argument("--gateway", default=GATEWAY)
    parser.add_argument("--simplex", type=int, default=4, help="单工并发路数")
    parser.add_argument("--timeout", type=float, default=120, help="每路超时秒数")
    parser.add_argument("--rounds", type=int, default=2, help="测试轮数")
    args = parser.parse_args()

    for round_num in range(1, args.rounds + 1):
        print(f"\n{'='*60}")
        print(f"  第 {round_num} 轮：同时启动 {args.simplex} 路单工会话")
        print(f"{'='*60}")

        tasks = [run_simplex_session(i, args.gateway, args.timeout) for i in range(args.simplex)]

        t_all = time.perf_counter()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        elapsed_all = (time.perf_counter() - t_all) * 1000

        ok_count = 0
        print(f"\n{'#':>3} {'status':<10} {'duration':>10}  {'text'}")
        print("-" * 70)
        for i, r in enumerate(results):
            if isinstance(r, Exception):
                print(f"{i:>3}  {'FAIL':<10} {'--':>10}  {r}")
                continue
            status = "PASS" if r.ok else "FAIL"
            if r.ok:
                ok_count += 1
            err_info = f"  [{r.error[:60]}]" if r.error else ""
            print(f"{i:>3}  {status:<10} {r.duration_ms:>8.0f}ms  {r.text[:80]}{err_info}")

        total = args.simplex
        print(f"\n结果: {ok_count}/{total} 通过, 总计耗时 {elapsed_all:.0f}ms")

        if ok_count < total:
            print(f"\n❌ 第 {round_num} 轮测试失败！")
            sys.exit(1)
        else:
            print(f"✅ 第 {round_num} 轮全部通过")

    print(f"\n{'='*60}")
    print(f"  全部 {args.rounds} 轮测试通过！")
    print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())
