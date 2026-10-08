#!/usr/bin/env python3
"""探针：Agent 到底会不会按 `callback_ic` 把 `apply_agent` 写回来？

`ORCH_IC_MODE=inprocess` 有一个**硬前置**，也是最容易静默失败的一环：

    Agent 必须读 `on_*` payload 里的 `callback_ic`，并把 apply_agent 写到
    `POST {callback_ic}/apply_agent`。

做不到（还按自己配置的地址写，或者压根不调 apply_agent）的后果：
`agent.status` / `session_end_pending` 永远写不进来 ⇒ policy 的
**SOP 39 与 PENDING_ANNOUNCE 两条分支失效，且不报错**。

为什么不能靠看接口判断：新旧 Agent 的 `/openapi.json` **逐字一样**
（`/interaction/on_*` 四个端点都在，请求体 schema 也都有 `callback_ic` 字段），
路由表和 schema 都区分不出「读了字段」和「只是声明了字段」。只能真打一次。

做法：本机起一个**假编排端点**，把它的地址当 `callback_ic` 发给 Agent 的四个
`/interaction/on_*`，然后看它会不会 `POST {callback_ic}/apply_agent` 回来。

    python -m orchestrator.tests.probe_agent_callback \
        --agent http://192.168.89.102:8081 \
        --callback-host 192.168.89.106 --listen-port 18100

⚠️ 只发**合成事件**，不碰任何在跑的会话。但 Agent 侧可能因此留下少量会话记录。
⚠️ `--callback-host` 必须是 **Agent 能回连的地址**（从 102 看得到的 106 地址），
   写 127.0.0.1 的话那是 Agent 自己。
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

#: 收到的请求（假编排端点这一侧）
HITS: list[dict] = []


class _Handler(BaseHTTPRequestHandler):
    """假编排端点：收什么都记下来，一律回 200 {"ok":true}。"""

    protocol_version = "HTTP/1.1"          # 支持 keep-alive，需自带 Content-Length

    def _record(self, body: str) -> None:
        HITS.append({
            "t": time.strftime("%H:%M:%S"),
            "method": self.command,
            "path": self.path,
            "session_header": self.headers.get("X-Session-Id"),
            "body": body[:800],
        })

    def _reply(self, obj: dict) -> None:
        payload = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_POST(self) -> None:  # noqa: N802
        n = int(self.headers.get("Content-Length") or 0)
        self._record(self.rfile.read(n).decode("utf-8", "replace"))
        self._reply({"ok": True, "session_id": "probe"})

    def do_GET(self) -> None:  # noqa: N802
        self._record("")
        self._reply({"ok": True})

    def log_message(self, *a) -> None:     # 静音 —— 自己打印
        pass


def _post(url: str, body: dict, timeout: float = 15.0) -> str:
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return f"HTTP {r.status}  {r.read().decode('utf-8', 'replace')[:300]}"
    except Exception as exc:  # noqa: BLE001
        detail = ""
        try:
            detail = exc.read().decode("utf-8", "replace")[:300]  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            pass
        return f"{type(exc).__name__}: {exc}  {detail}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default="http://192.168.89.102:8081")
    ap.add_argument("--callback-host", default="192.168.89.106",
                    help="Agent 回连本机用的地址（不能是 127.0.0.1）")
    ap.add_argument("--listen-port", type=int, default=18100)
    ap.add_argument("--wait", type=float, default=6.0,
                    help="每个事件后等多久看回写")
    ap.add_argument("--tls-cert", default="",
                    help="用 HTTPS 起假端点（传证书路径）—— 生产上 callback_ic 是 "
                         "**自签 https**，必须按真实形态验一次：Agent 若校验证书，"
                         "回写会失败，而失败在 Agent 侧是异步的，看不出来")
    ap.add_argument("--tls-key", default="")
    args = ap.parse_args()

    agent = args.agent.rstrip("/")
    scheme = "https" if args.tls_cert else "http"
    callback = f"{scheme}://{args.callback_host}:{args.listen_port}/v1/ic"

    srv = ThreadingHTTPServer(("0.0.0.0", args.listen_port), _Handler)
    if args.tls_cert:
        import ssl
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(args.tls_cert, args.tls_key or args.tls_cert)
        srv.socket = ctx.wrap_socket(srv.socket, server_side=True)
    threading.Thread(target=srv.serve_forever, daemon=True).start()

    sid = f"probe-{int(time.time())}"
    print(f"Agent        : {agent}")
    print(f"callback_ic  : {callback}   （假编排端点，监听 "
          f"0.0.0.0:{args.listen_port}）")
    print(f"session_id   : {sid}\n")

    # 顺序照真实会话：先 ANSWER（有人应答）再 INSERT（Agent 该动了）……
    # 四个都发，不必猜是哪个触发的 apply_agent —— 猜错就漏判。
    events = [
        ("on_answer", {"transcript": "你好，请问洗手间在哪？",
                       "identity_id": "probe-identity",
                       "display_name": "探针", "session_id": sid,
                       "callback_ic": callback}),
        ("on_insert", {"session_id": sid, "callback_ic": callback}),
        ("on_yield", {"session_id": sid, "callback_ic": callback}),
        ("on_end", {"session_id": sid, "callback_ic": callback}),
    ]

    for name, body in events:
        before = len(HITS)
        print(f"-- POST /interaction/{name} ...", flush=True)
        print(f"   ← {_post(f'{agent}/interaction/{name}', body)}", flush=True)
        deadline = time.time() + args.wait
        while time.time() < deadline and len(HITS) == before:
            time.sleep(0.2)
        if len(HITS) > before:
            print(f"   ✓ 收到 {len(HITS) - before} 个回写", flush=True)
        else:
            print(f"   ·  {args.wait:g}s 内没有回写", flush=True)

    srv.shutdown()

    print("\n== 假编排端点收到的全部请求 ==")
    if not HITS:
        print("  （一个都没有）")
    for h in HITS:
        print(f"  [{h['t']}] {h['method']} {h['path']}"
              f"  X-Session-Id={h['session_header']}")
        if h["body"]:
            print(f"        body: {h['body']}")

    applied = [h for h in HITS if "apply_agent" in h["path"]]
    print()
    if applied:
        print(f"✅ Agent 会按 callback_ic 回写 apply_agent（{len(applied)} 次）"
              f" —— 路径 {applied[0]['path']}")
        if not any(h["session_header"] for h in applied):
            print("⚠️ 但 apply_agent **没带 X-Session-Id** —— 多会话并存时编排"
                  "无法精确路由（`_find_session` 找不到会话 ⇒ 写不进去）")
        return 0
    print("❌ 没有收到 apply_agent 回写。可能的原因（按概率排）：")
    print("   1. Agent 还没改，仍写自己配置的 IC 地址（说明「已更新」不准确）")
    print("   2. 它读 callback_ic 但拼的路径不是 /apply_agent（看上面的 path）")
    print("   3. 它只在该会话先经 /interaction/on_answer 建立了上下文、"
          "且收到了别的事件之后才写（多试几个事件）")
    print("   ⇒ 没通之前**不要**切 106：SOP 39 / PENDING_ANNOUNCE 会静默失效")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
