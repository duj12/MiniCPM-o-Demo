"""Agent Platform 的 HTTP 客户端。

Agent Platform（魔珐科技）已实现四个端点，对应 InteractionCore 的
``AgentSink`` 四类 Action：

    POST {target}/interaction/on_answer   {"transcript","identity_id","display_name"}
    POST {target}/interaction/on_insert   {}
    POST {target}/interaction/on_yield    {}
    POST {target}/interaction/on_end      {}

**四个端点都是单向的**（响应体被丢弃）—— 回复文本**不从这里回来**。
Agent 生成后调 orchestrator 的 ``POST /v1/speak``，走既有 TTS 链路。

⚠️ **绝不能在事件回调里同步调用** —— ``run_downstream`` 是串行 await 循环，
HTTP 往返会把它占住（与 :mod:`.client` 同一个理由）。所以同样走
**后台队列 + 单线程消费**。

用 ``urllib`` 而不是 requests/httpx —— 不引新依赖，且这里是纯 POST。
"""
from __future__ import annotations

import json
import logging
import queue
import threading
import urllib.error
import urllib.request
from typing import Any, Optional

logger = logging.getLogger(__name__)

DEFAULT_AGENT_URL = "http://192.168.89.102:8081"

_QUEUE_MAX = 64
_POST_TIMEOUT = 5.0


class AgentClient:
    """一路会话一个实例。所有投递非阻塞，失败只告警。"""

    def __init__(self, target: str = DEFAULT_AGENT_URL) -> None:
        self.target = (target or DEFAULT_AGENT_URL).rstrip("/")
        self._q: "queue.Queue" = queue.Queue(maxsize=_QUEUE_MAX)
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self.sent = 0
        self.failed = 0
        self.dropped = 0

    # ------------------------------------------------------------------ #

    def start(self) -> None:
        """起后台投递线程（**不做探活** —— Agent 慢/挂不该阻塞建会话）。"""
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="agent-post", daemon=True)
        self._thread.start()
        logger.info("Agent 投递已启动: %s", self.target)

    # ------------------------------------------------------------------ #

    def on_answer(self, transcript: str, identity_id: Optional[str] = None,
                  display_name: Optional[str] = None) -> None:
        self._post("on_answer", {
            "transcript": transcript or "",
            "identity_id": identity_id,
            "display_name": display_name,
        })

    def on_insert(self) -> None:
        self._post("on_insert", {})

    def on_yield(self) -> None:
        self._post("on_yield", {})

    def on_end(self) -> None:
        self._post("on_end", {})

    # ------------------------------------------------------------------ #

    def _post(self, event: str, payload: dict) -> None:
        """入队（立即返回）。"""
        try:
            self._q.put_nowait((event, payload))
        except queue.Full:
            self.dropped += 1
            logger.warning("Agent 投递队列满，丢弃 %s（累计 %d）",
                           event, self.dropped)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                item = self._q.get(timeout=0.2)
            except queue.Empty:
                continue
            if item is None:
                break
            event, payload = item
            url = f"{self.target}/interaction/{event}"
            try:
                req = urllib.request.Request(
                    url, data=json.dumps(payload, ensure_ascii=False).encode(),
                    headers={"Content-Type": "application/json"}, method="POST")
                with urllib.request.urlopen(req, timeout=_POST_TIMEOUT):
                    pass          # 响应体无意义（单向端点）
                self.sent += 1
                logger.debug("Agent %s -> ok", event)
            except Exception as exc:  # noqa: BLE001
                self.failed += 1
                # 失败只告警 —— Agent 挂掉不该让会话崩
                logger.warning("Agent %s 投递失败（%s）: %s", event, url, exc)

    # ------------------------------------------------------------------ #

    def close(self) -> None:
        self._stop.set()
        try:
            self._q.put_nowait(None)
        except queue.Full:
            pass
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        logger.info("Agent 投递已停止: 成功 %d 失败 %d 丢弃 %d",
                    self.sent, self.failed, self.dropped)
