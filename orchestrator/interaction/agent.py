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
import os
import queue
import threading
import urllib.error
import urllib.request
from typing import Any, Optional

logger = logging.getLogger(__name__)

DEFAULT_AGENT_URL = "http://192.168.89.102:8081"

_QUEUE_MAX = 64
_POST_TIMEOUT = 5.0

#: 编排服务是否**也**把 Action 转发给 Agent（默认关）。
#:
#: ## 为什么默认关
#:
#: IC 判出 ANSWER / INSERT / YIELD / END 时，**IC 自己**就会经
#: ``AgentSink`` 把这些事件 POST 给 Agent（见 interactioncore 的
#: ``runtime.py._notify_sinks``）。编排服务这边**也**做一遍是**重复投递** ——
#: Agent 会收到两份 ``on_answer``（一份来自 IC、一份来自这里），
#: 行为变成**取决于 Agent 内部如何处置重复**（去重？覆盖？都处理？），
#: 而不是我们能保证的。实测确认过两份都发出去了。
#:
#: ## 那为什么还留着这个开关，而不是直接删掉
#:
#: 因为它曾经是**唯一的通路**：IC 的 HTTP 回调用的是和 ``/v1/speak`` /
#: ``/v1/stop`` 同一套机制，而那条通路实测会因为**证书 / 网络 / 找不到
#: session(404) 静默失效**（失败只记在 IC 侧日志，编排侧完全无感）。
#: 那时编排侧的这份转发就是**兜底**。
#:
#: 直接删掉 ⇒ 万一 IC 通路再出问题，现象是「**完全没有回复**」且日志干净，
#: 比"重复投递"难查得多。所以留成开关：
#:     ORCH_AGENT_RELAY=1    打开兜底（IC 通路有问题时用）
#:     未设置 / 0            只靠 IC（默认，正常部署就该这样）
_AGENT_RELAY = os.environ.get("ORCH_AGENT_RELAY", "0") == "1"


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
    #  把「本会话的 IC 地址」同步给 Agent
    # ------------------------------------------------------------------ #

    def set_ic_target(self, ic_target: str) -> bool:
        """告诉 Agent：本会话的 IC 在 ``ic_target``。

        Agent 拿到 IC 的 Action 后要靠这个地址回连，**不设就会派到它自己
        默认的那个 IC**（106）—— 于是「用别人的 IC 服务跑 replay」时，
        replay 收不到自己的 Action（现象：IC 决策一直不对/判题全错）。

        ⚠️⚠️ **这是 Agent 侧的``进程级全局``状态**（端点收的是单数
        ``interaction_core_target``，没有 session 维度）。所以：
            · 多人/多会话**并发**时，后设的会覆盖先设的，**最后关的赢**
            · 这一版只做「同步 + 冲突告警」，**不**试图用锁去串行化 ——
              本进程内的锁挡不住别人另起一个 orchestrator 实例
        真正的解法在 Agent 侧（按 session 存 target，或让请求带上 target）。
        在它实现之前，调用方**必须**能从日志里看出"地址被谁改了"。

        返回是否设置成功。**失败只告警，不抛** —— Agent 连不上不该让会话崩
        （与 `_post` 的失败处理一致：Agent 是可降级的旁路）。
        """
        url = f"{self.target}/interaction/set_target"
        body = json.dumps({"interaction_core_target": ic_target}).encode()
        try:
            req = urllib.request.Request(
                url, data=body,
                headers={"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(req, timeout=_POST_TIMEOUT) as r:
                r.read()          # 读掉响应体，别留连接
            logger.info("Agent(%s) 的 IC 目标已设为 %s"
                        "（此后 IC 的 Action 会派到这里回连）",
                        self.target, ic_target)
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("设置 Agent(%s) 的 IC 目标失败（%s）：%s —— "
                           "IC 的 Action 可能被派到 Agent 默认的 IC 上",
                           self.target, url, exc)
            return False

    # ------------------------------------------------------------------ #

    # ⚠️ 下面四个 ``on_*`` 默认**不发**（`_AGENT_RELAY=0`）—— IC 自己会发。
    #    同一个 Action 在这里再发一次就是**重复投递**，见 `_AGENT_RELAY` 的说明。
    #    只有把 `ORCH_AGENT_RELAY=1` 打开时才真的投递（IC 通路出问题时的兜底）。

    def on_answer(self, transcript: str, identity_id: Optional[str] = None,
                  display_name: Optional[str] = None) -> None:
        if not _AGENT_RELAY:
            return
        self._post("on_answer", {
            "transcript": transcript or "",
            "identity_id": identity_id,
            "display_name": display_name,
        })

    def on_insert(self) -> None:
        if not _AGENT_RELAY:
            return
        self._post("on_insert", {})

    def on_yield(self) -> None:
        if not _AGENT_RELAY:
            return
        self._post("on_yield", {})

    def on_end(self) -> None:
        # ⚠️ `on_session_end`（会话真正收尾）也调这一个 —— 那条路径**不是**
        #    重复（IC 没有"编排服务会话结束"这个概念），所以它**不受**开关
        #    影响：见 `engine_on_session_end()`。
        if not _AGENT_RELAY:
            return
        self._post("on_end", {})

    def engine_on_session_end(self) -> None:
        """**编排服务会话收尾**专用 —— 不受 `_AGENT_RELAY` 开关影响。

        `downstream.on_session_end` 里调的 `on_end` 与 IC 判 END 是**两件事**：
          · IC 判 END  = "IC 认为这轮交互结束"（IC 自己会通知 Agent）
          · 这里       = "编排服务这一路会话真的要关了"（客户端断开 / 收尾）
        后者只有编排服务知道，**不重复**，所以无论如何都要发。
        """
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
