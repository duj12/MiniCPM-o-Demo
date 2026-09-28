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

#: 会话收尾时是否通知 Agent（默认**关**）。
#:
#: ⚠️ 默认关是因为这个通知会**跨会话误杀** —— 详见
#: `AgentClient.engine_on_session_end` 的说明（实测：旧会话迟到的 on_end
#: 撞上新会话，新会话播报中被打断）。
#:
#: 等 Agent 侧支持按 `session_id` 区分后，设 `ORCH_AGENT_END_NOTIFY=1` 打开
#: —— payload 里已经带了 `session_id`。
_END_NOTIFY = os.environ.get("ORCH_AGENT_END_NOTIFY", "0") == "1"


class AgentClient:
    """一路会话一个实例。所有投递非阻塞，失败只告警。"""

    def __init__(self, target: str = DEFAULT_AGENT_URL,
                 session_id: str = "") -> None:
        self.target = (target or DEFAULT_AGENT_URL).rstrip("/")
        #: 本会话 id。**必须随每个请求带给 Agent** —— 见 `engine_on_session_end`
        #: 的说明（不带它 = Agent 分不清这是哪一路会话的结束，会**跨会话误杀**）。
        self.session_id = session_id or ""
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

    # ⚠️ **这里只有 Agent 的"寻址/生命周期"职责，没有事件转发。**
    #
    # 四个 Action 事件（on_answer / on_insert / on_yield / on_end）
    # **一律由 IC 自己发**（interactioncore 的 `AgentSink`，走
    # `runtime.py._notify_sinks`），而且它带的字段更全（从 IC 的 state 读）。
    # 编排侧曾也有一份（`ORCH_AGENT_RELAY` 开关），那和 IC 是**同一件事做两遍**
    # —— Agent 会收到两份 `on_answer`，行为变成"取决于 Agent 如何处置重复"。
    # 已整个删除：**编排侧只做编排**。

    def engine_on_session_end(self) -> None:
        """**编排服务会话收尾**通知 Agent。

        与 IC 判 END 是两件事（IC 不知道"编排服务这一路要关了"）：
          · IC 判 END  = "IC 认为这轮交互结束"（IC 自己会通知 Agent）
          · 这里       = "编排服务这一路会话真的要关了"（客户端断开 / 收尾）

        ## ⚠️⚠️ 默认**不发**（`ORCH_AGENT_END_NOTIFY=1` 才发）

        原因：**不带会话标识的 `on_end` 会跨会话误杀**。实测证据（106）：

            17:19:14  旧会话 3ecc03385b5c: IC → END(24)（人走了，开始收尾）
            17:19:21  新会话 5d42831a19e5 启动  ← 用户开了新会话
            17:19:27  新会话正在播报「好嘞，帮你去知识库里翻翻魔珐科技介绍～」
            17:19:31  旧会话收尾超时（20s），强制关闭   ← 收尾慢，on_end 迟到
            17:19:32  旧会话的 on_end 发出（成功 1）
            17:19:32  **新会话立刻"身份识别收尾落地"** ← 同一秒，被误杀

        两个原因叠加：
          ① `on_end` 原先 payload 是**空的 `{}`**、请求头只有 Content-Type
             —— Agent **无法区分**这是哪一路会话的结束（对比 IC 的
             `ExpressionSink` 会注入 `X-Session-Id`）；
          ② **收尾要等 20s 超时**才结束，`on_end` 因此**迟到**，正好撞上
             用户刚开的新会话。

        每场会话都发生（24 场 = 24 次 `成功 1`），很可能是「会话被切得很碎」
        的原因之一。

        ## 恢复方式

        等 Agent 侧支持按会话区分后：把 `ORCH_AGENT_END_NOTIFY=1` 打开即可
        —— payload 里已经带了 `session_id`，届时 Agent 直接可用。
        """
        if not _END_NOTIFY:
            logger.debug("跳过 on_end 通知（%s）—— 未带会话标识会误杀其他会话，"
                         "需 Agent 支持后设 ORCH_AGENT_END_NOTIFY=1 打开",
                         self.session_id or "?")
            return
        self._post("on_end", {"session_id": self.session_id})

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
