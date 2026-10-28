"""把 VLM 描述塞进 IC 派给 Agent 的 ``on_answer`` —— **编排侧私有包装**。

## 为什么要包装，而不是改 interactioncore

``interactioncore`` 的 ``AgentSink`` 是**多机共用**的库（105/106 跑同一份，
``run_ic_*.sh`` 起的两个常驻 gRPC 服务也 import 它）。它自己的注释就写明了
这条规矩（``runtime.py:227-230``）：

    为什么默认关：这**改变了对 Agent 的线上协议**……而 IC 是**多机共用**的
    （105/106 跑同一份代码，各自连不同的 Agent），一个进程的改动会同时
    影响两边。所以做成显式开关：确认目标 Agent 能接受之后，在**那一台**
    的启动脚本里打开。

而 sink 的**实例是编排侧构造**的（``inprocess.py`` 里唯一一处 ``AgentSink(``），
``Engine`` 只按鸭子接口调 ``agent_sink.on_answer(...)`` ⇒ 在这里包一层子类，
就能只改**我们自己发出去的那一个 JSON 键**，IC 一行不动。三个后果：

  · interactioncore **零改动**（不新增 RPC / HTTP 端点 / proto，也不用重启
    那两个常驻 IC 服务，106 下次重启也不受影响）
  · `gRPC 回退模式结构性拿不到 VLM` —— sink 在**另一个进程**里，包不到。
    回退面本来就不该有它，属已知限制
  · `legacy` 模式**根本不装这个包装**（不是包装里的一个分支）—— 回退路径
    连这段代码都不经过，payload 与今天逐字节一致

## 覆写点是 ``_send``，不是 ``_post``

IC 的调用链是 ``on_answer`` → ``_send``（同步入队）→ worker 线程 → ``_post``。
选 ``_send`` 是为了**取值时机**：它在 ``on_answer`` 里被**同步**调用，取到的
正是**推这一轮那一刻**缓存里的描述。若改在 ``_post`` 里取，取值发生在 worker
线程上，Agent 慢时队列积压会让描述漂到几秒之后 —— 与"推出去的就是当时那份
状态"的前提直接矛盾。

## 另一件只有"自己造 sink"才做得到的事：视觉背景轮

除正常话轮外，本子类还提供 ``emit_visual_background()``：把**当前缓存**里的
那份描述当成一轮「ASR 为空」的 `on_answer` 投出去 —— Agent 收到后走
`handle_visual_background`（只存不答），覆盖写 `scene.visual_background`。
IC **结构上产生不了**这一轮（它的 `on_answer` 只在 ANSWER 时发、
`transcript` 必非空），所以必须由编排侧合成。

**VLM 与 ASR 不同步（2026-10-28）**：每份描述一生成完就立刻这样推一次；
正常话轮（ASR 非空）的 ``content.VLM`` 因此留空 —— 视觉信息全部走
``scene.visual_background``，每轮随「会话状态」段注入 prompt。这样两边互不
等待：ASR 该发就发，VLM 算完就发。

⚠️ 这与 `interaction/agent.py` 里删掉的 `ORCH_AGENT_RELAY` **不是一回事**：
那条转发的是 IC **自己也会发**的事件（于是 Agent 收到两份）；这一轮 IC
**从不产生**，不存在重复。而且它走的是**同一个** sink 队列 ⇒ 与 IC 的真实
`on_answer` 端到端串行（`_post` 阻塞到 HTTP 返回才取下一件），顺序确定。
别把它当重复投递"修掉"。

## ⚠️ 代价：耦合一个私有方法

``_send(name, payload)`` 是 IC 的**私有**方法。它若被改名/改签名，这个覆写在
Python 里会**静默失效**（基类的 ``on_answer`` 照样工作，只是不带 ``content``）——
正是本仓库反复踩的那类静默故障（"改了、没生效、不报错"）。

两道防线：

  1. ``tests/test_omni_describe_stage.py`` 里的**契约漂移测试**：真 ``AgentSink``
     打到本机捕获端口，断言旧形态键集与投递次数 —— 覆写点一旦不在投递路径上，
     这个测试会失败
  2. ``[VLM 注入]`` 日志：没有它，`VLM=0字` 到底"是没描述"还是"包装没生效"
     就分不清（105 上就是靠这条日志判）
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

#: 允许的 payload 形态。
#:
#:   ``legacy``  ``{…, "transcript": "<ASR 文本>"}`` —— **与今天逐字节一致**
#:   ``dual``    额外带 ``content: {ASR, VLM}``，``transcript`` 原样保留
#:               —— 过渡期形态：老 Agent 忽略多出来的键即可
#:   ``dict``    ``transcript`` 本身换成 ``{ASR, VLM}`` —— 目标态
MODES = ("legacy", "dual", "dict")


def normalize_mode(mode: Optional[str]) -> str:
    """归一模式名；**未知值一律退到安全侧 `legacy`**。

    ⚠️ 别把未知值静默当成 ``dual``：那会在"配置写错"（比如 ``ORCH_AGENT_
    TRANSCRIPT_MODE=Dict `` 大小写/拼写错）时**改变对 Agent 的线上协议**，
    而症状出现在 Agent 侧、编排这边看不见 —— 与那两个 IC 模式半切换
    是同一类问题。宁可退到不改 payload，并在日志里说出来。
    """
    m = (mode or "").strip().lower()
    if m in MODES:
        return m
    if m:
        logger.warning("未知的 transcript 模式 %r —— 归一为 legacy"
                       "（不改 payload；可选 %s）", mode, "/".join(MODES))
    return "legacy"


def make_vlm_agent_sink(*, target: str, vlm_getter: Callable[[], str],
                        mode: str = "dual", **kwargs: Any) -> Any:
    """造一个"on_answer 带 VLM"的 ``AgentSink`` 子类实例。

    ``interaction`` 是**惰性 import** —— 包没装时只该让这一路会话降级，
    不该让 orchestrator 起不来（与 ``inprocess.connect()`` 的
    ``ImportError`` 兜底同款）。

    ``vlm_getter`` 是**回调而不是值**：描述每两秒刷新一次，值必须**取用时**
    才读（读的是调用方那一刻的缓存）。
    """
    from interaction.runtime import AgentSink

    mode = normalize_mode(mode)

    class VlmAgentSink(AgentSink):
        """只多一个 ``content`` 键，其余全交给基类。

        **不要**在这里重新拼 payload（``identity_id`` / ``display_name`` /
        信封字段……）：那样就复制了一份契约，IC 以后改了字段这里会静默落后。
        覆写 ``_send`` 是在基类**已经拼好**的 dict 上追加 —— 天然跟着基类走。
        """

        def _send(self, name: str, payload: dict) -> None:
            if name == "on_answer":
                asr = str(payload.get("transcript", "") or "")
                # VLM 与 ASR **不同步**（2026-10-28）：视觉状态一生成完就由
                # `emit_visual_background` 单独推（那种轮次 ASR 为空、VLM 取
                # 缓存）；正常话轮（ASR 非空）**只发 ASR**、VLM 留空。
                # 否则 Agent 手上会同时拿到「刚推的那份」和「本句缓存那份」，
                # 两份时刻不同，反而说不清哪份才是当前画面。
                vlm = str(vlm_getter() or "") if not asr.strip() else ""
                content = {"ASR": asr, "VLM": vlm}
                payload = {**payload, "content": content}
                if mode == "dict":
                    payload["transcript"] = content
                # ⚠️ 这条日志是 105 上判"VLM 到底有没有搭上车"的**唯一**依据
                #    （编排侧其余日志看不到 Agent 收到的 payload）。
                logger.info(
                    "[VLM 注入] mode=%s ASR=%d字 VLM=%d字 VLM=%r",
                    mode, len(asr), len(vlm), vlm[:400])
            super()._send(name, payload)

        def emit_visual_background(self) -> bool:
            """把**当前缓存里**的那份描述当「视觉轮」立刻推给 Agent。

            投出去的是一轮 ``ASR 为空`` 的 ``on_answer``：Agent 收到
            ``content`` 且 ``ASR``/``transcript`` 都空 ⇒ 走
            ``handle_visual_background``（**只存不答**，覆盖写会话状态
            ``scene.visual_background``，该键随 system prompt 的「会话状态」段
            进入每一轮 LLM 调用）。

            为什么需要这一发：描述**不与 ASR 同步**（2026-10-28 用户口径）——
            一生成完就推，而不是等下一次 ``on_answer`` 捎带。IC 自己产生不了
            这一轮（它的 ``on_answer`` 只在 ANSWER 时发，``transcript`` 必
            非空），所以由编排侧合成 —— **不是** IC 事件的转发。

            ⚠️ 前置条件：调用方必须在**刚把全量描述写进缓存之后同步调用**
            （见 ``downstream._on_vlm_description``）。本方法**不收文本参数**
            —— 它借基类的 ``on_answer(transcript="")`` 让上面的 ``_send``
            覆写按 ``asr == ""`` 自动补出 ``content={"ASR": "", "VLM": …}``，
            于是取到的正是那一份刚写入的描述。不收参数 ⇒ 没有"override 槽位"
            这类可变状态，也就不存在跨会话/跨线程泄漏的可能。

            ⚠️ 返回值只表示「**已交给 sink 的发送队列**」，**不是**「已送达」：
            基类 ``_send`` 吞掉了 ``_QueuedWorker.submit`` 的返回值
            （``runtime.py`` 的 ``AgentSink._send``），拿不到送达回执。
            所以调用方**不要**拿它当"Agent 已收到"的证据。
            """
            if mode == "dict":
                # dict 模式把 `transcript` 整个换成 `{ASR,VLM}`，而 Agent 的
                # `OnAnswerRequest.transcript` 是 `str` ⇒ 背景轮必被 **422** 挡掉。
                # 明确拒绝并说出来，别让它静默消失（dict 是目标态、当前未启用）。
                logger.warning("transcript 模式=dict：不发视觉背景轮"
                               "（Agent 的 transcript 字段是 str，背景轮会被拒）")
                return False
            logger.info("[视觉轮] 已交出 %d 字 → on_answer(ASR 空，只存不答)",
                        len(str(vlm_getter() or "")))
            self.on_answer(identity_id=None, display_name=None, transcript="")
            return True

    return VlmAgentSink(target, **kwargs)
