"""Host orchestration of local IC, independent Agent gRPC, and TTS actions."""
from __future__ import annotations

import asyncio
import logging
import queue
import time
from typing import Any, Dict, List, Optional

from ..omni.describe import (
    is_no_change, looks_conversational, parse_state, render_state,
)

from ..downstream.interface import (
    AsrFinal, AsrPartial, AsrStateUpdate, Describe, DownstreamAction,
    DownstreamEvent, FaceIdentity, FaceLipState, FaceState, FaceWake,
    OmniDescription, PlaybackReceipt, Tick,
    Speak, Cancel, PauseTts, ResumeTts,
)
from .agent import AgentClient

logger = logging.getLogger(__name__)

HIGH = "HIGH"
MEDIUM = "MEDIUM"
LOW = "LOW"
NONE = "NONE"

#: IC 的档位字符串 ↔ 我们的（两边都是 HIGH/MEDIUM/LOW/NONE，直接透传）。
_VALID_CONF = {HIGH, MEDIUM, LOW, NONE}


def _conf(v: Any, default: str = NONE) -> str:
    """归一档位字符串；非法值退回 ``NONE``。

    ⚠️ 不能直接把任意字符串塞给 IC —— 它的 codec 遇到不认识的值会抛
    ``ValueError``（``grpc_codec.conf_field``），把整个写操作打掉。
    """
    s = str(v or "").strip().upper()
    return s if s in _VALID_CONF else default


#: ASR 感知字段 → IC `apply_asr` 的**同名**关键字（P05 新增）。
#:
#: 值直接取自 `AsrState.to_dict()`。见 `_asr_perception_kwargs` 的映射说明。
_ASR_PERCEPTION_KEYS: tuple = (
    "language", "language_prob", "language_switch",
    "emotion", "emotion_prob", "speech_event", "snr_db",
)

#: VLM 八键在 `parse_state()` 里是**大写**（那是提示词里的字段名），
#: 而 IC 的 `apply_vision` 参数名是**小写** —— 两边差的就是这个 `.lower()`。
_VISION_KEYS: tuple = (
    "ATTENTION", "ACTION", "EMOTION", "FEATURE",
    "BEHAVIOR", "RELATION", "ENV", "POSITION",
)

#: `supports("apply_asr", …)` 的探测键 —— 认 `language` 就说明是 P05 及以后
#: （P05 之前 `apply_asr` 只有 transcript / asr_confidence / turn_complete）。
_ASR_CAP_PROBE: tuple = ("language", "noise_burst_detected", "overlap_prob")


def _asr_perception_kwargs(st: Dict[str, Any]) -> Dict[str, Any]:
    """把 `AsrState.to_dict()` 的感知字段拆成 IC `apply_asr` 的 kwargs。

    proto 里是**拆平的**四个噪声/交叠键，而编排侧的快照是嵌套字典：

        noise.burst.detected → noise_burst_detected
        noise.burst.type     → noise_burst_type
        overlap.detected     → overlap_detected
        overlap.prob         → overlap_prob

    ⚠️ 值为 ``None`` 的**不放进去** —— proto 的语义是"省略 = 不改"，
    而这里的 ``None`` 是"这一段没测出来"（服务端没开感知 / 还没出结果），
    不是"清空 IC 上的值"。感知值本来就只在一帧里出现（描述的是**刚说完的
    那一段**），IC 保留上一段的值才对。
    """
    out: Dict[str, Any] = {}
    for k in _ASR_PERCEPTION_KEYS:
        v = st.get(k)
        if v is not None:
            out[k] = v
    noise = st.get("noise")
    if isinstance(noise, dict):
        burst = noise.get("burst")
        if isinstance(burst, dict):
            if burst.get("detected") is not None:
                out["noise_burst_detected"] = bool(burst["detected"])
            if burst.get("type"):
                out["noise_burst_type"] = str(burst["type"])
    overlap = st.get("overlap")
    if isinstance(overlap, dict):
        if overlap.get("detected") is not None:
            out["overlap_detected"] = bool(overlap["detected"])
        if overlap.get("prob") is not None:
            out["overlap_prob"] = float(overlap["prob"])
    return out


class InteractionDownstream:
    """把事件接到 InteractionCore，把其 Action 派给 Agent / 模板播报。"""

    #: 常见态：IC 每个 tick 都可能返回，**永不产生动作**（只做状态上报）。
    #: ⚠️ 它们照样要经由 ``on_action`` 下发 —— 否则 viz 的 IC 时间轴上
    #: 只剩下 6 种非常见态，Policy「一直在 HOLD/LISTEN/WAIT」看不出来。
    QUIET_TYPES = ("LISTEN", "WAIT", "HOLD")

    #: 状态**没变**时的定时上报间隔（秒）。
    #: 变化时立刻发；不变时每这么久补一条心跳，证明「决策还在跑」。
    #: 10s 是「既能看出还活着、又不制造噪声」的折中 —— 一场 6 分钟会话的
    #: 稳态心跳约 36 条，而状态切换点仍是零延迟。
    REPORT_INTERVAL_S = 10.0

    def __init__(self, ic_target: str, agent_url: str,
                 session_id: str = "",
                 on_action: Optional[Any] = None,
                 report_interval_s: Optional[float] = None,
                 ic_mode: str = "inprocess",
                 omni_describe: bool = False,
                 on_output: Optional[Any] = None,
                 agent_mcp_servers: Optional[list] = None) -> None:
        self._desc_enabled = bool(omni_describe)
        #: 最近一份**生成完**的 VLM 描述（见 `_on_vlm_description`）。
        #: 由 `agent_sink` 的 getter 在**推视觉轮的那一刻**同步读取（正常话轮
        #: 不再取它，见 `agent_sink._send`）—— 所以这里必须是一份"随时可用"
        #: 的缓存，取值路径上**不做任何生成**。
        self._vlm = ""
        #: 解析出来的**结构化快照**（`KEY -> 值`，见 `omni/describe.py`）。
        #: 每份都是 8 键全量；合并到这里再渲染成 `self._vlm`（合并是为了兜住
        #: 某一行被截断的情况）。
        #: ⚠️ 解析失败（模型没按格式答）时保持为空，`self._vlm` 退回原文 ——
        #:    绝不能因为"格式不对"就把一份有效描述丢掉。
        self._vlm_state: Dict[str, str] = {}
        #: 该描述**生成完成**的单调时刻（算年龄用）
        self._vlm_at = 0.0
        #: 该描述属于哪个阶段（full / delta）
        self._vlm_stage = ""
        #: `AsrFinal` 到达时描述**还没就绪**的次数 —— 这是主指标的失败计数
        #: （见 plan 的硬指标：主指标是"offline 文本到时已有非空描述"）
        self._vlm_miss = 0
        #: 上一次 dispatch 的 IC Action 类型。用于识别**切入** GREET
        #: （不能用 `fresh`：见 `_dispatch` 里的说明）
        self._prev_atype = ""
        #: 「视觉轮交不出去」的告警是否已经打过（**只为去重日志**，不是闩）。
        #: 每份描述都会尝试投递（见 `_emit_visual_background`），gRPC / legacy
        #: 这类交不出去的配置下别每轮刷一条 warning。
        self._bg_fail_logged = False
        #: 「视觉通知已由 IC 自己发、编排侧让路」这件事只记一次（每会话）
        self._ic_vision_logged = False
        #: 「IC 版本不支持某组新状态」的告警去重（见 `_warn_capability_once`）。
        #: 这只是**日志**去重 —— 每次仍会重新探测（`ic.supports` 自己有缓存），
        #: IC 升级并重连后自动恢复推送，不用重编排。
        self._cap_warned: set = set()
        if (ic_mode or "inprocess").strip().lower() != "inprocess":
            raise ValueError("InteractionCore is an in-process algorithm; ORCH_IC_MODE=grpc is retired")
        self.ic_mode = "inprocess"
        self.session_id = session_id
        self._events = queue.SimpleQueue()
        self._feedback = {}
        self._expression_revision = 0
        self._on_output = on_output
        self._output_queue = asyncio.Queue()
        self._output_worker = None
        self._control_tasks = set()
        self._agent_target = agent_url
        self._agent_mcp_servers = agent_mcp_servers
        from .inprocess import InProcessICClient
        self.ic = InProcessICClient(
            session_id,
            on_agent_event=lambda event: self._events.put(("action", event)),
            on_expression_event=lambda event: self._events.put(("expression", event)))
        self.agent = self._make_agent()
        #: `(action_type, sop, text)` 回调 —— 给 UI/日志用（可选）
        self._on_action = on_action
        if report_interval_s is not None:
            self.REPORT_INTERVAL_S = float(report_interval_s)
        #: 最近一次 ASR 的 people 信息 —— ANSWER 时要带给 Agent
        self._transcript = ""
        self._identity_id: Optional[str] = None
        self._display_name: Optional[str] = None
        #: 上一次 tick 的单调时刻（算 dt_ms）
        self._last_tick: Optional[float] = None
        #: 最近一次 Action（去重，避免同一个 ANSWER 重复派发）
        self._last_action_key: Optional[tuple] = None
        #: 上一次**下发**给客户端的 Action 键（(atype, sop)）。
        #: 用来判「状态变没变」—— 变了立刻发，没变就等下一个心跳时刻。
        self._last_reported_key: Optional[tuple] = None
        #: 「上一次该发的动作**没送出去**」—— 下一拍重试同一条。
        #: 专门救 `GREET` 这类一瞬就过去的动作（只出现一两拍，丢了就没了）。
        self._pending_report: bool = False
        #: 上一次下发的单调时刻（节流用）
        self._last_report_at: float = 0.0
        #: 给 UI/日志看的 Action 流
        self.actions: List[Dict[str, Any]] = []
        self.counts: Dict[str, int] = {}

    # ------------------------------------------------------------------ #
    #  生命周期
    # ------------------------------------------------------------------ #

    def describe(self) -> Dict[str, Any]:
        return {
            "name": "InteractionDownstream",
            "ic": self.ic.target,
            "agent": self.agent.target,
            "capabilities": [
                "asr.final", "asr.partial", "asr.turnsense",
                "face.state", "face.identity", "face.lip", "face.wake",
                "playback", "tick",
                # 只有真开了才声明 —— 这行是"本路会话会不会产 Describe"
                # 的唯一可观测点（105 靠它判影子实例起对了没）
                *(["omni.describe"] if self._desc_enabled else []),
            ],
            #: 描述模式 / Agent 载荷形态的生效值。**同一份代码 105/106 共用**，
            #: 日志里没有这一行就没法判"我这台到底开没开"（106 的默认值即
            #: 关 —— 见 config 的说明）。
            "omni_describe": self._desc_enabled,
        }

    async def on_session_start(self, ctx) -> List[DownstreamAction]:
        """会话启动。

        ⚠️ **连接已经在 ``build_session`` 里建好了** —— 连不上要在装配阶段
        就决定降级（回退 OmniLLM），不能等会话跑起来才发现没有回复来源。
        所以这里**只在没连上时兜底重试一次**，不重复建连接
        （早先无条件 `connect()`，日志里会看到连接打两遍）。
        """
        if not self.ic.available:
            if not self.ic.connect():
                logger.warning("[%s] InteractionCore 不可用（%s）—— 无法决策",
                               self.session_id, self.ic.error)
                return []
        # ⚠️⚠️ **必须在任何状态写入之前重置** —— 见 `client.reset_session`。
        #    IC 的 Engine 是全局一份，状态跨会话保留。不清就会：
        #      · `mode` 停在 LISTENING → **永远不迎宾**
        #      · `speech.user_speaking` 残留 → 新会话刚开口被判**抢话**
        #        → `ic_stop` → **播报刚出声就被打断**（实测：刷新页面后）
        #    ⚠️ 用 `reset_session`（**不通知** agent/TTS），不是 `end_session`
        #       —— 后者是"用户主动结束"用的。
        self.ic.reset_session()
        # ⚠️ VLM 缓存与 `_prev_atype` 都必须**随新会话清掉**：
        #   · 描述属于**上一个人/上一个场景**，新会话的首个 GREET 全量描述
        #     还没到之前，留着旧的比空着更糟（Agent 会照着一个不属于眼前
        #     这个人的描述说话）。空 VLM 至少是"没有信息"，不是"错误信息"。
        #   · `_prev_atype` 留着 `"GREET"` 会让新会话的首次 GREET 被判成
        #     "不是切入" ⇒ **一次全量描述都不发**（静默，只在日志里看得出）。
        if self._vlm:
            logger.info("[%s] 新会话开始 —— 清掉上一会话的 VLM 描述（%d 字）",
                        self.session_id, len(self._vlm))
        self._vlm = ""
        self._vlm_state = {}
        self._vlm_at = 0.0
        self._vlm_stage = ""
        self._prev_atype = ""
        # 告警去重标记随会话复位（不是闩，投递本身每份都做）
        self._bg_fail_logged = False
        self._ic_vision_logged = False
        self.agent.start()          # 幂等（_thread 非空直接返回）
        return []

    # ------------------------------------------------------------------ #
    #  宿主 Agent 通信与输出任务
    # ------------------------------------------------------------------ #

    def _make_agent(self):
        return AgentClient(
            self._agent_target, session_id=self.session_id,
            on_event=lambda event: self._events.put(("agent", event)),
            mcp_servers=self._agent_mcp_servers)

    async def _execute_output(self, action):
        try:
            guard = getattr(action, "guard", None)
            if guard is not None and not guard():
                return
            await self._on_output(action)
        except Exception:
            logger.exception("[%s] Host output failed", self.session_id)

    async def _run_output(self):
        while True:
            action = await self._output_queue.get()
            try:
                if action is None:
                    return
                await self._execute_output(action)
            finally:
                self._output_queue.task_done()

    async def _deliver(self, actions):
        if self._on_output is None:
            return actions
        remaining = []
        for action in actions:
            if isinstance(action, Speak):
                if self._output_worker is None:
                    self._output_worker = asyncio.create_task(self._run_output())
                self._output_queue.put_nowait(action)
            elif isinstance(action, (Cancel, PauseTts, ResumeTts)):
                # Pause waits for a browser receipt; a hard stop must still run
                # during that wait. The executor serializes pause/resume itself.
                task = asyncio.create_task(self._execute_output(action))
                self._control_tasks.add(task)
                task.add_done_callback(self._control_tasks.discard)
            else:
                remaining.append(action)
        return remaining

    async def on_session_end(self, reason: str) -> None:
        try:
            self.ic.end_session()
            await self._deliver(self._drain_events())
            if self._control_tasks:
                await asyncio.wait_for(asyncio.gather(*self._control_tasks), timeout=3)
            if self._output_worker is not None:
                await asyncio.wait_for(self._output_queue.join(), timeout=3)
        finally:
            for task in tuple(self._control_tasks):
                task.cancel()
            if self._output_worker is not None:
                self._output_worker.cancel()
                await asyncio.gather(self._output_worker, return_exceptions=True)
                self._output_worker = None
            await asyncio.gather(*self._control_tasks, return_exceptions=True)
            try:
                await asyncio.to_thread(self.agent.close)
            finally:
                await asyncio.to_thread(self.ic.close)

    def _drain_events(self) -> List[DownstreamAction]:
        actions = []
        while True:
            try:
                source, event = self._events.get_nowait()
            except queue.Empty:
                break
            if source == "action":
                if event.get("type") in ("answer", "yield", "end"):
                    self._feedback.clear()
                if self.agent._failed and event.get("type") == "answer":
                    # A new user turn explicitly establishes a fresh RPC/context.
                    self.agent.close(timeout=0)
                    self.agent = self._make_agent()
                    self.agent.start()
                if not self.agent.submit(event):
                    logger.warning("[%s] Agent action was not sent: %s", self.session_id, event.get("type"))
            elif source == "expression":
                guard = event.get("guard")
                if guard is not None and not guard():
                    continue
                kind = event.get("type")
                if kind in ("speak", "speak_stream"):
                    revision = self._expression_revision
                    valid = lambda revision=revision, guard=guard: (revision == self._expression_revision and (guard is None or guard()))
                    if event.get("interrupt"):
                        actions.append(Cancel(reason="ic_speak"))
                    actions.append(Speak(text=event.get("text", ""),
                                         stream_id=event.get("stream_id", ""),
                                         is_final=event.get("is_final", True),
                                         prompt=bool(event.get("prompt")), guard=valid))
                elif kind == "stop":
                    self._expression_revision += 1
                    actions.append(Cancel(reason=event.get("reason") or "ic_stop"))
                elif kind == "pause":
                    actions.append(PauseTts(reason=event.get("reason", ""),
                                            fade_ms=float(event.get("fade_ms") or 0)))
                elif kind == "resume":
                    actions.append(ResumeTts(reason=event.get("reason", "")))
            else:
                if event.get("connection_id", self.agent.connection_id) != self.agent.connection_id:
                    continue
                kind = event.get("type")
                if kind == "agent.state" and self.agent.is_current(event):
                    self.ic.apply_agent(status=event.get("status"),
                                        session_end_pending=event.get("session_end_pending"))
                elif kind == "agent.text" and self.agent.is_current(event, text=True):
                    if event["kind"] == "feedback":
                        key = (event["ci"], event["stream_id"])
                        self._feedback[key] = self._feedback.get(key, "") + event["text"]
                        if event["is_final"]:
                            text = self._feedback.pop(key)
                            if text:
                                actions.append(Speak(text=text, guard=lambda event=event, client=self.agent: client.is_current(event, text=True)))
                    else:
                        actions.append(Speak(text=event["text"],
                                             stream_id=f"agent:{event['ci']}:{event['stream_id']}",
                                             is_final=event["is_final"],
                                             guard=lambda event=event, client=self.agent: client.is_current(event, text=True)))
                elif kind in ("agent.error", "agent.disconnect"):
                    self._feedback.clear()
                    if kind == "agent.disconnect" or self.agent.is_current(event):
                        self.ic.apply_agent(status="IDLE", session_end_pending=False)
                        actions.append(Cancel(reason="agent_disconnect" if kind == "agent.disconnect" else "agent_error"))
                    logger.warning("[%s] %s: %s", self.session_id, kind, event.get("error"))
                elif kind == "agent.ack" and not event.get("accepted"):
                    if (event.get("act") == "ANSWER" and event.get("action_type") in (None, "answer")
                            and self.agent.is_current(event)):
                        self.ic.apply_agent(status="IDLE", session_end_pending=False)
                        actions.append(Cancel(reason="agent_rejected"))
                    logger.warning("[%s] Agent rejected action: %s", self.session_id, event.get("reason"))
                # done is only a generation terminal, never a playback observation.
        return actions

    def apply_agent(self, status: Optional[str] = None,
                    session_end_pending: Optional[bool] = None) -> None:
        """把宿主接收并过滤后的 Agent 状态投影到本会话的本地 IC。

        省略字段保持原值；播放事实由宿主真实播放回执单独回写。
        """
        self.ic.apply_agent(status=status, session_end_pending=session_end_pending)
        logger.info("[%s] Agent → IC: apply_agent(status=%s end_pending=%s)",
                    self.session_id, status, session_end_pending)

    @property
    def available(self) -> bool:
        """IC 是否可用 —— 不可用时调用方应降级到 OmniLLM 回复。"""
        return self.ic.available

    # ------------------------------------------------------------------ #
    #  事件 → IC
    # ------------------------------------------------------------------ #

    async def on_event(self, ev: DownstreamEvent) -> List[DownstreamAction]:
        # ---- 每次 Tick 向 IC 要一次决策 ----
        pending = self._drain_events()
        if isinstance(ev, Tick):
            pending.extend(await self._on_tick())
            pending.extend(self._drain_events())
            return await self._deliver(pending)

        # ---- 其余事件写进 IC（全部非阻塞投递）----
        self._feed(ev)

        # 记录身份/转写 —— ANSWER 时要带给 Agent
        if isinstance(ev, (AsrPartial, AsrFinal)) and ev.text:
            self._transcript = ev.text
        elif isinstance(ev, FaceIdentity):
            self._identity_id = ev.uid
            self._display_name = ev.name
        elif isinstance(ev, FaceState):
            st = ev.state or {}
            # state 里的身份是权威（它有 identity_confidence 档位）
            if st.get("identity_id"):
                self._identity_id = st["identity_id"]
            if st.get("display_name"):
                self._display_name = st["display_name"]

        pending.extend(self._drain_events())
        return await self._deliver(pending)

    def _feed(self, ev: DownstreamEvent) -> None:
        """把事件映射成 IC 的 ``apply_*``（**只投队列，立即返回**）。"""
        # ⚠️ OmniDescription **放在可用性判据之前** —— 它只写本对象的缓存，
        #    不调 `ic.apply`，与 IC 在不在无关。放在后面会被那条 `return`
        #    连带跳过，于是"IC 掉线 ⇒ VLM 缓存永远是空"，而症状（Agent 收到
        #    空 VLM）看起来像 prompt 的问题，很难往这儿想。
        if isinstance(ev, OmniDescription):
            self._on_vlm_description(ev)
            return

        if not self.ic.available:
            return

        if isinstance(ev, AsrStateUpdate):
            # tick 驱动的**周期性快照** —— 与内部状态**完全一致**（含归零）。
            # ⚠️ 这是 IC 唯一能得知"状态已经清了"的途径：`AsrPartial`/`AsrFinal`
            #    只在用户说话时才来，静音后没有任何消息，IC 会一直停在最后
            #    一条快照上（实测：一轮结束后 `说`/`抢` 仍显示 HIGH）。
            #
            # 转写**忠实透传**（归零后就是空串）—— 与 IC 的 `apply_asr` 口径一致。
            st = ev.state or {}
            self.ic.apply(
                "apply_vad",
                user_speaking_confidence=_conf(st.get("user_speaking_confidence")),
                barge_in_confidence=_conf(st.get("barge_in_confidence")),
            )
            asr_kwargs: Dict[str, Any] = dict(
                transcript=str(st.get("transcript") or ""),
                asr_confidence=_conf(st.get("asr_confidence")),
                turn_complete_confidence=_conf(st.get("turn_complete_confidence")),
            )
            asr_kwargs.update(self._asr_perception_kwargs(st))
            self.ic.apply("apply_asr", **asr_kwargs)

        elif isinstance(ev, AsrPartial):
            # 流式帧只更新"用户在说话"的证据；转写等 final 更准
            st = ev.state or {}
            self.ic.apply(
                "apply_vad",
                user_speaking_confidence=_conf(st.get("user_speaking_confidence")),
                barge_in_confidence=_conf(st.get("barge_in_confidence")),
            )

        elif isinstance(ev, AsrFinal):
            st = ev.state or {}
            text = (ev.text or "").strip()
            # ⚠️⚠️ **空文本的 final 绝不能覆盖已有转写。**
            #
            # ASR 在一段话结束后会再补一条**收尾帧**（`2pass-offline`、
            # `is_final=true`、**text 为空**）。早先这里无条件写
            # `transcript=ev.text or ""`，于是那条空帧把刚拿到的转写**擦成了
            # 空串**。后果很隐蔽：
            #   · IC 侧 transcript='' 而 turn_complete=HIGH
            #   · Policy 用 ('', conf) 当轮次键 → 判出一个**转写为空**的 ANSWER
            #   · Agent 拿到空转写、生成不出内容 → **没有任何播报**
            # 现象是「ASR 明明识别到了，却什么都不播」，而服务端日志里
            # `apply_asr` 的文本是对的 —— 只看服务端永远查不出来（实测踩过）。
            #
            # 所以空文本时**只更新档位，不动 transcript**。
            logger.debug("[%s] → IC: apply_asr text=%r asr=%s turn=%s%s",
                         self.session_id, text[:30],
                         _conf(st.get("asr_confidence")),
                         _conf(st.get("turn_complete_confidence")),
                         "" if text else "（空文本，保留上次转写）")
            kwargs: Dict[str, Any] = {}
            if text:
                # 有文本 = 真正的识别结果：转写与档位一起写
                kwargs["transcript"] = text
                kwargs["asr_confidence"] = _conf(st.get("asr_confidence"))
            else:
                # 空文本 = 收尾帧：**整条都不写**，让 IC 保留上一次的结果。
                # 连档位也不能写 —— 空帧的 asr_confidence 是 NONE，写进去会把
                # 刚拿到的 HIGH 降级，Policy 判 `asr_confidence == LOW` 时会
                # 误走「没听清，请您再说一遍」(SOP 01) 分支。
                # ⚠️ 例外：`turn_complete` 要写 —— 收尾帧正是「这轮说完了」
                #    的信号，IC 靠它推进话轮（fresh_turn → ANSWER）。
                kwargs["turn_complete_confidence"] = _conf(
                    st.get("turn_complete_confidence"))
            self.ic.apply("apply_asr", **kwargs)
            self.ic.apply(
                "apply_vad",
                user_speaking_confidence=_conf(st.get("user_speaking_confidence")),
                barge_in_confidence=_conf(st.get("barge_in_confidence")),
            )
            # ---- 主指标的观测点 ---------------------------------------- #
            # 这里就是「offline 文本到达」那一刻（`2pass-offline` = AsrFinal）。
            # 从这行往下到 Agent 收到 `on_answer`，**不许有任何等待** ——
            # VLM 只是 sink 里读一次缓存。所以这里只需要**度量**，
            # 不做任何"没就绪就等一会儿"的事（等就破坏了硬指标）。
            if text and self._desc_enabled:
                self._note_vlm_readiness()

        elif isinstance(ev, FaceState):
            st = ev.state or {}
            # 无人时 IC 要求显式写回 NONE + bbox 0
            self.ic.apply(
                "apply_face",
                face_present_confidence=_conf(st.get("face_present_confidence")),
                bbox_area_ratio=float(st.get("bbox_area_ratio") or 0.0),
            )
            # ⚠️ `track_id` 必须转成**字符串**：G1 的 track_id 是 int64，
            #    而 IC 的 proto 里是 ``optional string``。传 int 会让 gRPC
            #    序列化抛错 —— 而写队列**把异常吞成 debug 日志**，表现为
            #    IC 侧 track_id 恒为 None、dwell_ms 恒为 0，于是 policy 永远
            #    判 `passerby`（dwell < 2000ms）→ 永远 HOLD 21 → 永不 GREET
            #    → 永进不了 LISTENING → **ANSWER 永不触发**（实测踩过）。
            _tid = st.get("track_id")
            self.ic.apply(
                "apply_track",
                track_id=(str(_tid) if _tid is not None else None),  # None=清空
                dwell_ms=int(st.get("dwell_ms") or 0),
            )
            self.ic.apply(
                "apply_lip",
                lip_speaking_confidence=_conf(st.get("lip_speaking_confidence")),
            )
            # 身份：只有识别到人才写；否则清空（IC 要求失败时三件套都清）
            self.ic.apply(
                "apply_identity",
                identity_id=st.get("identity_id"),
                identity_confidence=_conf(st.get("identity_confidence")),
                display_name=st.get("display_name"),
            )

        elif isinstance(ev, FaceLipState):
            # 唇动事件是 10Hz 的聚合，也该反映到 IC（SOP 07 的判据之一）
            self.ic.apply(
                "apply_lip",
                lip_speaking_confidence=(
                    HIGH if ev.speaking and ev.lip_state == "SPEAKING"
                    else MEDIUM if ev.speaking
                    else LOW),
            )

        elif isinstance(ev, FaceWake):
            # 唤醒 = 主目标切换，清身份让 IC 重新学
            if ev.phase == "end":
                self.ic.apply("apply_identity", identity_id=None,
                              identity_confidence=NONE, display_name=None)

        elif isinstance(ev, PlaybackReceipt):
            # ⚠️ **这里不再写 playback_active**。
            #
            # `PlaybackReceipt` 整条流都流经 ``session.on_playback_receipt``，
            # 由那边**唯一一处**按 phase 决定是否写 IC（只有 playing/stopped
            # 写）。早先这里又按 ended/cancelled 写了一次 False，于是同一句
            # 播完 IC 会收到 2 次 False（session 一次 + 这里一次），打断时
            # 叠加 executor 那次最多 3 次。IC 是幂等覆盖所以没出故障，但纯属
            # 浪费 gRPC 往返，而且两条写入路径的语义会各自漂移。
            #
            # 表达层事实**只有一个权威写入点**（见 ``_notify_playback_active``）。
            pass

    # ------------------------------------------------------------------ #
    #  感知状态透传（IC P05：apply_asr 的语音子集 + 新的 apply_vision）
    # ------------------------------------------------------------------ #
    #
    # 两条新链路都**按 IC 版本自适应**：`ic.supports()` 探到不认识就整组不发，
    # 只告警一次。这样编排与 IC 的部署顺序**互不牵连** —— 共享树上还是旧 IC
    # 时，这里静默跳过；IC 换成 P05 后（重连会话即可）自动开始透传。

    def _asr_perception_kwargs(self, st: Dict[str, Any]) -> Dict[str, Any]:
        """`apply_asr` 的感知字段部分（不支持/没值的键一律不带）。"""
        kw = _asr_perception_kwargs(st)
        if not kw:
            return {}
        if not self.ic.supports("apply_asr", *_ASR_CAP_PROBE):
            self._warn_capability_once(
                "asr_perception",
                "apply_asr 的感知字段（语种/情绪/突发事件/SNR/交叠）")
            return {}
        return kw

    def _push_vision_to_ic(self) -> None:
        """把刚更新的 VLM 八键快照推给 IC 的 `apply_vision`（P05）。

        ⚠️ **`updated_at_ms` 不传** —— 由 IC 在自己收到时打时间戳
        （`state.apply_vision` 里有值变化就 `now_ms()`）。编排与 IC 的墙钟
        未必同源，传我们的反而是个"看起来更准、实际更不准"的数。

        ⚠️ 空快照（`parse_state` 没解析出任何键 —— 模型没按格式答）直接
        跳过：那种情况下 `_vlm` 已退回原文给 Agent，但 IC 的八个字段是
        结构化的，没有可写的东西，**不要**拿空串去覆盖上一份。
        """
        if not self._vlm_state:
            return
        if not self.ic.available:
            return
        if not self.ic.supports("apply_vision", *(k.lower() for k in _VISION_KEYS)):
            self._warn_capability_once(
                "vision", "apply_vision（VLM 八键：注意力/动作/情绪/…）")
            return
        kwargs = {k.lower(): v for k, v in self._vlm_state.items() if v}
        if not kwargs:
            return
        logger.debug("[%s] → IC: apply_vision %d 键（stage=%s）",
                     self.session_id, len(kwargs), self._vlm_stage or "-")
        self.ic.apply("apply_vision", **kwargs)

    def _warn_capability_once(self, key: str, what: str) -> None:
        """「IC 版本不支持 X」只告警一次（每会话每能力）。

        这不是闩：`_feed` / `_on_vlm_description` 每次都会再问 `supports`
        （结论按方法缓存），IC 升级重连后立刻恢复推送。
        """
        if key in self._cap_warned:
            return
        self._cap_warned.add(key)
        logger.warning(
            "[%s] 当前 InteractionCore 版本不支持 %s —— 这部分状态不会传给 IC"
            "（属于预期：共享树上的 IC 还是 P05 之前；升级 IC 并重开会话后"
            "自动生效，编排侧不用改）", self.session_id, what)

    # ------------------------------------------------------------------ #
    #  VLM 描述缓存（`Describe` 触发 → `OmniDescription` 落这里）
    # ------------------------------------------------------------------ #

    def _on_vlm_description(self, ev: OmniDescription) -> None:
        """收下一份**生成完**的描述，覆盖缓存。

        只覆盖**非空且非「无变化」**的 —— 空描述（模型什么都没说）与
        「无变化」（模型说没变化）都意味着"上一份描述仍然成立"，用它去覆盖
        只会把缓存清空，于是断句时发出去的是空串。缓存里**始终留着一份
        非空描述**正是主指标的立足点。
        """
        text = (ev.text or "").strip()
        if not text:
            logger.info("[%s] VLM 描述为空（stage=%s）—— 保持上一轮",
                        self.session_id, ev.stage)
            return
        if is_no_change(text):
            logger.info("[%s] VLM 无变化，保持上一轮（stage=%s，已缓存 %d 字）",
                        self.session_id, ev.stage, len(self._vlm))
            return
        if looks_conversational(text):
            # 启发式告警（不丢数据）：system prompt 没镇住时模型会开始聊天，
            # 而"聊天内容"混进 VLM 会让 Agent 以为有人在跟它说话。
            logger.warning("[%s] ⚠️ VLM 描述像是**对话**而非描述（prompt 没镇住？）"
                           "：%r", self.session_id, text[:60])
        self._vlm = text
        # ---- 结构化合并 ----
        #
        # 解析不到任何键 ⇒ **保持原文**（模型没按格式答时不丢数据，只是回到
        # 老行为：`content.VLM` 是自由文本）。这是本改动唯一的安全阀。
        parsed = parse_state(text)
        if parsed:
            if ev.stage == "full":
                # 起点全量是**完整替换**：旧的键不该残留（换人后 FEATURE 必须换）
                self._vlm_state = dict(parsed)
            else:
                # 每轮（`turn`）也是全量 8 键，合并 == 替换；用 `update` 是为了
                # 兜住"某一行被截断/没答"的情形 —— 缺的键保旧值，而不是丢掉。
                self._vlm_state.update(parsed)
            self._vlm = render_state(self._vlm_state)
        else:
            # 解析失败 ⇒ 快照与原文脱节，**清掉快照**，免得下一份增量合并到
            # 一个过时的底座上（旧键会以"没变化"的名义永远留着）。
            self._vlm_state = {}
        self._vlm_at = time.monotonic()
        self._vlm_stage = ev.stage
        logger.info("[%s] VLM 描述已更新（stage=%s）%d 字，快照 %d 键：%r",
                    self.session_id, ev.stage, len(self._vlm),
                    len(self._vlm_state), self._vlm[:400])
        # ---- IC 的视觉状态（P05 `apply_vision`）----
        # 与下面推给 Agent 的「视觉轮」**各走各的出口**：这里进 IC 的状态机
        # （以后 policy 能用上画面），下面那条进 Agent 的 `scene` 背景。
        # 两者都只读快照，谁先谁后不影响正确性 —— 但放在这里（而不是
        # `_emit_visual_background` 里）是为了"IC 掉线时也更新"，
        # 与 `_vlm` 缓存的取向保持一致。
        self._push_vision_to_ic()
        # ---- 有了就推：VLM **不与 ASR 强行同步**（2026-10-28 用户口径）----
        #
        # 每份描述（full 与 turn 一样）一生成完就**立刻**作为「ASR 空的视觉轮」
        # 推给 Agent（只存不答），而不是塞进缓存等下一次 `on_answer` 捎带。
        # 两条理由：
        #  · 断句触发的描述本来就晚于本句的 `on_answer`（窗口几十毫秒 vs
        #    decode 秒级）⇒ 缓存只会让 Agent 永远看到「上一句那时候」的画面；
        #  · 用户不说话、画面却变了的时候，也该及时更新 —— 缓存模式下没有
        #    `on_answer` 就永远发不出去。
        # 每份都是 8 键全量快照 ⇒ 直接覆盖，不存在"旧值残留"。
        # **必须在 `self._vlm = text` 之后同步调**（sink 的 getter 读的就是它）。
        # 传**缓存里那份**（已渲染成快照），不是模型原文 —— 日志里的字数要能
        # 对上 Agent 实际收到的东西。
        #
        # ⚠️ 新版 IC 自己会发：ea4ea22 起 `Engine.apply_vision` 写完状态就调
        #    `agent_sink.on_vision(...)`（ASR 空的 ANSWER）。IC 的 ADR 0001
        #    要求编排侧**同一会话不要再投** `emit_visual_background`，否则每份
        #    描述变成两轮、后一轮覆盖前一轮。
        #    只在 `parsed` 时让路：解析失败时 `_push_vision_to_ic` 什么也没写
        #    ⇒ IC 不会发；此时原文仍要走编排侧那一轮（保底不丢数据）。
        if parsed and self._ic_owns_vision_round():
            return
        self._emit_visual_background(len(self._vlm))

    def _ic_owns_vision_round(self) -> bool:
        """IC 自己会不会把这份快照当 Vision 通知发给 Agent（见上）。

        IC 客户端用鸭子接口回答（进程内：探测 sink 上的 `on_vision`；
        gRPC：恒假）。老客户端 / 探测异常一律当"不会" —— 编排侧照旧补一轮，
        宁可多一轮也不能让 Agent 收不到画面。
        """
        fn = getattr(self.ic, "ic_owns_vision_round", None)
        if not callable(fn):
            return False
        try:
            owns = bool(fn())
        except Exception as exc:  # noqa: BLE001
            logger.debug("[%s] 探测 IC 的 Vision 通知能力失败：%s",
                         self.session_id, exc)
            return False
        if owns and not self._ic_vision_logged:
            self._ic_vision_logged = True
            logger.info("[%s] 视觉通知改由 IC 自己发（ApplyVision → on_vision）"
                        "—— 编排侧不再补视觉背景轮", self.session_id)
        return owns

    def _emit_visual_background(self, n_chars: int) -> None:
        # Structured snapshots already publish exactly once from Engine.apply_vision.
        # Unparsed but useful free text still reaches the Agent through the host.
        if self._ic_owns_vision_round() and self._vlm_state:
            return
        if self._vlm:
            self._events.put(("action", {"type": "vision", "session_id": self.session_id,
                                         "vlm": self._vlm}))

    def _note_vlm_readiness(self) -> None:
        """`AsrFinal` 到了 —— 记下**Agent 手上那份视觉状态**有多旧（纯度量）。

        2026-10-28 起 VLM 与 ASR **不再同步发送**（描述一生成完就单独推给
        Agent，见 `_emit_visual_background`），所以这里不再是"本句有没有带上
        VLM"的判据，只剩一个可观测性指标：**断句那一刻，Agent 看到的画面有多
        旧**。数字大就说明描述触发得太稀（断句/心跳都没跟上）。

        ⚠️ **纯度量，不做任何等待/补触发** —— 等就等于把 VLM 的生成耗时加到
        ASR→Agent 这条链路上。
        """
        if self._vlm:
            age_ms = int((time.monotonic() - self._vlm_at) * 1000)
            logger.info("[VLM 状态] 断句点最近一份描述（stage=%s，年龄=%dms，%d 字）"
                        "—— 已单独推给 Agent，与本句 ASR 无关",
                        self._vlm_stage or "-", age_ms, len(self._vlm))
        else:
            self._vlm_miss += 1
            logger.warning("[VLM 状态] 断句点还没有任何描述（第 %d 次）—— "
                           "Agent 手上没有视觉状态，但**不等待**",
                           self._vlm_miss)

    def _should_report(self, atype: str, sop: Optional[str]) -> bool:
        """这次决策要不要下发给客户端 —— **变化时立刻发，不变时定时刷**。

          · `(atype, sop)` 与上次不同 → 立即发（状态真的切了）
          · 相同 → 距上次下发超过 `REPORT_INTERVAL_S` 才补一条心跳

        为什么比 `(atype, sop)` 而不是只比 `atype`：`WAIT` 带不同 `sop`
        代表**不同的等待原因**（06=嘴还在动、21=路人/脸不够清），只比
        atype 会把这种切换吞掉。

        为什么不比 `transcript`：它在稳态下**每拍都在变**（流式累积），
        拿它当判据等于没节流。但下发时会把当前内容带上（回调里取
        `action.text`），所以信息不丢。

        ⚠️ **这里只"判断该不该发"，不记账** —— 记账由调用方在**确认送出后**
        执行（见 `_dispatch`）。早先在这里就更新 `_last_reported_key`，于是
        送失败时标记已置位、下一拍不再重发 —— 恰好丢掉 `GREET` 这类
        一瞬就过去的动作（D16/D18 判题失败的根因）。
        """
        import time
        now = time.monotonic()
        key = (atype, sop)
        if key != self._last_reported_key:
            return True
        # ⚠️ key 相同，但**上一次没送出去**（`_pending_report`）→ 重试。
        #    这条路径专门救 `GREET` 这类一瞬就过去的动作：它只出现一两拍，
        #    那一两拍内送不出去就永远没了。
        #
        #    ⚠️ 重试**只针对"没送出去"**，不是"每次都发" —— 早先写成
        #    「一瞬动作永不走心跳节流」，结果**送出成功后仍每拍重发**，
        #    IC / replay 会收到重复的 GREET（实测发现）。
        if self._pending_report:
            return True
        return now - self._last_report_at >= self.REPORT_INTERVAL_S

    def _mark_reported(self, atype: str, sop: Optional[str]) -> None:
        """确认送出后记账 —— 清掉"待重试"标记。"""
        import time
        self._last_reported_key = (atype, sop)
        self._last_report_at = time.monotonic()
        self._pending_report = False

    # ------------------------------------------------------------------ #
    #  Tick → 要决策
    # ------------------------------------------------------------------ #

    async def _on_tick(self) -> List[DownstreamAction]:
        import time
        if not self.ic.available:
            return []

        now = time.monotonic()
        dt_ms = 0
        if self._last_tick is not None:
            dt_ms = int(min(2000.0, (now - self._last_tick) * 1000))
        self._last_tick = now

        action = await self.ic.tick(dt_ms=dt_ms)
        if action is None:
            return []
        return self._dispatch(action)

    def _dispatch(self, action) -> List[DownstreamAction]:
        """IC 的 Action → 我们的动作。"""
        atype = str(getattr(action, "type", "") or "")
        # ActionType 是 str Enum，``str(x)`` 在 py3.11 前后不一致，统一取 value
        atype = getattr(getattr(action, "type", None), "value", atype)
        sop = getattr(action, "sop", None)

        self.counts[atype] = self.counts.get(atype, 0) + 1
        self.actions.append({"type": atype, "sop": sop, "t": self._last_tick})
        if len(self.actions) > 200:
            del self.actions[:-200]

        key = (atype, sop, getattr(action, "transcript", None))
        fresh = key != self._last_action_key
        # ⚠️ 「切入 GREET」的判据 —— **不能用上面的 `fresh`**。
        #
        # `_last_action_key` 只在**非 QUIET** 动作时更新（见下面 `:722`，QUIET
        # 在 `:719-720` 提前 return），而 IC 迎宾后会长久停在
        # LISTEN/WAIT/HOLD。于是**换人后的第二次 GREET**，其
        # `(atype, sop, transcript)` 与上次 GREET 可能**完全相同** ⇒ `fresh`
        # 为 False ⇒ 漏掉重迎宾，表现为「换了个人却不重新描述画面」。
        # `_prev_atype` 只看**上一拍**，与中间夹了多少 QUIET 无关。
        prev_atype, self._prev_atype = self._prev_atype, atype

        # ---- 下发给客户端：**变化时立刻发，不变时定时刷** ----
        #
        # ⚠️ 演进过程（两次都踩过，别再退回）：
        #   ① 最早对常见态**直接 `return []`**（且在调 `_on_action` 之前），
        #      于是客户端根本收不到 LISTEN/WAIT/HOLD —— viz 的 IC 时间轴上
        #      只剩 6 种非常见态，Policy「一直在等待」这段过程完全看不出来。
        #   ② 改成每 tick 全量发（50ms 一条）。时间轴没缺口了，但量大
        #      （一场 6 分钟约 7000 条），而且**绝大多数是重复的**——
        #      稳态下 IC 连着几百拍返回同一个 HOLD，逐条发没有信息量。
        #
        # 现在：**状态变了立刻发；没变则每 `REPORT_INTERVAL_S` 补一条心跳。**
        #   · 变化 = (atype, sop) 不同 → 立即下发，零延迟（viz 看得到切换点）
        #   · 不变 = 每 REPORT_INTERVAL_S 一条 → 证明「决策还在跑」，而非卡死
        #   · transcript 变不算"变化"（它在稳态下每拍都在变），但**内容会带上**
        #
        # 心跳频率由 `REPORT_INTERVAL_S` 控制，会话建立时可用
        # `report_interval_s` 覆盖（`InteractionDownstream` 的构造参数）。
        # ⚠️⚠️ **只有"真的送出去了"才记 `_last_reported_key`**。
        #
        # 早先这里先记 key、再调 `_on_action` —— 而 `_on_action` 底层是
        # **可丢的显示队列**（满了就丢）。于是：
        #   ① 队列满 → 这条被丢
        #   ② 但 key 已经记成「已上报」
        #   ③ 下一拍动作变成 `HOLD`，key 也变了 → **永远不会补发那条**
        # 恰好丢掉的就是 `GREET` 这种**一瞬就过去**的动作，表现为
        # 「喇叭响了、离线 dump 里却没有 GREET」（D16/D18 判题失败，实测踩过）。
        #
        # 现在：送失败就**不记 key**，下一拍会重试同一条，直到送达。
        # （IC 动作队列已改为不丢，正常情况下一次就成；这是第二道保险。）
        if self._on_action is not None and self._should_report(atype, sop):
            try:
                ok = self._on_action(atype, sop, getattr(action, "text", None))
            except Exception:  # noqa: BLE001
                ok = False
            if ok is False:
                # 没送出去 —— 置「待重试」，下一拍会重发**同一条**动作。
                self._pending_report = True
                logger.warning("[%s] IC 动作 %s（sop=%s）**下发失败**，"
                               "下一拍重试 —— 该动作若丢失，判题侧会漏记",
                               self.session_id, atype, sop)
            else:
                self._mark_reported(atype, sop)

        # 下面只处理「要不要产生动作」—— 与上报无关
        if atype in self.QUIET_TYPES:
            return []              # 常见态**永不产生动作**，只做上报

        self._last_action_key = key

        if atype == "ANSWER":
            logger.info("[%s] IC → ANSWER（sop=%s）转写=%r（由 IC 派给 Agent）",
                        self.session_id, sop,
                        (getattr(action, "transcript", "") or "")[:40])
            # ⚠️ **编排侧不转发给 Agent** —— IC 自己会发。
            #
            # IC 判 ANSWER 时经 `AgentSink.on_answer()` POST 到 Agent
            # （interactioncore 的 `runtime.py._notify_sinks`），且**字段更全**
            # （从 IC 的 state 读 identity_id / display_name / transcript）。
            # 编排侧再发一份 ⇒ Agent 收到**两份 on_answer**，行为变成
            # "取决于 Agent 内部如何处置重复"，而不是我们能保证的。
            #
            # 早先这里是兜底（`ORCH_AGENT_RELAY`，默认已是关的），现已整个删除。
            return []

        if atype == "INSERT":
            logger.info("[%s] IC → INSERT（sop=%s）放行慢结果（由 IC 派给 Agent）",
                        self.session_id, sop)
            return []

        if atype == "YIELD":
            logger.info("[%s] IC → YIELD（sop=%s）用户抢话，停播", self.session_id, sop)
            # ⚠️ **编排侧不主动停播** —— 由 IC 自己调 `POST /v1/stop` 完成。
            #
            # IC 判 YIELD 时会经 `ExpressionSink.stop()` 发 `/v1/stop`
            # （interactioncore 的 `runtime.py._notify_sinks`），那才是
            # 设计上该驱动停播的地方。编排侧再 `return [Cancel]` 就是
            # **同一件事做两遍**：两条 Cancel 都会掐断浏览器 + 截参考轨，
            # 后到的那条作用在已被截断的轨上（`ref 清 0 采样`）。
            #
            # 早先这里返回 Cancel 是**兜底**（IC 的 HTTP 通路曾因证书
            # 静默失效，见 `_is_duplicate_speak` 的说明）。现在按
            # 「编排侧只做编排，播报/打断由 IC 与 Agent 主动调用」的划分
            # 去掉 —— 重复比兜底的收益更明确。
            # 需要恢复兜底时：把下面这行取消注释。
            # return [Cancel(reason="barge_in")]
            return []

        if atype == "END":
            logger.info("[%s] IC → END（sop=%s）会话收尾（停播由 IC 发起）",
                        self.session_id, sop)
            # ⚠️ **编排侧不转发、不主动停播** —— IC 自己会做两件事
            #    （interactioncore 的 `runtime.py._notify_sinks`）：
            #      · `AgentSink.on_end()`      → 通知 Agent
            #      · `ExpressionSink.stop()`   → `POST /v1/stop` 停播
            #
            # 历史（别再往回退）：
            #   · 早先这里 `self.agent.on_end()` + `return [Cancel]`，与 IC
            #     **完全重复**。
            #   · 9-24 我一度以为 IC 没有停播机制，按 sop 给 24/23 加了
            #     Cancel 兜底 —— 诊断是错的：IC 一直有，只是它的 HTTP 通路
            #     被证书问题切断（证书 SAN 缺 106），编排日志里 `ic_stop`
            #     从 15 次归零。证书修好后实测已恢复：
            #         `外部请求停播（reason=ic_stop）` → `打断 ...（ic_stop）`
            return []

        # ---- GREET / UTTER：**编排侧不播**，由 IC 自己下发 ----
        #
        # IC 判出 GREET/UTTER 时会调 `ExpressionSink.play_template(text)`
        # → `POST /v1/speak`（interactioncore 的 `runtime.py._notify_sinks`）。
        # 那才是播报的**唯一**来源。
        #
        # ⚠️ 早先这里 `return [Speak(text)]`，于是同一句话被播**两遍**
        #    （IC 一份 + 这里一份），靠 `_is_duplicate_speak` 去重挡住第二遍。
        #    那是"事后补救"：重复依然产生，只是被拦下。
        #    现在按「编排侧只做编排，播报/打断由 IC 与 Agent 主动调用」的
        #    划分去掉 —— 从源头不再产生重复，去重也随之成为纯粹的防御。
        #
        # ⚠️ 代价：迎宾/切句提示**完全依赖 IC 的 HTTP 通路**。该通路实测会因
        #    证书 / 网络 / 找不到 session(404) 静默失效（失败只记 IC 侧日志）。
        #    恢复兜底：把下面 `return [Speak(...)]` 取消注释。
        if atype in ("GREET", "UTTER"):
            text = getattr(action, "text", None)
            logger.info("[%s] IC → %s（sop=%s）: %s（由 IC 自行播报）",
                        self.session_id, atype, sop, (text or "")[:40])
            # return [Speak(text=text)]        # ← 兜底，默认关
            # ---- 切入 GREET ⇒ 让 OmniLLM 出**一份全量描述** ----
            #
            # 为什么挂在 GREET 上：GREET 是 IC 认定的「新的人/换人了」——
            # 正是需要把画面重新讲一遍的时刻（用户明确要求就接在这里）。
            # 之后画面持续输入，靠 `session` 的滚动增量刷新跟上变化。
            #
            # ⚠️ 只用 `prev_atype != "GREET"` 判「切入」。同一段 GREET 会连着
            #    好几拍，逐拍触发就是每 50ms 一次全量推理 —— 会把 GPU 打满，
            #    而且是本次改动最容易造成的自伤。代价：IC 若真的
            #    GREET→LISTEN→GREET 抖动，会多发一次全量（可接受，且日志里
            #    看得见）。
            if atype == "GREET" and self._desc_enabled and prev_atype != "GREET":
                logger.info("[%s] 切入 GREET（sop=%s）⇒ 触发全量 VLM 描述",
                            self.session_id, sop)
                return [Describe(stage="full")]
        return []
