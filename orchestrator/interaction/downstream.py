"""InteractionCore + Agent 的 ``Downstream`` 实现。

替换 ``PassthroughDownstream``：回复不再由 OmniLLM 生成，而是

    orchestrator 事件 ──apply_*──▶ InteractionCore（决策）
                                        │ Tick() → Action
                                        ▼
                                  ACTION 派发给 Agent
                                        │ Agent 生成回复
                                        ▼
                        Agent POST /v1/speak ──▶ Speak ──▶ TTS

**本类不产生 Speak** —— 回复文本由 Agent 通过 HTTP 回调送进来（见
``main.py`` 的 ``/v1/speak``）。这里只负责：

  1. 把感知事件喂给 IC（非阻塞）
  2. 每个 Tick 向 IC 要一次决策，把 Action 翻译成「通知 Agent / 停播 / 收尾」
  3. ``GREET`` / ``UTTER`` 直接用 IC 给的模板文案 Speak（不用等 Agent）

⚠️ **IC 的 Action 有 9 种**（不只是 4 种）：

    ANSWER / INSERT / YIELD / END   → 派给 Agent
    GREET  / UTTER                  → 模板文案，直接 Speak
    LISTEN / WAIT / HOLD            → 不动（继续听）

⚠️ IC 的状态是**服务端全局一份**（没有 session 概念），所以一个进程里
不要同时跑多个会话接同一个 IC —— 会互相覆盖。当前部署是单会话场景。
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from ..downstream.interface import (
    AsrFinal, AsrPartial, AsrStateUpdate, Cancel, DownstreamAction,
    DownstreamEvent, FaceIdentity, FaceLipState, FaceState, FaceWake,
    PlaybackReceipt, Speak, Tick,
)
from .agent import AgentClient
from .client import InteractionClient

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
                 report_interval_s: Optional[float] = None) -> None:
        self.ic = InteractionClient(ic_target)
        self.agent = AgentClient(agent_url)
        self.session_id = session_id
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
            ],
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
        self.agent.start()          # 幂等（_thread 非空直接返回）
        return []

    async def on_session_end(self, reason: str) -> None:
        try:
            self.agent.on_end()
        except Exception:  # noqa: BLE001
            pass
        self.agent.close()
        self.ic.close()

    @property
    def available(self) -> bool:
        """IC 是否可用 —— 不可用时调用方应降级到 OmniLLM 回复。"""
        return self.ic.available

    # ------------------------------------------------------------------ #
    #  事件 → IC
    # ------------------------------------------------------------------ #

    async def on_event(self, ev: DownstreamEvent) -> List[DownstreamAction]:
        # ---- 每次 Tick 向 IC 要一次决策 ----
        if isinstance(ev, Tick):
            return await self._on_tick()

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

        return []

    def _feed(self, ev: DownstreamEvent) -> None:
        """把事件映射成 IC 的 ``apply_*``（**只投队列，立即返回**）。"""
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
            self.ic.apply(
                "apply_asr",
                transcript=str(st.get("transcript") or ""),
                asr_confidence=_conf(st.get("asr_confidence")),
                turn_complete_confidence=_conf(st.get("turn_complete_confidence")),
            )

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
        """
        import time
        now = time.monotonic()
        key = (atype, sop)
        if key != self._last_reported_key:
            self._last_reported_key = key
            self._last_report_at = now
            return True
        if now - self._last_report_at >= self.REPORT_INTERVAL_S:
            self._last_report_at = now
            return True
        return False

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
        if self._on_action is not None and self._should_report(atype, sop):
            try:
                self._on_action(atype, sop, getattr(action, "text", None))
            except Exception:  # noqa: BLE001
                pass

        # 下面只处理「要不要产生动作」—— 与上报无关
        if atype in self.QUIET_TYPES:
            return []              # 常见态**永不产生动作**，只做上报

        self._last_action_key = key

        if atype == "ANSWER":
            if not fresh:
                return []
            logger.info("[%s] IC → ANSWER（sop=%s）转写=%r", self.session_id,
                        sop, (getattr(action, "transcript", "") or "")[:40])
            # 交给 Agent —— 它生成后调 /v1/speak 回来
            self.agent.on_answer(
                transcript=getattr(action, "transcript", None) or self._transcript,
                identity_id=self._identity_id,
                display_name=self._display_name,
            )
            return []

        if atype == "INSERT":
            logger.info("[%s] IC → INSERT（sop=%s）放行慢结果", self.session_id, sop)
            self.agent.on_insert()
            return []

        if atype == "YIELD":
            logger.info("[%s] IC → YIELD（sop=%s）用户抢话，停播", self.session_id, sop)
            self.agent.on_yield()
            # 真正停播 —— 复用既有的 Cancel 链路（掐断浏览器 + 截参考轨）
            return [Cancel(reason="barge_in")]

        if atype == "END":
            logger.info("[%s] IC → END（sop=%s）会话收尾", self.session_id, sop)
            self.agent.on_end()
            return []

        # ---- GREET / UTTER：IC 已给出模板文案，直接播 ----
        text = getattr(action, "text", None)
        if atype in ("GREET", "UTTER") and text:
            logger.info("[%s] IC → %s（sop=%s）: %s",
                        self.session_id, atype, sop, text)
            return [Speak(text=text)]
        if atype in ("GREET", "UTTER"):
            logger.warning("[%s] IC → %s 但没有文案（text 为空），跳过",
                           self.session_id, atype)
        return []
