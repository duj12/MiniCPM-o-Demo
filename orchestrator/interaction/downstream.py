"""InteractionCore + Agent 的 ``Downstream`` 实现。

替换 ``PassthroughDownstream``：回复不再由 OmniLLM 生成，而是

    orchestrator 事件 ──apply_*──▶ InteractionCore（决策）
                                        │ Tick() → Action
                                        ▼
                                  ACTION 派发给 Agent
                                        │ Agent 生成回复
                                        ▼
                        Agent POST /v1/speak ──▶ Speak ──▶ TTS

**本类不产生 Speak，也不主动 Cancel** —— 播报与打断**全部**由 IC / Agent
通过 HTTP 主动调用（见 ``main.py`` 的 ``/v1/speak`` 与 ``/v1/stop``）。

> ⚠️ 这是**刻意的职责划分**，别再往这里加播报/打断：
>   · **IC**    判 GREET/UTTER → `ExpressionSink.play_template` → `/v1/speak`
>   · **IC**    判 YIELD/END   → `ExpressionSink.stop`         → `/v1/stop`
>   · **Agent** 生成完回复     → `/v1/speak`（整段或流式）
>
> 早先本类**也**做这些（`return [Speak]` / `return [Cancel]`），结果是
> 同一句话被播两遍、同一次打断被执行两次，靠去重和"后到的 Cancel 作用在
> 已截断的轨上"掩盖着。**从源头去掉**才是干净的 —— 去重保留，但降级为
> 纯粹的防御性检查（见 `executor._is_duplicate_speak`）。

这里只负责：

  1. 把感知事件喂给 IC（非阻塞）
  2. 每个 Tick 向 IC 要一次决策，把 Action 翻译成「通知 Agent / 记日志 / 收尾」
  3. ``GREET`` / ``UTTER`` 只记日志（播报由 IC 自己发起）

⚠️ **IC 的 Action 有 9 种**（不只是 4 种）：

    ANSWER / INSERT / YIELD / END   → 事件派给 Agent（默认关，见
                                      `agent._AGENT_RELAY`）
    GREET  / UTTER                  → 只记日志；播报由 IC 自己发起
    LISTEN / WAIT / HOLD            → 不动（继续听）

⚠️ IC 的状态是**服务端全局一份**（没有 session 概念），所以一个进程里
不要同时跑多个会话接同一个 IC —— 会互相覆盖。当前部署是单会话场景。
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

from ..omni.describe import is_no_change, looks_conversational

from ..downstream.interface import (
    AsrFinal, AsrPartial, AsrStateUpdate, Describe, DownstreamAction,
    DownstreamEvent, FaceIdentity, FaceLipState, FaceState, FaceWake,
    OmniDescription, PlaybackReceipt, Tick,
    # ⚠️ `Speak` / `Cancel` **当前没有代码使用** —— 编排侧不再主动播报/打断
    #    （见上面的类文档）。它们被注释掉的兜底分支引用着，取消注释时要用，
    #    所以**留着 import**；`flake8` 若报 F401 属预期，别顺手删。
    Speak, Cancel,
)
from .agent import AgentClient
from .client import InteractionClient

logger = logging.getLogger(__name__)

#: **模块级**记录「Agent 当前被设成了哪个 IC 地址」。
#:
#: ⚠️ 必须是模块级而非实例级 —— Agent 的那个设置是**进程级全局**的
#: （见 `AgentClient.set_ic_target`），多个会话共享同一个 Agent 状态，
#: 每个会话各自拿实例变量记会记岔。这里只做**记账**，供
#: 「结束时该不该归还」和「有没有被别的会话覆盖」这两处判断使用。
#: None = 本进程还没设过（Agent 用它自己的默认值）。
_AGENT_IC_TARGET: Optional[str] = None

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
                 report_interval_s: Optional[float] = None,
                 restore_ic_target: str = "",
                 ic_advertise: str = "",
                 ic_mode: str = "grpc",
                 ic_expression_url: str = "",
                 ic_expression_ca: Optional[str] = None,
                 ic_callback_ic: Optional[str] = None,
                 agent_set_target: bool = True,
                 omni_describe: bool = False,
                 agent_transcript_mode: str = "legacy") -> None:
        #: 描述模式总开关（`ORCH_OMNI_DESCRIBE`）。**只管两件事**：
        #:   ① 决定在 IC 判 GREET 时要不要发 `Describe("full")`
        #:   ② 决定要不要给 IC 的 sink 装 VLM 包装（`agent_transcript_mode`
        #:      也必须是非 legacy 才有意义）
        #: 它**不**改 omni 的人设 —— 那是 `main.py` 按它挑 system prompt。
        self._desc_enabled = bool(omni_describe)
        #: Agent 载荷形态：legacy（逐字节旧版）/ dual / dict。
        #: ⚠️ 默认值是**安全侧**，与 `config.agent_transcript_mode` 一致 ——
        #:    编排代码 105/106 共用，默认开等于替 106 决定了对 Agent 的协议。
        self._agent_transcript_mode = (agent_transcript_mode or "legacy").strip().lower()
        #: 最近一份**生成完**的 VLM 描述（见 `_on_vlm_description`）。
        #: 由 `agent_sink` 的 getter 在 `on_answer` 的**断句那一刻**读取 ——
        #: 所以这里必须是一份"随时可用"的缓存，取值路径上**不做任何生成**。
        self._vlm = ""
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
        #: 本会话是否已把「会话起点的视觉背景」投给 Agent（见
        #: `_emit_visual_background`）。**只发本会话的第一份全量描述** ——
        #: 理由见那里的说明。`on_session_start` 必须复位它。
        self._bg_sent = False
        self.ic_mode = (ic_mode or "grpc").strip().lower()
        if self.ic_mode == "inprocess":
            #: 每路会话**独占**一份进程内 Engine（并发由构造消解）。
            from .inprocess import InProcessICClient
            self.ic = InProcessICClient(
                session_id,
                expression_url=ic_expression_url,
                agent_url=agent_url,
                expression_ca=ic_expression_ca,
                callback_ic=ic_callback_ic,
                # VLM 只在"描述模式生效 **且** 真要改 payload"时才装包装 ——
                # 任一条不满足都传 None，sink 走原始路径（payload 与旧版
                # 逐字节一致）。
                vlm_getter=(lambda: self._vlm)
                if (self._desc_enabled
                    and self._agent_transcript_mode != "legacy") else None,
                agent_transcript_mode=self._agent_transcript_mode,
            )
            logger.info("[%s] IC 模式=进程内 Engine（本会话独占，无单例限制）",
                        session_id or "?")
        else:
            # 带上会话 id —— 用于「单一驱动者」接管时的日志定位，以及
            # 让 IC 的 tick/apply 在被别的会话接管后能被正确挂起。
            self.ic = InteractionClient(ic_target, owner_key=session_id)
        #: 过渡开关：Agent 尚未支持从 payload 读 callback_ic 时保留 set_target
        self._agent_set_target = bool(agent_set_target)
        # ⚠️ `session_id` 必须传进去 —— 会话收尾的 `on_end` 要带上它，
        #    否则 Agent 分不清是哪一路会话结束，会**跨会话误杀**
        #    （见 `AgentClient.engine_on_session_end` 的实测证据）。
        self.agent = AgentClient(agent_url, session_id=session_id)
        self.session_id = session_id
        #: 会话结束时把 Agent 的 IC 目标**归还**到哪 —— 传服务端配置里的
        #: 默认 IC（**对外可回连的地址**）。空 = 不归还（调用方没给默认值）。
        self._restore_ic = restore_ic_target or ""
        #: **告诉 Agent 回连哪个地址**。空 = 退回 `self.ic.target`（同机部署
        #: 时的常见情形）。跨机部署**必须**显式给 —— 见 `config.ic_advertise`
        #: 的说明（把 `127.0.0.1` 告诉 Agent 会派错，还会污染 106）。
        self._ic_advertise = ic_advertise or self.ic.target
        #: 本会话是否真的设过 Agent 的目标（没设过就不该归还 —— 否则会把
        #: 别人设的值改掉）
        self._agent_ic_set_by_me = False
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
            "transcript_mode": self._agent_transcript_mode,
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
        self._vlm_at = 0.0
        self._vlm_stage = ""
        self._prev_atype = ""
        # ⚠️ 与上面两项**同理、且必须一起清**：`_bg_sent` 若不随新会话复位，
        #    新会话的首份全量描述会被上一会话的闩挡在门外 ⇒ Agent 的
        #    `scene.visual_background` 又是空的 —— **症状与本次要修的那个 bug
        #    一模一样**（"Agent 拿着一个从没收到的背景"），更难往这里想。
        self._bg_sent = False
        self.agent.start()          # 幂等（_thread 非空直接返回）
        self._sync_agent_ic_target()
        return []

    # ------------------------------------------------------------------ #
    #  Agent 的 IC 目标同步
    # ------------------------------------------------------------------ #

    def _sync_agent_ic_target(self) -> None:
        """把**本会话的** IC 地址告诉 Agent。

        Agent 收到 IC 的 Action 后要靠这个地址回连。不同的人用不同的 IC
        （比如别人 replay 时用自己的 IC 服务），不告诉它就会派到 Agent
        默认的那个（106）——于是 replay 收不到自己的 Action，表现为
        「IC 决策一直不对 / 判题全错」。

        ⚠️⚠️ **Agent 的这个设置是进程级全局的**（端点收单数
        ``interaction_core_target``，无 session 维度）。所以并发时后设的
        覆盖先设的。这一版**只同步 + 告警**，不去串行化 —— 详见
        ``AgentClient.set_ic_target`` 的说明。

        ⚠️ 冲突**只告警不阻断**：真实部署里 106 的 IC 是共享的，两个会话
        用同一个 IC 完全正常，那不算冲突。只有当地址**不同**时才说明
        「Agent 只能指向其中一个」，这时另一方的 Action 一定会派错。
        """
        # 过渡开关：Agent 已能读 payload 里的 callback_ic 时关掉这里
        if not self._agent_set_target:
            logger.info("[%s] 跳过 Agent 的 set_target（ORCH_AGENT_SET_TARGET=0）"
                        "—— 改由 IC 在 AgentSink payload 里带 callback_ic",
                        self.session_id)
            return
        # ⚠️ `global` 必须在**首次使用之前**声明（否则 SyntaxError）
        global _AGENT_IC_TARGET
        want = self._ic_advertise     # 注意：不是 self.ic.target（见下）
        prev = _AGENT_IC_TARGET
        if prev == want:
            return                      # 已经是本会话的地址，不必重复设
        if prev is not None and prev != want:
            logger.warning(
                "[%s] ⚠️ Agent(%s) 的 IC 目标正被另一个会话指向 %s，"
                "本会话将改为 %s —— **Agent 是进程级全局单例**，"
                "被覆盖的那一方 IC Action 会派错。并发用不同 IC 时无解，"
                "需 Agent 侧支持每会话 target。",
                self.session_id, self.agent.target, prev, want)
        if self.agent.set_ic_target(want):
            _AGENT_IC_TARGET = want
            self._agent_ic_set_by_me = True

    async def on_session_end(self, reason: str) -> None:
        """会话收尾（客户端断开 / 点停止）。

        ## ⚠️ 只调 `IC.end_session()`，**编排侧不再做别的**

        IC 的 ``end_session()`` 内部会一次做完三件事
        （见 interactioncore 的 ``runtime.py``）：
            · ``AgentSink.on_end()``      → 通知 Agent 结束
            · ``ExpressionSink.stop()``   → `POST /v1/stop` 停播
            · ``reset_all_state()``       → 清状态

        所以编排侧**不需要**、也**不应该**再自己：
            · 通知 Agent（`engine_on_session_end` —— 已删）
            · 归还 Agent 的 IC 目标（`set_ic_target` —— 已删）
        那些是跟 IC 重复的第二份，且**跨机时会互相踩**
        （Agent 的 `interaction_core_target` 是进程级全局单例）。

        编排服务的定位是「**只做编排**」：播报、打断、通知 Agent 一律
        由 IC 与 Agent 自己发起。
        """
        # ⚠️ 必须在 `ic.close()` **之前**调 —— 它要发 gRPC / HTTP。
        #
        # ⚠️⚠️ **必须 off-loop**：进程内模式下 `end_session()` 里有**同步 HTTP**
        #     （`ExpressionSink.stop` 打 `/v1/stop`，超时 1s），`close()` 还要
        #     flush + join 两个 sink 的 worker 线程（各 2s）。走 gRPC 时这些都
        #     在服务端线程上，编排侧无感；进程内会**阻塞事件循环**，把同一
        #     loop 上其他会话的 tick / 音频路径一起卡住。
        await asyncio.to_thread(self._end_and_close)

    def _end_and_close(self) -> None:
        """收尾的**同步**部分 —— 由 `on_session_end` 放进线程里跑。"""
        self.ic.end_session()
        self.agent.close()
        self.ic.close()

    # ------------------------------------------------------------------ #
    #  Agent → IC：apply_agent（智脑薄投影）
    # ------------------------------------------------------------------ #

    def apply_agent(self, status: Optional[str] = None,
                    session_end_pending: Optional[bool] = None) -> None:
        """Agent 写 `agent.status` / `session_end_pending`。

        由 `main.py` 的 ``POST /v1/ic/apply_agent`` 路由过来（Agent 从 IC 的
        payload 里拿到 `callback_ic` 后就把写入投到这里）。

        ⚠️ 这一步是 `ORCH_IC_MODE=inprocess` 的**硬前置**：Engine 进了编排
        进程之后，Agent 原来直连 IC gRPC `ApplyAgent` 的那条路就不存在了。
        Agent 不改，`agent.status` / `session_end_pending` 永远写不进去
        ⇒ **SOP 39 与 PENDING_ANNOUNCE 两条分支失效**。

        两种模式**同名同语义**（`InProcessICClient` / `InteractionClient` 都
        实现 `apply_agent`），所以调用方不必按模式分支。

        ⚠️ **同步**（grpc 模式是一次 gRPC 往返）—— 调用方要 off-loop。
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
        self._vlm_at = time.monotonic()
        self._vlm_stage = ev.stage
        logger.info("[%s] VLM 描述已更新（stage=%s）%d 字：%r",
                    self.session_id, ev.stage, len(text), text[:60])
        # 全量描述 ⇒ 补一发「会话起点的视觉背景」给 Agent。**必须在
        # `self._vlm = text` 之后同步调**（sink 的 getter 读的就是它）。
        if ev.stage == "full":
            self._emit_visual_background(len(text))

    def _emit_visual_background(self, n_chars: int) -> None:
        """把刚写进缓存的全量描述当「视觉背景轮」投给 Agent（ASR 为空）。

        Agent 收到 `content` 且 `ASR`/`transcript` 都空 ⇒ 走
        `handle_visual_background`（**只存不答**，写 `scene.visual_background`）。
        没有这一发，Agent 收到的 `scene.visual_change` 就在引用一个它**从没
        收到过**的背景（全量只活在本地缓存里、随即被增量覆盖）。

        ⚠️ 这**不是** `agent.py` 里删掉的那种「转发 IC 自己也会发的事件」
        （那次让 Agent 收到两份 `on_answer`）：IC 的 `on_answer` 只在 ANSWER
        时发、`transcript` 必非空，**结构上产生不了**这一轮。而且它走同一个
        sink 队列 ⇒ 与 IC 的真实 `on_answer` 端到端串行，顺序确定。
        别把它当重复投递"修掉"。

        **只发第一份**（`_bg_sent` 闩）：第一份全量做完后
        `session._maybe_switch_delta_persona` 会把 omni 换成「只报变化」的短
        人设（**幂等**，只重连一次）。换人后的第二次 GREET 仍会派
        `Describe("full")`，但它是在短人设下答的 ⇒ 回的是**几十字的变化描述**；
        拿它覆盖背景就是用 46 字冲掉 476 字的有效背景 —— 正是本要在修的 bug
        换了个入口。Agent 那个字段的名字就是「**会话起点**的视觉背景」，
        换人后的差异由每轮的 `scene.visual_change` 承载。
        """
        if not self._desc_enabled:
            return
        if self._bg_sent:
            logger.info("[%s] 已有视觉背景（stage=full %d 字）—— 不重复投递"
                        "（换人后的差异走每轮的 change）", self.session_id, n_chars)
            return
        # ⚠️ 闩**只在成功交出后**才置：`_feed` 处理 OmniDescription 是在
        #    `if not self.ic.available: return` **之前**（那是故意的，见 `_feed`），
        #    所以 IC 掉线时也会走到这儿。若先闩后发，一次瞬时失败就**永久**
        #    丢掉这份背景。
        if self.ic.send_visual_background():
            self._bg_sent = True
            logger.info("[%s] 视觉背景已交给 Agent（stage=full，%d 字）"
                        "—— 后续轮次会注入 scene.visual_background",
                        self.session_id, n_chars)
        else:
            logger.warning("[%s] 视觉背景**未能交出**（无 Agent sink / gRPC 模式 / "
                           "dict 模式 / worker 已关）—— 不置闩，下份全量再试",
                           self.session_id)

    def _note_vlm_readiness(self) -> None:
        """`AsrFinal` 到了 —— 记下缓存**此刻**的状态（主指标的度量）。

        ⚠️ **纯度量，不做任何等待/补触发**。这里若是"没就绪就再等 100ms"，
        就等于把 VLM 的生成耗时加到了 ASR→Agent 这条链路上，直接违反硬指标
        （"发送的那一刻不生成"）。
        """
        if self._vlm:
            age_ms = int((time.monotonic() - self._vlm_at) * 1000)
            logger.info("[VLM 就绪] 断句点已有描述（stage=%s，年龄=%dms，%d 字）"
                        "—— 本句随 ASR 一起发给 Agent",
                        self._vlm_stage or "-", age_ms, len(self._vlm))
        else:
            self._vlm_miss += 1
            logger.warning("[VLM 未就绪] 断句点还没有描述（第 %d 次）—— "
                           "本条 on_answer 的 VLM 为空，**不等待**。"
                           "频率高就调小 ORCH_OMNI_DELTA_INTERVAL_S",
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
