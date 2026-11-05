"""One in-process InteractionCore algorithm object per orchestrator session.

Sinks synchronously enqueue events to the host. All Agent connections, TTS,
interrupt execution, and real playback observations belong to the host.
"""
from __future__ import annotations
import logging
import os
from typing import Any, Callable, Optional
from .client import _signature_accepts
logger = logging.getLogger(__name__)
_MAX_FAIL_METHODS = 64

_CONF_FIELDS: dict[str, tuple[str, ...]] = {
    "apply_face": ("face_present_confidence",),
    "apply_lip": ("lip_speaking_confidence",),
    "apply_identity": ("identity_confidence",),
    "apply_vad": ("user_speaking_confidence", "barge_in_confidence"),
    "apply_asr": ("asr_confidence", "turn_complete_confidence"),
}


def _to_confidence(value: Any) -> Any:
    """裸字符串 / 枚举 → `Confidence` 枚举成员；不认识的原样返回。

    **不要在这里抛** —— 编排侧的 `_conf()` 已经保证只给
    HIGH/MEDIUM/LOW/NONE；真给了别的值，让 IC 自己按它的口径处理
    （抛或忽略），别在中间层吞掉或改变行为。
    """
    if isinstance(value, str):
        try:
            from interaction.state import Confidence
        except ImportError:
            return value
        try:
            return Confidence(value.strip().upper())
        except ValueError:
            return value
    return value




class InProcessICClient:
    def __init__(self, session_id: str = "", *,
                 on_agent_event: Callable[[dict], None] | None = None,
                 on_expression_event: Callable[[dict], None] | None = None,
                 thresholds: Any = None):
        self.owner_key = session_id
        self.target = "inprocess"
        self._on_agent_event = on_agent_event
        self._on_expression_event = on_expression_event
        self._thresholds = thresholds
        self._engine = self._agent_sink = self._decision_probe = None
        self.available = False
        self.error = None
        self.tick_calls = self.bad_ticks = self.dropped = 0
        self.suspended = False
        self.suspended_ticks = self.suspended_writes = 0
        self.last_action = ""
        self.failures = {}
        self._caps = {}

    @property
    def engine(self):
        return self._engine

    def connect(self) -> bool:
        if self._engine is not None:
            return self.available
        try:
            from interaction.runtime import AgentSink, Engine, ExpressionSink
            agent_sink = (AgentSink(self._on_agent_event, session_id=self.owner_key)
                          if self._on_agent_event is not None else None)
            expression_sink = (ExpressionSink(self._on_expression_event, session_id=self.owner_key)
                               if self._on_expression_event is not None else None)
            probe = None
            if os.environ.get("ORCH_IC_DECISION_PROBE", "1").lower() not in ("0", "false", "off", "no"):
                from interaction.decision_probe import DecisionProbe
                probe = DecisionProbe()
            self._decision_probe = probe
            self._agent_sink = agent_sink
            self._engine = Engine(agent_sink=agent_sink, expression_sink=expression_sink,
                                  thresholds=self._thresholds, decision_probe=probe)
            self.available = True
            self.error = None
            return True
        except Exception as exc:
            if self._decision_probe is not None:
                self._decision_probe.close()
                self._decision_probe = None
            self.error = f"{type(exc).__name__}: {exc}"
            logger.warning("[%s] Local InteractionCore unavailable: %s", self.owner_key, self.error)
            return False

    def supports(self, method: str, *keys: str) -> bool:
        """IC 的这个 ``apply_*`` 是否接受这些关键字参数（**版本探测**）。

        ⚠️ 探测的是 ``engine.state`` 上的**真方法**，不是 ``engine.apply_asr``：
        ``Engine.__init__`` 用 ``_locked`` 把它们包成 ``(*args, **kwargs)``
        的壳子（没有 ``__wrapped__``），拿壳子做签名检查**永远返回"接受"**
        —— 旧 IC 上照样放行，然后每秒几十次 TypeError（进程内模式下只累加
        ``failures`` 计数，症状是 IC 字段恒为默认值）。

        ``getattr(self._engine, method)`` 的存在性检查也不能省：旧 IC
        根本没有 ``apply_vision`` 这个属性。
        """
        if not self.available or self._engine is None:
            return False
        if getattr(self._engine, method, None) is None:
            return False
        cache_key = (method,) + tuple(keys)
        hit = self._caps.get(cache_key)
        if hit is None:
            state = getattr(self._engine, "state", None)
            fn = getattr(state, method, None) if state is not None else None
            if fn is None:      # 没有 state 的非常规实现 —— 退回引擎上的方法
                fn = getattr(self._engine, method, None)
            hit = _signature_accepts(fn, tuple(keys))
            self._caps[cache_key] = hit
        return hit

    def apply(self, method: str, **kwargs: Any) -> None:
        """直接调 `Engine` 上的同名方法（**同步、无队列**）。

        走 gRPC 时为了不阻塞串行事件循环，写操作必须投进后台队列；进程内
        `apply_*` 就是 RLock 下的几次属性写入（微秒级），队列反而是多余的
        延迟与丢失点。
        """
        if not self.available or self._engine is None:
            return
        fn = getattr(self._engine, method, None)
        if fn is None:
            logger.warning("进程内 InteractionCore 无此方法: %s", method)
            return
        # ⚠️ 档位字符串必须先转成 `Confidence` 枚举 —— 见 `_CONF_FIELDS`。
        #    漏了这一步会在第一次 tick 抛 AttributeError（policy 调
        #    `.at_least()`），且症状是「会话完全没有决策」，很难往这里想。
        conf_fields = _CONF_FIELDS.get(method)
        if conf_fields:
            kwargs = {
                k: (_to_confidence(v) if k in conf_fields else v)
                for k, v in kwargs.items()
            }
        try:
            fn(**kwargs)
        except Exception as exc:  # noqa: BLE001
            # ⚠️ **持续失败必须看得见**：早先这类异常被吞成 debug，于是
            #    「track_id 传 int 而 proto 要 string」这种错误完全不可见，
            #    表现为 IC 侧字段恒为默认值、决策永远不变，排查了很久。
            n = self.failures.get(method, 0) + 1
            if len(self.failures) < _MAX_FAIL_METHODS:
                self.failures[method] = n
            if n == 1 or n % 100 == 0:
                logger.warning(
                    "进程内 InteractionCore %s 失败（第 %d 次）: %s: %s"
                    "  ← 持续失败会让 IC 状态停在默认值、决策永远不变",
                    method, n, type(exc).__name__, exc)

    def apply_agent(self, status: Optional[str] = None,
                    session_end_pending: Optional[bool] = None) -> None:
        """智脑薄投影 —— Agent 经 `POST /v1/ic/apply_agent` 写进来。

        `None` = 不改该字段（与 proto 的 UNSET 语义一致）；非法 status 归一为
        不改 —— 直接塞未知字符串会让 codec 抛 `ValueError`，把整次写入打掉。

        ⚠️⚠️ **必须映射到 `AgentStatus` 枚举，不能只把字符串大写。**

        IC 里两个枚举的**大小写约定是相反的**（实测）：

            Confidence   值是大写   NONE / LOW / MEDIUM / HIGH
            AgentStatus  值是**小写** idle / busy / pending_announce

        而 ``policy.py`` 是拿 ``== AgentStatus.PENDING_ANNOUNCE`` 判的。存一个
        大写的 ``'PENDING_ANNOUNCE'`` 进去，比较是 **False**（'PENDING_ANNOUNCE'
        != 'pending_announce'）—— 不报错、不告警，**SOP 39 只是永远不生效**。
        这正是本仓库反复踩的那类「静默失效」。所以这里用枚举来归一，
        让大小写两种写法都能对，且写进去的一定是枚举成员。
        """
        # 归一逻辑与 grpc 模式**共用**一份（见 client.agent_patch_kwargs）——
        # 两个模式对同一个请求必须解释成同一个结果，否则「换模式」会静默
        # 改变语义。
        from .client import agent_patch_kwargs
        kwargs = agent_patch_kwargs(status, session_end_pending)
        if not kwargs:
            return
        self.apply("apply_agent", **kwargs)


    def ic_owns_vision_round(self) -> bool:
        return self._engine is not None

    def reset_session(self) -> bool:
        if not self.available:
            return False
        self._engine.reset_session()
        return True

    def end_session(self) -> bool:
        if not self.available:
            return False
        self._engine.end_session()
        return True

    async def tick(self, dt_ms: int = 0):
        if not self.available:
            self.bad_ticks += 1
            return None
        try:
            action = self._engine.tick(dt_ms)
        except Exception as exc:
            self.bad_ticks += 1
            self.error = f"{type(exc).__name__}: {exc}"
            self.available = False
            logger.exception("[%s] InteractionCore tick failed", self.owner_key)
            return None
        self.tick_calls += 1
        self.last_action = action.type.value
        return action

    async def snapshot(self):
        if not self.available:
            return None
        from interaction.snapshot import state_to_blocks
        return state_to_blocks(self._engine.state)

    def close(self) -> None:
        if self._decision_probe is not None:
            self._decision_probe.close()
            self._decision_probe = None
        if self._engine is not None:
            self._engine.flush_sinks()
            self._engine.close_sinks()
            self._engine = None
        self.available = False
