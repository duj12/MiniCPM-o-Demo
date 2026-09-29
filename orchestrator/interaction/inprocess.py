"""进程内 InteractionCore —— 每路会话**独占**一份 `Engine`，不走 gRPC。

## 为什么有这个

`interactioncore` 的远端服务只装配**一个** `Engine`（内含一个
`InteractionState`），所有客户端共用。两个会话同时驱动会互相覆盖
`mode` / `barge_hold_ms` / `turn`，现象是「ASR 识别完美却零决策」。
编排侧此前的应对是 `client.py` 里的 `_OWNER` 模块级单例 ——「后来者接管，
先来的挂起」，本质是把静默出错变成显式踢人。

本模块把那个问题**从根上消掉**：每路会话在自己的进程内建一份 Engine，
天然隔离，不需要 session 路由，也不需要单独跑一个 IC 服务。

## 与 `InteractionClient` 的关系

**同鸭子接口**，由 `InteractionDownstream` 按 `ORCH_IC_MODE` 二选一
（见 `downstream.py`）。两者必须保持接口一致：

    available / error / tick_calls / bad_ticks / last_action / failures
    suspended_ticks / suspended_writes        （恒 0，见下）
    connect() -> bool
    apply(method, **kwargs)
    apply_agent(status, session_end_pending)
    async tick(dt_ms) -> Action | None
    async snapshot() -> dict | None
    reset_session() / end_session() / close()

> `suspended_ticks` / `suspended_writes` 在这里**恒为 0**：那是「被别的会话
> 接管」的计数，进程内模式不存在这个状态。保留它们是因为 `session.summary()`
> 会读 —— 等 `client.py` 那套闸门删掉后可以一起删。

## ⚠️ 两个必须 off-loop 的调用

`end_session()` 与 `close()` 里有**同步 HTTP**：

  · `end_session` → `Engine.end_session()` → `ExpressionSink.stop()`
    打 `POST /v1/stop`，超时 `STOP_TIMEOUT_S = 1.0`
  · `close` → `flush_sinks()` / `close_sinks()`，每个各 join worker 线程
    （各 2s 超时）

走 gRPC 时它们跑在**服务端线程**里，编排侧感觉不到；进程内会**阻塞事件循环**。
所以调用方（`Downstream.on_session_end`）必须用 `asyncio.to_thread` 包起来。
"""
from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

#: 与方法名无关的失败计数上限，避免 dict 无限增长
_MAX_FAIL_METHODS = 64

#: 每个 `apply_*` 里**取 Confidence 枚举**的参数名。
#:
#: ⚠️⚠️ **必须做这个转换，否则 IC 会在第一次 tick 就崩。**

#: 走 gRPC 时，`grpc_codec.conf_field` 已经把 proto 枚举转成了
#: `state.Confidence` **枚举成员**；进程内路径是直接调 `Engine.apply_*`，
#: 传进去的是编排侧 `_conf()` 给的**裸字符串** `'HIGH'`。
#:
#: 危险之处在于它**看起来能用**：`Confidence` 是 `str` 枚举，值是**大写**
#: （`'HIGH' == Confidence.HIGH` 为真），所以 `==` 比较全对、状态也存得进去。
#: 但 `policy.py` 有一批判据是**调方法**的 ——
#: `person.face_present_confidence.at_least(Confidence.MEDIUM)`
#: （policy.py:138/140/233/352）—— 裸字符串没有 `at_least`，于是抛
#: `AttributeError: 'str' object has no attribute 'at_least'`，
#: `tick()` 直接失败 ⇒ `_mark_dead` ⇒ **整个会话降级为无决策**。
#:
#: 单元测试没暴露它是因为只 `apply_asr` 就 tick（那条路上 `at_least` 恰好
#: 没被走到）；真会话会 `apply_face`/`apply_vad`/`apply_lip`，一 tick 就炸。
#: 实测就是这么发现的 —— **别只靠单元测试验这条路径**。
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
    """一路会话一个实例 —— 本实例**独占**一个 `interaction.runtime.Engine`。"""

    def __init__(self, session_id: str = "", *,
                 expression_url: str = "",
                 agent_url: str = "",
                 agent_timeout: float = 5.0,
                 expression_ca: Optional[str] = None,
                 expression_verify: bool = True,
                 callback_ic: Optional[str] = None,
                 thresholds: Any = None) -> None:
        self.owner_key = session_id or ""
        self.target = "inprocess"
        self.expression_url = expression_url
        self.agent_url = agent_url
        self._agent_timeout = float(agent_timeout)
        self._expression_ca = expression_ca
        self._expression_verify = bool(expression_verify)
        self._callback_ic = callback_ic
        self._thresholds = thresholds

        self._engine: Any = None
        self.available = False
        self.error: Optional[str] = None

        # ---- 诊断字段（与 InteractionClient 同名同义）----
        #: `tick` 真调到 IC 并拿到返回的次数
        self.tick_calls = 0
        #: 没调到的次数（未连接 / 没装包）
        self.bad_ticks = 0
        self.last_action = ""
        #: 各方法写失败计数（按方法名）
        self.failures: dict = {}
        #: 进程内模式**不存在**「被接管」，恒 0（见模块文档）
        self.suspended = False
        self.suspended_ticks = 0
        self.suspended_writes = 0
        self.dropped = 0

    # ------------------------------------------------------------------ #

    @property
    def engine(self) -> Any:
        """本会话的 Engine（测试/诊断用）。未连接时为 None。"""
        return self._engine

    def connect(self) -> bool:
        """建本会话的 Engine。失败返回 False（**不抛** —— 调用方降级到 OmniLLM）。

        懒 import：`interaction` 包没装时**只该让本会话降级**，不该让
        orchestrator 起不来（与 `InteractionClient.connect` 的 `ImportError`
        兜底同款）。
        """
        if self._engine is not None:
            self.available = True
            return True
        try:
            from interaction.runtime import AgentSink, Engine, ExpressionSink
        except ImportError as exc:
            self.error = f"未安装 interactioncore 包（{exc}）"
            logger.warning("进程内 InteractionCore 不可用：%s", self.error)
            self.available = False
            return False

        expression_sink = None
        if self.expression_url:
            expression_sink = ExpressionSink(
                self.expression_url,
                session_id=self.owner_key or None,
                ca_file=self._expression_ca,
                verify_ssl=self._expression_verify,
            )
        else:
            logger.warning("[%s] 未配 ic_expression_url —— IC 的 GREET/UTTER "
                           "与 YIELD/END 将无处播报/停播", self.owner_key or "?")

        agent_sink = None
        if self.agent_url:
            agent_sink = AgentSink(
                self.agent_url,
                timeout=self._agent_timeout,
                session_id=self.owner_key or None,
                callback_ic=self._callback_ic,
                # ⚠️ 进程内模式下**直接开**：这里没有「共用 IC 的第三方」
                #    顾虑 —— Agent 收到的就是我们这一路会话的 Action，
                #    补上 session_id / callback_ic 正是我们要的。
                session_envelope=True,
            )
        else:
            logger.warning("[%s] 未配 agent_url —— IC 的 ANSWER/INSERT "
                           "派不出去", self.owner_key or "?")

        try:
            self._engine = Engine(
                expression_sink=expression_sink,
                agent_sink=agent_sink,
                thresholds=self._thresholds,
            )
        except Exception as exc:  # noqa: BLE001
            self.error = f"{type(exc).__name__}: {exc}"
            logger.error("[%s] 进程内 InteractionCore 装配失败：%s",
                         self.owner_key or "?", self.error)
            self.available = False
            return False

        self.available = True
        self.error = None
        logger.info(
            "进程内 InteractionCore 已就绪（会话 %s expression=%s agent=%s "
            "callback_ic=%s）",
            self.owner_key or "?", self.expression_url or "-",
            self.agent_url or "-", self._callback_ic or "-")
        return True

    # ------------------------------------------------------------------ #
    #  写
    # ------------------------------------------------------------------ #

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

    # ------------------------------------------------------------------ #
    #  生命周期
    # ------------------------------------------------------------------ #

    def reset_session(self) -> bool:
        """新会话开始：清全部交互状态（**不通知** agent / TTS）。

        进程内模式下这是新 Engine，本来就是干净的 —— 保留它是为了与 grpc
        模式的启动序列一致（显式优于隐式），并作为双保险。
        """
        if not self.available or self._engine is None:
            return False
        try:
            self._engine.reset_session()
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("进程内 InteractionCore 重置失败：%s", exc)
            return False

    def end_session(self) -> bool:
        """用户主动结束：通知 TTS 停播、Agent 收工，再清状态。

        ⚠️ **同步 HTTP**（`ExpressionSink.stop` 最长 1s）—— 调用方必须
        off-loop（见模块文档）。
        """
        if not self.available or self._engine is None:
            return False
        try:
            self._engine.end_session()
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("进程内 InteractionCore 收尾失败：%s", exc)
            return False

    # ------------------------------------------------------------------ #
    #  读
    # ------------------------------------------------------------------ #

    async def tick(self, dt_ms: int = 0):
        """调一次 `Engine.tick()`，返回 `Action`；失败返回 `None`。

        **不需要 `to_thread`** —— policy 是纯 Python 运算（微秒级），没有 I/O。
        保持 `async` 只是为了与 `InteractionClient` 同接口。

        ⚠️ `tick()` **不是只读的**：它会推进状态机（`advance_*` → `decide()`
        → `apply_action()`）。进程内模式下这没有并发问题 —— 每路会话的
        Engine 是自己的。
        """
        if not self.available or self._engine is None:
            self.bad_ticks += 1
            return None
        try:
            action = self._engine.tick(dt_ms)
        except Exception as exc:  # noqa: BLE001
            self.bad_ticks += 1
            self._mark_dead(f"tick 失败: {type(exc).__name__}: {exc}")
            return None
        self.tick_calls += 1
        self.last_action = getattr(getattr(action, "type", None), "value",
                                   str(getattr(action, "type", "?")))
        return action

    async def snapshot(self) -> Optional[dict]:
        """读整份状态（调试/诊断用）。"""
        if not self.available or self._engine is None:
            return None
        try:
            from interaction.snapshot import state_to_blocks
            return state_to_blocks(self._engine.state)
        except Exception as exc:  # noqa: BLE001
            self._mark_dead(f"snapshot 失败: {type(exc).__name__}: {exc}")
            return None

    # ------------------------------------------------------------------ #

    def _mark_dead(self, why: str) -> None:
        """连接失效 —— 置位并告警，调用方据此降级。"""
        if not self.available:
            return
        self.available = False
        self.error = why
        logger.warning("进程内 InteractionCore 失效（会话 %s）—— %s",
                       self.owner_key or "?", why)

    def close(self) -> None:
        """收尾：把两个 sink 的队列排空再关线程。

        ⚠️ `flush` 必须在 `close` **之前** —— `close` 会 join worker 线程，
        队列里还没发出的 `on_end` / speak 会被停机哨兵直接丢掉
        （这正是「Agent 收不到会话结束」那类故障的成因）。

        ⚠️ **同步、可能阻塞数秒** —— 调用方必须 off-loop。
        """
        engine, self._engine = self._engine, None
        if engine is None:
            self.available = False
            return
        for name in ("flush_sinks", "close_sinks"):
            fn = getattr(engine, name, None)
            if fn is None:
                # 老版本 interactioncore 没有这两个方法 —— 退回逐个 sink 处理
                continue
            try:
                fn(2.0)
            except Exception as exc:  # noqa: BLE001
                logger.debug("%s 异常: %s", name, exc)
        if not hasattr(engine, "close_sinks"):
            for sink in (getattr(engine, "expression_sink", None),
                         getattr(engine, "agent_sink", None)):
                if sink is None:
                    continue
                for name in ("flush", "close"):
                    fn = getattr(sink, name, None)
                    if fn is not None:
                        try:
                            fn(2.0)
                        except Exception as exc:  # noqa: BLE001
                            logger.debug("sink.%s 异常: %s", name, exc)
        self.available = False
        logger.info("进程内 InteractionCore 已关闭（会话 %s）", self.owner_key or "?")
