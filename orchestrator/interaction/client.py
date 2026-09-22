"""InteractionCore 的 gRPC 封装。

对应仓库 ``interactioncore``（``pip install -e .``），服务默认 50051。

**为什么写操作走后台队列，不直接调**：

    ``run_downstream`` 是**串行 await** 的循环（``session.py``）——
    ``on_event`` 里一旦阻塞（gRPC 往返 1~10ms，慢时更久），下游就被占住，
    后续事件全堵在队列里。这与之前流式 TTS 踩过的死锁是同一类问题。
    所以 ``apply_*`` 一律**投进队列立即返回**，由专用线程按序消费。

    ``tick()`` 是**读**操作（要拿返回值），用 ``asyncio.to_thread`` 包，
    让出事件循环。

⚠️ **所有失败只告警不抛** —— IC 挂了不该拖垮会话。降级由
``InteractionDownstream`` 负责（回退 OmniLLM 回复）。
"""
from __future__ import annotations

import asyncio
import logging
import queue
import threading
from typing import Any, Optional

logger = logging.getLogger(__name__)

#: 写队列上限。满了就丢最旧的 —— 状态是**最新值有意义**的（幂等覆盖），
#: 积压旧值重放反而会让 IC 看到过期状态。
_QUEUE_MAX = 256

#: 调用的默认超时（秒）。IC 是本地服务，正常 1ms 级。
_CALL_TIMEOUT = 2.0


#: 当前**唯一**被允许驱动 IC 的客户端实例（模块级单例）。
#:
#: ⚠️ IC 的 ``InteractionState`` 是服务端**全局一份**，没有 session 概念 ——
#: 所以「谁在驱动它」必须**全局唯一**。两个会话同时驱动同一个 IC 时：
#:   · 两者的 ``tick()`` 一起推进同一个状态机 → ``dt`` 累加翻倍
#:   · 各自的 ``apply_action`` 互相覆盖 ``mode`` / ``barge_hold_ms`` 等
#:   · 一方写进去的 ``turn=HIGH``，可能被另一方的 ``apply_action`` 先冲掉
#: 现象是「ASR 识别完美、也送进 IC 了，却没有任何回复」—— 而 IC 自己的
#: 日志里**明明出过 ANSWER**（被另一路的 tick 领走了）。实测踩过。
_OWNER: Optional["InteractionClient"] = None


class InteractionClient:
    """一路会话对应一个实例（IC 的状态是**进程内单例**，非按会话隔离）。

    ⚠️ IC 的 ``InteractionState`` 是服务端**全局一份**，没有 session 概念。
    所以本类实现了「**单一驱动者**」约束：同一时刻只允许一个实例真正读写
    IC，其余实例被**挂起**（``suspended``）。

    为什么要挂起而不是直接不管：会话**收尾很慢**（等 ASR/Omni drain，实测
    10~20 秒），而新会话立刻就起来了 —— 这段重叠**必然发生**。若不挂起旧
    实例，两个会话会同时 tick 同一个 IC，互相踩状态（见 ``_OWNER`` 的注释）。

    多路真并发需要 IC 侧支持多实例（不在本次范围）。
    """

    def __init__(self, target: str = "localhost:50051",
                 owner_key: str = "") -> None:
        self.target = target
        #: 谁在用这个实例（会话 id）—— 只为日志可读
        self.owner_key = owner_key or "?"
        self._client = None            # InteractionStateClient
        self._q: "queue.Queue" = queue.Queue(maxsize=_QUEUE_MAX)
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self.available = False         # 连上了才 True
        self.error: Optional[str] = None
        self.dropped = 0               # 队列满丢掉的写次数
        #: 被别的会话接管了 ⇒ 本实例**停止读写 IC**（见 ``activate``）。
        self.suspended = False
        #: 挂起后丢弃的 tick 次数（诊断用）
        self.suspended_ticks = 0
        #: 各方法写失败计数（按方法名）—— 持续失败必须看得见，
        #: 否则 IC 状态静静停在默认值、决策永远不变（踩过）
        self.failures: dict = {}
        self._fail_streak = 0

    def activate(self) -> None:
        """声明本实例为 IC 的**唯一驱动者**，把上一个挂起。

        在 ``connect()`` 成功后调用 —— 时机天然正确：新会话建立时接管，
        旧会话（正在收尾）自动让位。

        ⚠️ 「后来者接管」策略：真正多用户并发时，后开的会话会把先开的
        **踢下线**（它的 IC 决策全停）。这是 IC 单例的固有限制，不是本闸门
        引入的 —— 闸门只是让它**从静默出错变成显式接管**。
        """
        global _OWNER
        prev, _OWNER = _OWNER, self
        self.suspended = False
        if prev is not None and prev is not self:
            prev.suspended = True
            logger.warning(
                "IC 被会话 %s 接管 —— 会话 %s 暂停驱动 IC"
                "（它若还在收尾，其 IC 决策将不再生效；这是单例 IC 的固有限制）",
                self.owner_key, prev.owner_key)

    # ------------------------------------------------------------------ #

    def connect(self) -> bool:
        """建连接并**探活**（连不上返回 False，不抛）。

        探活很重要：grpc 的 ``insecure_channel`` 是**懒连接**，构造时不报错，
        真正调用才知道连不上。所以这里主动发一个轻量的 ``GetSnapshot`` 确认。
        """
        try:
            from interaction import InteractionStateClient  # type: ignore
        except ImportError as exc:
            self.error = f"未安装 interactioncore 包（{exc}）"
            logger.warning("InteractionCore 不可用：%s", self.error)
            return False
        try:
            self._client = InteractionStateClient(self.target)
            # 懒连接：必须真调一次才知道通不通
            self._client.get_snapshot()
        except Exception as exc:  # noqa: BLE001
            self.error = f"{type(exc).__name__}: {exc}"
            logger.warning("InteractionCore 连接失败（%s）：%s",
                           self.target, self.error)
            self._client = None
            return False

        self.available = True
        self.error = None
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="ic-apply", daemon=True)
        self._thread.start()
        # 连上就声明接管 —— 把正在收尾的旧会话挂起，避免两个会话同时驱动 IC
        self.activate()
        logger.info("InteractionCore 已连接: %s（会话 %s）",
                    self.target, self.owner_key)
        return True

    # ------------------------------------------------------------------ #
    #  写：非阻塞投递
    # ------------------------------------------------------------------ #

    def apply(self, method: str, **kwargs: Any) -> None:
        """把一次 ``apply_*`` 投进队列（**立即返回**）。

        ``method`` 是 IC 客户端的方法名（``apply_face`` / ``apply_asr`` …）。

        ⚠️ **被别的会话接管后（``suspended``）直接丢弃** —— 否则本会话
        （正在收尾）的状态会覆盖掉新会话写进去的，后者拿不到自己的转写。
        """
        if not self.available or self.suspended:
            return
        try:
            self._q.put_nowait((method, kwargs))
        except queue.Full:
            # 丢最旧的：状态是"最新值有意义"，重放旧值没意义
            try:
                self._q.get_nowait()
                self._q.put_nowait((method, kwargs))
            except queue.Empty:
                pass
            self.dropped += 1
            if self.dropped % 100 == 1:
                logger.debug("InteractionCore 写队列满，已丢 %d 次", self.dropped)

    def _run(self) -> None:
        """专用线程：按序消费写队列。"""
        while not self._stop.is_set():
            try:
                item = self._q.get(timeout=0.2)
            except queue.Empty:
                continue
            if item is None:
                break
            method, kwargs = item
            fn = getattr(self._client, method, None)
            if fn is None:
                logger.warning("InteractionCore 无此方法: %s", method)
                continue
            try:
                fn(**kwargs)
                self._fail_streak = 0
            except Exception as exc:  # noqa: BLE001
                # 写失败不抛、不退出线程（IC 挂了不该拖垮会话），但要**可见**。
                # ⚠️ 早先这里打的是 debug —— 于是"参数类型不对"这种持续失败
                #    完全看不见，表现为 IC 侧字段恒为默认值、决策永远不变，
                #    排查了很久（实测：track_id 传 int 而 proto 要 string）。
                #    改成：按方法名计数，**首次必 log.warning**，之后每 100 次
                #    提醒一次（避免每 tick 刷屏）。
                self._fail_streak += 1
                self.failures[method] = self.failures.get(method, 0) + 1
                n = self.failures[method]
                if n == 1 or n % 100 == 0:
                    logger.warning(
                        "InteractionCore %s 失败（第 %d 次）: %s: %s"
                        "  ← 持续失败会让 IC 状态停在默认值、决策永远不变，"
                        "请检查参数类型/取值", method, n,
                        type(exc).__name__, exc)

    # ------------------------------------------------------------------ #
    #  读：tick 取决策
    # ------------------------------------------------------------------ #

    async def tick(self, dt_ms: int = 0):
        """调一次 ``Tick``，返回 ``Action``；失败返回 ``None``。

        用 ``to_thread`` 包 —— 这是**读**操作要拿返回值，不能走队列。

        ⚠️⚠️ **``tick()`` 不是只读的**：IC 侧它会推进状态机
        （``advance_*`` → ``decide()`` → ``apply_action()``，后者会改
        ``mode`` / ``barge_hold_ms``）。所以**两个会话同时 tick 就是两个
        驱动者在踩同一个状态机** —— 这正是「ASR 完美却无回复」的根因。
        被接管后必须**直接返回 None**，一次都不能发（``_dispatch(None)``
        已经是安全的空操作）。
        """
        if not self.available or self._client is None or self.suspended:
            if self.suspended:
                self.suspended_ticks += 1
            return None
        try:
            return await asyncio.to_thread(self._client.tick, dt_ms)
        except Exception as exc:  # noqa: BLE001
            self._mark_dead(f"tick 失败: {type(exc).__name__}: {exc}")
            return None

    async def snapshot(self) -> Optional[dict]:
        """读整份状态（调试/诊断用）。

        被接管后返回 ``None`` —— 此时 IC 里装的是**别的会话**的状态，
        读出来只会误导诊断。
        """
        if not self.available or self._client is None or self.suspended:
            return None
        try:
            return await asyncio.to_thread(self._client.get_snapshot)
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
        logger.warning("InteractionCore 连接失效（%s）—— %s", self.target, why)

    def close(self) -> None:
        global _OWNER
        self._stop.set()
        try:
            self._q.put_nowait(None)     # 唤醒阻塞的 get
        except queue.Full:
            pass
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._client is not None:
            try:
                self._client.close()
            except Exception:  # noqa: BLE001
                pass
            self._client = None
        self.available = False
        # 释放所有权（若还是自己）—— 但**不自动移交给别人**：
        # 谁是下一个驱动者由它的 connect() → activate() 决定，这里只清空。
        if _OWNER is self:
            _OWNER = None
        logger.info("InteractionCore 已关闭（丢写 %d 次%s）", self.dropped,
                    f"，被接管后丢弃 tick {self.suspended_ticks} 次"
                    if self.suspended_ticks else "")
