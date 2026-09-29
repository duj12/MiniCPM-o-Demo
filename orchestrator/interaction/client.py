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


def normalize_agent_status(raw):
    """把大小写不限的 status 字串归一到 `AgentStatus` 枚举；非法返回 None。

    ⚠️ **必须映射到枚举，不能只把字符串大写。** IC 里两个枚举的大小写约定
    是**相反**的（实测）：

        Confidence   值是大写   NONE / LOW / MEDIUM / HIGH
        AgentStatus  值是**小写** idle / busy / pending_announce

    而 `policy.py` 是拿 `== AgentStatus.PENDING_ANNOUNCE` 判的。往状态里存
    一个大写字串 `'PENDING_ANNOUNCE'`，比较就是 **False**
    （`'PENDING_ANNOUNCE' != 'pending_announce'`）—— 不报错、不告警，
    **SOP 39 只是永远不生效**。另一个方向更直接：`AgentStatus('BUSY')`
    会抛 `ValueError`，把整次写入打掉。

    所以统一走枚举反查：大小写两种写法都能对，且写进去的一定是枚举成员。
    """
    try:
        from interaction.state import AgentStatus
    except ImportError:      # interaction 包没装 —— 调用方会降级
        return None
    key = str(raw).strip().upper()
    for member in AgentStatus:
        if member.name.upper() == key:
            return member
    return None


def agent_patch_kwargs(status=None, session_end_pending=None) -> dict:
    """把 `apply_agent` 的两个入参归一成 `Engine.apply_agent` 的 kwargs。

    语义与 proto 的 `ApplyAgent` 逐字对齐：**省略（None）= 不改该字段**。
    非法 status 归一为「不改」而不是抛 —— 一个坏字段不该把整次写入打掉。
    """
    import logging

    kwargs = {}
    if status:
        norm = normalize_agent_status(status)
        if norm is None:
            logging.getLogger(__name__).warning(
                "apply_agent 收到未知 status=%r —— 已忽略该字段"
                "（合法值 IDLE/BUSY/PENDING_ANNOUNCE，大小写不限）", status)
        else:
            kwargs["status"] = norm
    if session_end_pending is not None:
        kwargs["session_end_pending"] = bool(session_end_pending)
    return kwargs


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
        #: 挂起后**消费线程**丢弃的写入次数 —— 这些是接管前入队、接管后
        #: 才被取出的"过期条目"（见 `_run` 里的第二道闸门）
        self.suspended_writes = 0
        #: 诊断计数 —— `tick` 是**唯一**能出决策的路径。若 `tick_calls` 在涨
        #: 而 `IC → X` 一条都没有，说明 IC 一直返回 HOLD/LISTEN/WAIT
        #: （看 `last_action`）；若 `bad_ticks` 在涨，说明根本没调到。
        #: 这两个是**完全不同**的故障，不打出这个值就分不清（实测卡过）。
        self.tick_calls = 0        # 真调到 IC 并拿到返回的次数
        self.bad_ticks = 0         # 没调（未连接 / 未就绪）的次数
        self.last_action = ""      # 最近一次 IC 返回的 ActionType
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

    def reset_session(self) -> bool:
        """**新会话开始**：把 IC 全部交互状态重置。**同步调用**。

        IC 的 ``Engine`` / ``InteractionState`` 是**进程级全局一份**
        （``serve()`` 里只建一次），状态**跨会话保留** —— 不清就会：

          · ``mode`` 停在 ``LISTENING`` → **永远不迎宾**
            （GREET 的前置条件是 ``mode == SessionMode.IDLE``）
          · ``speech.user_speaking`` 残留 → 新会话刚开口就被判
            **"用户抢话" (YIELD)** → ``ic_stop`` → **播报被打断**
            （实测：刷新页面后新会话的播报立刻被停）

        ``reset_all_state`` 是**整体替换** ``session`` / ``person`` /
        ``speech`` / ``agent``（不是逐字段清），所以不会漏。

        ## 与 `end_session` 的分工（**别混用**）

            reset_session  新会话开始（本方法）—— **不通知** agent / TTS
            end_session    用户主动结束          —— **通知** agent / TTS

        ## 为什么是同步的（不走 ``apply`` 的异步队列）

        调用点在 ``session.start`` **之前**，必须**清完再开始**。走异步队列
        会和紧随其后的 ``apply_face`` / ``apply_asr`` 乱序，清掉的可能是
        本会话刚写进去的状态。

        ## 失败不阻断

        IC 没这个接口（旧版本）时返回 False 并告警 —— 会话照常跑，
        只是可能缺一次迎宾。不能因为清理失败就不让用户开会话。
        """
        if not self.available or self._client is None:
            return False
        if self.suspended:
            # 与 `end_session` 同理：被接管后 IC 归别人驱动，不该由我们清。
            # 当前调用点（`on_session_start`）不会走到这里 —— 那时本会话
            # 刚 `activate()`，`suspended` 必为 False。留这道闸门是防御：
            # 将来若有人从别处调，不至于跨会话清掉别人的状态。
            logger.info("跳过 IC 状态重置（%s）—— 本会话已被别的会话接管",
                        self.owner_key)
            return False
        try:
            self._client.reset_session()
            logger.info("InteractionCore 会话状态已重置（%s）—— "
                        "避免上一个会话的 mode/greet/说话的残留影响本次",
                        self.target)
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("重置 InteractionCore 会话状态失败（%s）：%s —— "
                           "若 IC 是旧版本可忽略；否则本次可能不迎宾、"
                           "或播报刚开口就被误判抢话",
                           self.target, exc)
            return False

    def end_session(self) -> bool:
        """**用户主动结束**：让 IC 收尾（**通知 agent / TTS**，再清状态）。

        与 `reset_session` 的关键区别是**它会对外通知**：

            IC.end_session() → AgentSink.on_end()     → 通知 Agent
                            → ExpressionSink.stop()   → `POST /v1/stop` 停播
                            → reset_all_state()

        所以**编排侧不需要再自己通知 Agent / 停播** —— 那些都归 IC 管
        （编排服务的定位是"只做编排"，播报/打断/通知一律由 IC 与 Agent
        自己发起）。

        ⚠️ 调用时机：**客户端断开 / 点停止**（`on_session_end`）。
        ⚠️ **不要**在 IC 自己判 END 时调 —— 那是 IC 内部的行为，
        它自己会走 `clear_session_on_end`。

        ## ⚠️⚠️ `suspended` 时**绝不能调** —— 会跨会话误杀

        `end_session()` 是**进程级全局**操作（清 IC 全部状态 + 全局停播 +
        通知 Agent）。而**旧会话收尾很慢**（等 ASR/Omni drain，10~20 秒），
        期间新会话往往已经接管 IC 并在正常对话了。

        实测踩过（106，15:17）：

            15:17:09  IC 被会话 021a66846f54 接管 —— 1eb85ade38db 暂停驱动
            15:17:36  两个旧会话客户端断开（收尾转入后台）
            15:17:47  IC 被新会话 55ca1168ca7f 接管
            15:17:51  新会话播报 GREET「您好，金涵，欢迎光临。」
            15:17:57  **被挂起的旧会话调了 end_session** → 清空 IC 状态
            15:17:59  新会话重新 GREET（状态没了）→ 打断 91b24373（superseded）
            15:17:59  ic_stop → **新会话的播报被打断** ❌

        `apply()` / `tick()` 早就有这个闸门（见各自的 `suspended` 判断），
        `end_session` 当时漏了 —— 收尾路径不在"驱动 IC"的直觉范围内，
        但 `end_session` 恰恰是**动作最重**的那个。

        失败不阻断收尾（IC 挂了不该让会话关不掉）。
        """
        if not self.available or self._client is None:
            return False
        if self.suspended:
            # 本会话已被接管 —— IC 现在归别人驱动，它的状态不属于我们，
            # 我们没有资格去清空/停播（见上面 15:17 的实测）。
            logger.info("跳过 IC 收尾（%s）—— 本会话已被别的会话接管，"
                        "IC 状态归当前驱动者所有", self.owner_key)
            return False
        try:
            self._client.end_session()
            logger.info("InteractionCore 已收尾（%s）—— IC 会通知 Agent "
                        "并停播，编排侧不再单独做这些", self.target)
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("InteractionCore 收尾调用失败（%s）：%s —— "
                           "会话照常关闭，但 Agent 可能收不到结束通知",
                           self.target, exc)
            return False

    def apply_agent(self, status=None, session_end_pending=None) -> None:
        """智脑薄投影 —— Agent 写 `agent.status` / `session_end_pending`。

        走 `grpc` 模式时 IC 在**远端**，这条写入要经过 gRPC 的
        ``ApplyAgent``。与进程内模式（``InProcessICClient.apply_agent``）
        **同名同语义** —— 于是 `main.py` 的 ``/v1/ic/apply_agent`` 端点
        不必按模式分支，两种模式一条路。

        ⚠️ **同步调用**（gRPC 往返）—— 调用方必须 off-loop，别在事件循环里调。
        """
        kwargs = agent_patch_kwargs(status, session_end_pending)
        if not kwargs or self._client is None:
            return
        try:
            self._client.apply_agent(**kwargs)
        except Exception as exc:  # noqa: BLE001
            logger.warning("apply_agent 失败（%s）：%s —— 已忽略",
                           self.target, exc)

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
            # ⚠️⚠️ **消费时再查一次 `suspended`** —— 这是闸门的第二道，必需。
            #
            # `apply()` 里的检查只管「入队那一刻」。而队列是**异步消费**的：
            # 旧会话在被接管**之前**投进去的条目还压在队列里，接管之后才被
            # 消费线程取出 —— 若这里不查，那些**过期条目照发不误**。
            #
            # 实测踩过：两个被挂起的旧会话仍在往 IC 写空转写帧，把新会话
            # 刚写进去的 `text='你好呀，你是谁呀？'` 冲掉 → 新会话拿不到
            # 自己的转写 → 永远出不了 ANSWER。现象是「三个会话全部零决策」。
            if self.suspended:
                self.suspended_writes += 1
                continue
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
        if self.suspended:
            self.suspended_ticks += 1
            return None
        if not self.available or self._client is None:
            self.bad_ticks += 1
            return None
        try:
            action = await asyncio.to_thread(self._client.tick, dt_ms)
        except Exception as exc:  # noqa: BLE001
            self.bad_ticks += 1
            self._mark_dead(f"tick 失败: {type(exc).__name__}: {exc}")
            return None
        self.tick_calls += 1
        # ⚠️ 诊断：`tick` 是**唯一**能出决策的路径。若 `tick_calls` 在涨、
        #    但 `IC → X` 一条都没有，说明 IC 那边一直返回 HOLD/LISTEN/WAIT
        #    （见 ``last_action``）—— 这与「tick 没调到」是**完全不同**的
        #    两个故障，不打出这个值就分不清（实测排查时卡过）。
        self.last_action = getattr(getattr(action, "type", None), "value",
                                   str(getattr(action, "type", "?")))
        return action

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
