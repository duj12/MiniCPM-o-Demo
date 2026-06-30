"""Worker 连接池 + FIFO 队列调度器

管理多个 Worker 的连接、状态追踪、请求路由、FIFO 排队。

核心职责：
1. Worker 发现和健康检查
2. 统一 FIFO 请求排队（容量 1000，先到先服务）
3. 队列位置追踪和等待时间估算（ETA）
4. 请求取消和清理

路由策略：
- Chat（无状态）：任意空闲 Worker
- Half-Duplex（独占）：任意空闲 Worker，会话期间独占
- Duplex（独占）：任意空闲 Worker

排队机制：
- 统一 FIFO 队列，所有请求类型共享
- 每个等待者持有 asyncio.Future，Worker 空闲时 resolve 队头
- 单一调度点 _dispatch_next()，消除竞争
"""

import asyncio
import heapq
import logging
import uuid
from typing import Dict, List, Optional, Tuple, Any, Callable, Awaitable
from collections import OrderedDict
from datetime import datetime
from dataclasses import dataclass, field

import httpx

from .models import (
    GatewayWorkerStatus,
    WorkerInfo,
    QueueTicket,
    QueueTicketSummary,
    QueueStatus,
    RunningTaskInfo,
    EtaConfig,
    EtaStatus,
)
from runtime.protocol import DEFAULT_WORKER_CAPABILITIES, capability_for_request

logger = logging.getLogger("gateway.worker_pool")

HEALTH_CHECK_INTERVAL = 10.0


# ============ Worker 连接 ============

@dataclass
class WorkerConnection:
    """Worker 连接（Gateway 侧维护）

    支持多路并发：一个 Worker 可同时处理 max_concurrency 个 session。
    active_tickets 跟踪当前活跃的 ticket 列表。
    current_ticket_id 保持向后兼容（指向 active_tickets[0]）。
    """
    worker_id: str
    host: str
    port: int
    gpu_id: int
    status: GatewayWorkerStatus = GatewayWorkerStatus.OFFLINE
    max_concurrency: int = 4
    current_ticket_id: Optional[str] = None
    active_tickets: List[str] = field(default_factory=list)
    total_requests: int = 0
    avg_inference_time_ms: float = 0.0
    last_heartbeat: Optional[datetime] = None
    current_request_type: Optional[str] = None
    task_started_at: Optional[datetime] = None
    capabilities: Optional[List[str]] = None
    _gateway_dispatched: bool = False

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def is_idle(self) -> bool:
        return not self.active_tickets

    @property
    def is_busy(self) -> bool:
        return len(self.active_tickets) >= self.max_concurrency

    @property
    def has_capacity(self) -> bool:
        """Worker 还能接受更多并发 session"""
        return len(self.active_tickets) < self.max_concurrency

    def supports(self, request_type: str) -> bool:
        required = capability_for_request(request_type)
        return required in (self.capabilities or DEFAULT_WORKER_CAPABILITIES)

    def to_info(self) -> WorkerInfo:
        return WorkerInfo(
            worker_id=self.worker_id,
            host=self.host,
            port=self.port,
            gpu_id=self.gpu_id,
            status=self.status,
            current_ticket_id=self.current_ticket_id,
            total_requests=self.total_requests,
            avg_inference_time_ms=self.avg_inference_time_ms,
            last_heartbeat=self.last_heartbeat,
            current_request_type=self.current_request_type,
            task_started_at=self.task_started_at,
            capabilities=list(self.capabilities or DEFAULT_WORKER_CAPABILITIES),
        )

    def mark_busy(self, status: GatewayWorkerStatus, request_type: str,
                  ticket_id: Optional[str] = None) -> None:
        """标记 Worker 的一个并发槽位为忙碌

        注意：enqueue 和 caller 会各自调用一次 mark_busy（同一个 ticket_id），
        因此需要去重防止 active_tickets 中出现重复 ticket。
        """
        self.status = status
        self.current_request_type = request_type
        if not self.active_tickets:
            self.task_started_at = datetime.now()
        if ticket_id and ticket_id not in self.active_tickets:
            self.active_tickets.append(ticket_id)
        self.current_ticket_id = self.active_tickets[0] if self.active_tickets else None
        self._gateway_dispatched = True

    def release_ticket(self, ticket_id: str) -> None:
        """释放指定的 ticket，仅移除该 ticket 而非整个 Worker"""
        if ticket_id in self.active_tickets:
            self.active_tickets.remove(ticket_id)
        self.current_ticket_id = self.active_tickets[0] if self.active_tickets else None
        if not self.active_tickets:
            self.mark_idle()

    def update_duplex_status(self, status: GatewayWorkerStatus) -> None:
        """更新 Duplex 会话的状态（pause/resume），不重置 task_started_at"""
        self.status = status

    def mark_idle(self) -> None:
        """标记 Worker 完全空闲（所有 session 已结束）"""
        self.status = GatewayWorkerStatus.IDLE
        self.current_request_type = None
        self.task_started_at = None
        self.current_ticket_id = None
        self.active_tickets.clear()
        self._gateway_dispatched = False


# ============ 队列条目 ============

@dataclass
class QueueEntry:
    """队列中的一个等待条目"""
    ticket: QueueTicket
    future: "asyncio.Future[Optional[WorkerConnection]]"
    history_hash: Optional[str] = None


# ============ EMA 追踪器 ============

class EtaTracker:
    """ETA 估算追踪器

    维护 Admin 配置的基准值和运行时 EMA 动态值。
    """

    def __init__(self, eta_config: EtaConfig, ema_alpha: float = 0.3,
                 ema_min_samples: int = 3) -> None:
        self.config = eta_config
        self.ema_alpha = ema_alpha
        self.ema_min_samples = ema_min_samples

        # EMA 状态
        self._ema: Dict[str, float] = {}
        self._samples: Dict[str, int] = {"chat": 0, "half_duplex_audio": 0, "audio_duplex": 0, "omni_duplex": 0}

    def get_eta(self, request_type: str) -> float:
        """获取指定类型的预估耗时

        有足够 EMA 样本时用 EMA，否则用 Admin 基准值。
        """
        if self._samples.get(request_type, 0) >= self.ema_min_samples:
            return self._ema.get(request_type, self._get_base(request_type))
        return self._get_base(request_type)

    def record_duration(self, request_type: str, duration_s: float) -> None:
        """记录一次请求的实际耗时，更新 EMA"""
        count = self._samples.get(request_type, 0)
        if count == 0:
            self._ema[request_type] = duration_s
        else:
            old = self._ema.get(request_type, duration_s)
            self._ema[request_type] = self.ema_alpha * duration_s + (1 - self.ema_alpha) * old
        self._samples[request_type] = count + 1

    def update_config(self, new_config: EtaConfig) -> None:
        """更新 Admin 配置的基准值（含可选 ema_alpha）"""
        if new_config.ema_alpha is not None:
            self.ema_alpha = new_config.ema_alpha
        self.config = new_config

    def get_status(self) -> EtaStatus:
        """返回 ETA 状态（含 EMA 值）"""
        return EtaStatus(
            config=self.config,
            ema_alpha=self.ema_alpha,
            ema_chat_s=self._ema.get("chat"),
            ema_half_duplex_s=self._ema.get("half_duplex_audio"),
            ema_audio_duplex_s=self._ema.get("audio_duplex"),
            ema_omni_duplex_s=self._ema.get("omni_duplex"),
            ema_chat_samples=self._samples.get("chat", 0),
            ema_half_duplex_samples=self._samples.get("half_duplex_audio", 0),
            ema_audio_duplex_samples=self._samples.get("audio_duplex", 0),
            ema_omni_duplex_samples=self._samples.get("omni_duplex", 0),
        )

    def _get_base(self, request_type: str) -> float:
        """获取 Admin 配置的基准值"""
        if request_type == "chat":
            return self.config.eta_chat_s
        elif request_type == "half_duplex_audio":
            return self.config.eta_half_duplex_s
        elif request_type == "audio_duplex":
            return self.config.eta_audio_duplex_s
        elif request_type == "omni_duplex":
            return self.config.eta_omni_duplex_s
        return 30.0  # 未知类型兜底


# ============ Worker 连接池 + FIFO 队列 ============

class WorkerPool:
    """Worker 连接池 + FIFO 队列调度器

    核心设计：
    - 统一 FIFO 队列（OrderedDict），所有请求类型共享
    - 每个 QueueEntry 持有 asyncio.Future，分配 Worker 时 resolve
    - 单一调度点 _dispatch_next()：Worker 释放 / 取消 / 入队时调用
    """

    def __init__(
        self,
        worker_addresses: List[str],
        max_queue_size: int = 1000,
        request_timeout: float = 300.0,
        eta_config: Optional[EtaConfig] = None,
        ema_alpha: float = 0.3,
        ema_min_samples: int = 3,
        worker_max_concurrency: int = 4,
    ):
        """初始化 Worker 池

        Args:
            worker_addresses: Worker 地址列表
            max_queue_size: 最大排队容量
            request_timeout: 请求超时时间（秒）
            eta_config: ETA 基准配置
            ema_alpha: EMA 平滑系数
            ema_min_samples: EMA 生效最少样本数
            worker_max_concurrency: 每个 Worker 的最大并发 session 数
        """
        self.workers: Dict[str, WorkerConnection] = {}
        self.max_queue_size = max_queue_size
        self.request_timeout = request_timeout
        self.worker_max_concurrency = worker_max_concurrency

        # FIFO 队列
        self._queue: OrderedDict[str, QueueEntry] = OrderedDict()

        # 队列变更回调（用于通知前端）
        self._on_queue_change: Optional[Callable[[], Awaitable[None]]] = None

        # ETA 追踪器
        self.eta_tracker = EtaTracker(
            eta_config=eta_config or EtaConfig(),
            ema_alpha=ema_alpha,
            ema_min_samples=ema_min_samples,
        )

        # HTTP 客户端
        self._client: Optional[httpx.AsyncClient] = None

        # 健康检查任务
        self._health_check_task: Optional[asyncio.Task] = None

        # 解析 Worker 地址
        for i, addr in enumerate(worker_addresses):
            if ":" in addr:
                host, port_str = addr.rsplit(":", 1)
                port = int(port_str)
            else:
                host = addr
                port = 10031 + i

            worker_id = f"worker_{i}"
            gpu_id = i

            self.workers[worker_id] = WorkerConnection(
                worker_id=worker_id,
                host=host,
                port=port,
                gpu_id=gpu_id,
                max_concurrency=worker_max_concurrency,
                capabilities=list(DEFAULT_WORKER_CAPABILITIES),
            )

        logger.info(f"WorkerPool initialized with {len(self.workers)} workers, "
                     f"max_concurrency={worker_max_concurrency}, "
                     f"max_queue_size={max_queue_size}")

    async def start(self) -> None:
        """启动连接池"""
        self._client = httpx.AsyncClient(timeout=self.request_timeout)
        await self._refresh_all_status()
        self._health_check_task = asyncio.create_task(self._health_check_loop())

    async def stop(self) -> None:
        """停止连接池"""
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass

        if self._client:
            await self._client.aclose()
            self._client = None

        # 取消所有排队中的请求
        for entry in list(self._queue.values()):
            if not entry.future.done():
                entry.future.cancel()
        self._queue.clear()

    # ========== Worker 状态管理 ==========

    async def _health_check_loop(self) -> None:
        """定期健康检查"""
        while True:
            try:
                await asyncio.sleep(HEALTH_CHECK_INTERVAL)
                await self._refresh_all_status()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check error: {e}")

    async def _refresh_all_status(self) -> None:
        """刷新所有 Worker 状态"""
        tasks = [self._refresh_worker_status(w) for w in self.workers.values()]
        await asyncio.gather(*tasks, return_exceptions=True)

        # 健康检查后尝试调度（可能有 Worker 恢复为 IDLE）
        self._dispatch_next()

    async def _refresh_worker_status(self, worker: WorkerConnection) -> None:
        """刷新单个 Worker 状态"""
        if self._client is None:
            return

        try:
            resp = await self._client.get(f"{worker.url}/health", timeout=5.0)
            if resp.status_code == 200:
                data = resp.json()
                raw_status = data.get("worker_status", "idle")

                status_map = {
                    "loading": GatewayWorkerStatus.LOADING,
                    "idle": GatewayWorkerStatus.IDLE,
                    "busy_chat": GatewayWorkerStatus.BUSY_CHAT,
                    "busy_half_duplex": GatewayWorkerStatus.BUSY_HALF_DUPLEX,
                    "duplex_active": GatewayWorkerStatus.DUPLEX_ACTIVE,
                    "duplex_paused": GatewayWorkerStatus.DUPLEX_PAUSED,
                    "error": GatewayWorkerStatus.ERROR,
                }
                new_status = status_map.get(raw_status, GatewayWorkerStatus.ERROR)

                # Gateway 调度权威：仅当 Gateway 显式 dispatch 了任务时，
                # 才阻止 health check 将状态降级为 IDLE（防止 Worker 清理期间的瞬态 IDLE 误报）。
                # 关键：release_worker 会清除 _gateway_dispatched，此后 health check 恢复权威，
                # 防止 release 后收到滞后的 busy 状态导致 IDLE→BUSY 逆向升级死锁。
                if worker._gateway_dispatched and new_status == GatewayWorkerStatus.IDLE:
                    logger.debug(
                        f"Health check: {worker.worker_id} reports idle but "
                        f"Gateway dispatched task ({worker.status.value}), keeping Gateway state"
                    )
                    worker.last_heartbeat = datetime.now()
                    return

                worker.status = new_status
                reported_ticket_id = data.get("current_ticket_id")
                if reported_ticket_id is not None or not worker._gateway_dispatched:
                    worker.current_ticket_id = reported_ticket_id
                worker.total_requests = data.get("total_requests", 0)
                worker.avg_inference_time_ms = data.get("avg_inference_time_ms", 0.0)
                worker.capabilities = data.get("capabilities") or list(DEFAULT_WORKER_CAPABILITIES)
                worker.last_heartbeat = datetime.now()
            else:
                worker.status = GatewayWorkerStatus.ERROR
        except Exception as e:
            logger.warning(f"Health check failed for {worker.worker_id}: {e}")
            worker.status = GatewayWorkerStatus.OFFLINE

    # ========== 路由策略 ==========

    def _gpu_busy_counts(self) -> Dict[int, int]:
        """Count active sessions per GPU for load-aware scheduling."""
        counts: Dict[int, int] = {}
        for w in self.workers.values():
            if w.active_tickets:
                counts[w.gpu_id] = counts.get(w.gpu_id, 0) + len(w.active_tickets)
        return counts

    def _get_idle_worker(self, request_type: Optional[str] = None) -> Optional[WorkerConnection]:
        """获取有空闲容量的 Worker（不再要求完全 IDLE，有 capacity 即可）"""
        gpu_busy = self._gpu_busy_counts()
        candidates = (
            w for w in self.workers.values()
            if w.has_capacity and (request_type is None or w.supports(request_type))
        )
        return min(
            candidates,
            key=lambda w: (gpu_busy.get(w.gpu_id, 0), len(w.active_tickets), w.worker_id),
            default=None,
        )

    # ========== FIFO 队列核心 ==========

    class QueueFullError(Exception):
        """队列已满"""
        pass

    def enqueue(
        self,
        request_type: str,
        history_hash: Optional[str] = None,
    ) -> Tuple[QueueTicket, "asyncio.Future[Optional[WorkerConnection]]"]:
        """入队请求

        如果有空闲 Worker，Future 立即 resolve（不进入队列）。
        否则加入 FIFO 队列等待。

        Args:
            request_type: "chat" | "half_duplex_audio" | "audio_duplex" | "omni_duplex"

        Returns:
            (ticket, future)

        Raises:
            QueueFullError: 队列已满
        """
        loop = asyncio.get_running_loop()

        worker = self._get_idle_worker(request_type)

        if worker is not None:
            dispatch_status = self._DISPATCH_STATUS_MAP.get(
                request_type, GatewayWorkerStatus.BUSY_CHAT
            )
            ticket = QueueTicket(
                ticket_id=f"q_{uuid.uuid4().hex[:12]}",
                request_type=request_type,
                position=0,
                estimated_wait_s=0.0,
            )
            worker.mark_busy(dispatch_status, request_type, ticket.ticket_id)
            future: asyncio.Future[Optional[WorkerConnection]] = loop.create_future()
            future.set_result(worker)
            logger.info(
                f"[{ticket.ticket_id}] Immediate assign → {worker.worker_id} "
                f"(type={request_type})"
            )
            return ticket, future

        if len(self._queue) >= self.max_queue_size:
            raise WorkerPool.QueueFullError(
                f"Queue full ({self.max_queue_size} requests)"
            )

        ticket = QueueTicket(
            ticket_id=f"q_{uuid.uuid4().hex[:12]}",
            request_type=request_type,
        )
        future = loop.create_future()
        entry = QueueEntry(ticket=ticket, future=future, history_hash=history_hash)
        self._queue[ticket.ticket_id] = entry

        self._recalc_positions_and_eta()

        logger.info(
            f"[{ticket.ticket_id}] Enqueued: type={request_type}, "
            f"position={ticket.position}, eta={ticket.estimated_wait_s:.1f}s, "
            f"queue_len={len(self._queue)}"
        )

        return ticket, future

    _DISPATCH_STATUS_MAP: Dict[str, GatewayWorkerStatus] = {
        "chat": GatewayWorkerStatus.BUSY_CHAT,
        "half_duplex_audio": GatewayWorkerStatus.BUSY_HALF_DUPLEX,
        "audio_duplex": GatewayWorkerStatus.DUPLEX_ACTIVE,
        "omni_duplex": GatewayWorkerStatus.DUPLEX_ACTIVE,
    }

    def _dispatch_next(self) -> None:
        """尝试将队头分配到空闲 Worker

        这是唯一的 Worker 分配入口（排队后）。
        在 Worker 释放、取消、健康检查后调用。

        关键：分配后立即标记 Worker 为忙碌，防止同一 Worker 被重复分配。
        Gateway 侧收到 Future 结果后会用 mark_busy() 设置正式状态。
        """
        while self._queue:
            selected_ticket_id = None
            selected_entry = None
            selected_worker = None

            for ticket_id, entry in list(self._queue.items()):
                # Future 已完成（被取消或超时）→ 移除，继续扫描
                if entry.future.done():
                    self._queue.pop(ticket_id, None)
                    continue

                worker = self._get_idle_worker(entry.ticket.request_type)
                if worker is not None:
                    selected_ticket_id = ticket_id
                    selected_entry = entry
                    selected_worker = worker
                    break

            if selected_entry is None or selected_worker is None or selected_ticket_id is None:
                break

            # 分配成功：立即标记 Worker 为忙碌，防止重复分配
            req_type = selected_entry.ticket.request_type
            selected_worker.mark_busy(
                self._DISPATCH_STATUS_MAP.get(req_type, GatewayWorkerStatus.BUSY_CHAT),
                req_type,
                selected_entry.ticket.ticket_id,
            )

            self._queue.pop(selected_ticket_id)
            selected_entry.ticket.position = 0
            selected_entry.ticket.estimated_wait_s = 0.0

            if not selected_entry.future.done():
                selected_entry.future.set_result(selected_worker)
                logger.info(
                    f"[{selected_ticket_id}] Dispatched → {selected_worker.worker_id} "
                    f"(type={req_type})"
                )

        # 重算剩余项的位置和 ETA
        if self._queue:
            self._recalc_positions_and_eta()

    def release_worker(self, worker: WorkerConnection, request_type: Optional[str] = None,
                       duration_s: Optional[float] = None,
                       ticket_id: Optional[str] = None) -> None:
        """释放 Worker 的一个并发槽位（任务完成后调用）

        Args:
            worker: 要释放的 Worker
            request_type: 完成的任务类型（用于 EMA 更新）
            duration_s: 任务实际耗时（用于 EMA 更新）
            ticket_id: 要释放的 ticket ID。为 None 时释放全部（完全空闲）。
        """
        if ticket_id and ticket_id in worker.active_tickets:
            worker.release_ticket(ticket_id)
        else:
            worker.mark_idle()

        # 更新 EMA（只在 Worker 完全空闲时记录）
        if request_type and duration_s is not None and duration_s > 0:
            self.eta_tracker.record_duration(request_type, duration_s)

        # 尝试调度下一个
        self._dispatch_next()

    def cancel(self, ticket_id: str) -> bool:
        """取消排队

        Args:
            ticket_id: 要取消的 ticket ID

        Returns:
            是否成功取消
        """
        entry = self._queue.pop(ticket_id, None)
        if entry is None:
            return False

        entry.ticket.cancelled = True
        if not entry.future.done():
            entry.future.cancel()

        logger.info(f"[{ticket_id}] Cancelled, queue_len={len(self._queue)}")

        # 重算位置和 ETA
        self._recalc_positions_and_eta()

        # 取消后可能有空闲 Worker，尝试调度
        self._dispatch_next()

        return True

    def get_ticket(self, ticket_id: str) -> Optional[QueueTicket]:
        """获取指定 ticket 的当前状态"""
        entry = self._queue.get(ticket_id)
        if entry:
            return entry.ticket
        return None

    # ========== ETA 计算 ==========

    def _recalc_positions_and_eta(self) -> None:
        """重算所有排队项的位置和预估等待时间

        多路并发版本：每个 Worker 有 max_concurrency 个槽位，
        已占用的槽位有剩余等待时间，空闲槽位立即可用（wait=0）。
        复杂度 O(Q log S)，S = total_slots。
        """
        if not self._queue:
            return

        now = datetime.now()
        heap: List[Tuple[float, int, int]] = []  # (remaining_seconds, worker_index, slot_id)

        for i, w in enumerate(self.workers.values()):
            num_active = len(w.active_tickets)
            cap = w.max_concurrency
            # 没有活跃 session 时，所有槽位都可用
            if num_active == 0:
                for slot in range(cap):
                    heapq.heappush(heap, (0.0, i, slot))
                continue

            eta = self.eta_tracker.get_eta(w.current_request_type or "chat")
            elapsed = (now - w.task_started_at).total_seconds() if w.task_started_at else 0.0
            remaining = max(0.0, eta - elapsed) if elapsed < eta else 15.0

            for slot in range(cap):
                wait = remaining if slot < num_active else 0.0
                heapq.heappush(heap, (wait, i, slot))

        if not heap:
            for pos_0based, entry in enumerate(self._queue.values()):
                entry.ticket.position = pos_0based + 1
                entry.ticket.estimated_wait_s = 0.0
            return

        for pos_0based, entry in enumerate(self._queue.values()):
            entry.ticket.position = pos_0based + 1
            finish_time, widx, _ = heapq.heappop(heap)
            entry.ticket.estimated_wait_s = round(max(0.0, finish_time), 1)
            next_baseline = self.eta_tracker.get_eta(entry.ticket.request_type)
            heapq.heappush(heap, (finish_time + next_baseline, widx, pos_0based))

    def _get_running_tasks(self) -> List[RunningTaskInfo]:
        """获取当前正在运行的任务列表"""
        now = datetime.now()
        tasks: List[RunningTaskInfo] = []
        for w in self.workers.values():
            if not w.active_tickets or not w.task_started_at or not w.current_request_type:
                continue
            elapsed = (now - w.task_started_at).total_seconds()
            eta = self.eta_tracker.get_eta(w.current_request_type)
            remaining = max(0.0, eta - elapsed)
            tasks.append(RunningTaskInfo(
                worker_id=w.worker_id,
                request_type=w.current_request_type,
                started_at=w.task_started_at,
                elapsed_s=round(elapsed, 1),
                estimated_remaining_s=round(remaining, 1),
                concurrent_sessions=len(w.active_tickets),
            ))
        return tasks

    # ========== 队列状态查询 ==========

    @property
    def queue_length(self) -> int:
        return len(self._queue)

    @property
    def queue_full(self) -> bool:
        return len(self._queue) >= self.max_queue_size

    def get_queue_status(self) -> QueueStatus:
        """获取队列快照"""
        now = datetime.now()
        items: List[QueueTicketSummary] = []
        for entry in self._queue.values():
            t = entry.ticket
            items.append(QueueTicketSummary(
                ticket_id=t.ticket_id,
                request_type=t.request_type,
                position=t.position,
                estimated_wait_s=t.estimated_wait_s,
                enqueued_at=t.enqueued_at,
                wait_elapsed_s=round((now - t.enqueued_at).total_seconds(), 1),
            ))

        return QueueStatus(
            queue_length=len(self._queue),
            max_queue_size=self.max_queue_size,
            items=items,
            running_tasks=self._get_running_tasks(),
        )

    # ========== Worker 信息 ==========

    def get_all_workers(self) -> List[WorkerInfo]:
        return [w.to_info() for w in self.workers.values()]

    def get_worker(self, worker_id: str) -> Optional[WorkerConnection]:
        return self.workers.get(worker_id)

    @property
    def idle_count(self) -> int:
        return sum(1 for w in self.workers.values() if w.is_idle)

    @property
    def busy_count(self) -> int:
        return sum(1 for w in self.workers.values() if w.is_busy)

    @property
    def duplex_count(self) -> int:
        return sum(
            1 for w in self.workers.values()
            if w.status in (GatewayWorkerStatus.DUPLEX_ACTIVE, GatewayWorkerStatus.DUPLEX_PAUSED)
        )

    @property
    def loading_count(self) -> int:
        return sum(1 for w in self.workers.values() if w.status == GatewayWorkerStatus.LOADING)

    @property
    def error_count(self) -> int:
        return sum(1 for w in self.workers.values() if w.status == GatewayWorkerStatus.ERROR)

    @property
    def offline_count(self) -> int:
        return sum(1 for w in self.workers.values() if w.status == GatewayWorkerStatus.OFFLINE)
