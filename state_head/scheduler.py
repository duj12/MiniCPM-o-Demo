"""并发控制 —— **只保证"同一个会话不重复打"，跨会话完全并发**。

## 为什么这里不能有全局闸

推理的合批发生在 Flide 内部（``--batch-window-ms`` 让 GPU 轮次等几毫秒
收拢并发请求）。客户端一旦按到达顺序串起来，模型永远只看到 1 个请求，
合批就退化成"一次一个" —— 并发曲线从"每请求 250ms、8 路总吞吐 32 QPS"
掉成"8 路串行、第 8 个用户等 2 秒"。这正是用户明确否掉的"排队"。

所以本模块只有两类东西：

1. ``SessionGate`` —— **按 key（session_id）的互斥**。同一会话的 turn
   与 heartbeat 同时触发时，后到的等前一个结束再发（它抓到的是更新的帧，
   结果更值得采用；这也是"最新覆盖旧"的语义）。不同会话之间**无任何
   交集**，全部并发。
2. ``Capacity`` —— 一个**很宽**的信号量（默认 64），只用来防止把下游连接
   打爆（fd / 队列内存），**不是**推理闸。打满时会记录告警，说明部署
   规模已经到了该加副本或调 ``--batch-window-ms`` 的地方了。

## 为什么不是"取消 + 重发"

取消一条已经在 GPU 上跑的推理既救不回它的算力，又会让 Flide 那边多一次
abort。等它跑完（通常几百毫秒）再发新的，总代价更低、行为更可预期。
"""
from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Callable, Dict, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class SessionGate:
    """按 key 互斥：同 key 串行，跨 key 并发。"""

    def __init__(self) -> None:
        self._inflight: Dict[str, asyncio.Task] = {}
        self.stats = {"coalesced": 0}

    def busy(self, key: str) -> bool:
        task = self._inflight.get(key)
        return task is not None and not task.done()

    async def run(self, key: str, factory: Callable[[], Awaitable[T]]) -> T:
        """执行 ``factory()``；同 key 已有在途时先等它结束。

        ``key`` 为空串时**不做任何互斥**（调用方明确不在乎重复）。
        """
        if not key:
            return await factory()
        prev = self._inflight.get(key)
        if prev is not None and not prev.done():
            self.stats["coalesced"] += 1
            # 等前一条跑完再发自己这条 —— 用自己的帧与本轮题集，不拿旧结果
            # 冒充新结果（那会让画面比 ASR 还旧，比晚几百毫秒更糟）。
            try:
                await asyncio.shield(prev)
            except Exception:  # noqa: BLE001
                pass
        task: asyncio.Task = asyncio.ensure_future(factory())
        self._inflight[key] = task
        try:
            return await task
        finally:
            if self._inflight.get(key) is task:
                self._inflight.pop(key, None)

    async def close(self) -> None:
        """会话收尾：取消该会话的在途请求。"""
        for key, task in list(self._inflight.items()):
            if not task.done():
                task.cancel()
            self._inflight.pop(key, None)


class Capacity:
    """宽上限的并发计数 —— 防打爆连接，不防并发。"""

    def __init__(self, limit: int = 64) -> None:
        self.limit = max(1, int(limit))
        self._sem = asyncio.Semaphore(self.limit)
        self.inflight = 0
        self.peak = 0
        self.waited = 0

    async def __aenter__(self) -> "Capacity":
        if self._sem.locked():
            self.waited += 1
            logger.warning("[state_head] 并发已达上限 %d —— 该调大 "
                           "STATE_MAX_CONCURRENCY 或加副本了", self.limit)
        await self._sem.acquire()
        self.inflight += 1
        self.peak = max(self.peak, self.inflight)
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        self.inflight -= 1
        self._sem.release()


def parse_urls(raw: Optional[str], default_port: int = 22521) -> list:
    """``STATE_BACKEND_URLS`` → 端点列表。

    支持两种写法：

    * 完整列表：``"http://127.0.0.1:22521,http://127.0.0.1:22522"``；
    * 空 —— entrypoint 已按 ``STATE_REPLICAS`` 生成好列表并注入该变量，
      这里只在它缺失时按副本数兜底（端口从 ``default_port`` 起）。
    """
    if raw:
        urls = [u.strip() for u in raw.split(",") if u.strip()]
        if urls:
            return urls
    return [f"http://127.0.0.1:{default_port}"]
