"""AEC 服务客户端（speech_frontend 的 ``/ws/asr_frontend``）。

**协议形态已实测确认**（见 `orchestrator/tests/README.md`）：

  - 握手：text ``{"type":"hello","service":"asr_frontend","options":{...}}`` → text ``{"type":"ready",...}``
  - 数据：binary 帧，布局 ``header_len(uint32 **大端**) + UTF-8 JSON header + raw payload``
  - 返回：binary ``result`` 帧（0..N 段 float32），以 text ``{"type":"stream_end"}`` 收尾
  - 每条 WS 连接 = **1 路独占会话**（服务端为每条连接构造独立 StreamInference 实例）

实测性能（2026-09，`ws://192.168.88.253:30255`）：
  - 窗口 **3200 样本（200ms）**，首窗延迟 **126–179ms**
  - 60s 长时流无断流、样本守恒 1.0000

**关于 ``farend``**：源码里有"省略 farend 走直通分支会 20% 时间拉伸"的
推断缺陷，但**生产部署实测样本守恒 1.0000，该缺陷不存在**（部署版本与
仓库版本不同）。不过本客户端**始终传 farend**（空闲时传等长静音），
理由与缺陷无关：
  1. 分支不切换 → 时间轴无跳变，时间基准是常量偏移
  2. 给静音是物理上诚实的——确实没在播
  3. 实测三种模式（合成参考/静音/省略）行为一致，传 ref 不引入风险

帧编解码**复用 speech_frontend 的 ``webserver/protocol.py``**，不自己实现。
"""
from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# 复用官方协议实现（唯一权威）
_SF_ROOT = Path(__file__).resolve().parents[3] / "speech_frontend"
if _SF_ROOT.is_dir() and str(_SF_ROOT) not in sys.path:
    sys.path.insert(0, str(_SF_ROOT))
try:
    from webserver.protocol import (  # type: ignore
        ProtocolError,
        encode_frame,
        pack_arrays,
        parse_result_frame,
    )
except ImportError:  # pragma: no cover
    raise ImportError(
        f"需要 speech_frontend 的 webserver.protocol（在 {_SF_ROOT}）。"
        " 若未同步该仓库，请在 106 上同步后再运行。"
    )

import websockets  # noqa: E402

SR = 16000
DEFAULT_CHUNK = 1600  # 100ms —— 官方客户端 main() 里硬编码的值
WS_MAX_SIZE = 64 * 1024 * 1024


class AecClient:
    """一路 AEC 会话。**不是线程安全的**，请在单个 asyncio loop 里用。

    用法::

        aec = AecClient(url)
        await aec.connect()
        async with asyncio.TaskGroup() as tg:
            tg.create_task(aec.recv_loop(on_processed))
            tg.create_task(aec.send_loop(...))       # 或手动 push()
    """

    def __init__(self, url: str, connection_id: str = "orchestrator",
                 enable_se: bool = True, enable_aec: bool = True,
                 max_chunk: int = DEFAULT_CHUNK) -> None:
        self.url = url
        self.connection_id = connection_id
        self.enable_se = enable_se
        self.enable_aec = enable_aec
        self.max_chunk = max_chunk

        self.ws: Optional[websockets.WebSocketClientProtocol] = None
        self.ready = False
        self.closed = False
        self.error: Optional[str] = None

        # 统计
        self.chunks_sent = 0
        self.samples_sent = 0
        self.samples_recv = 0
        self.result_frames = 0
        self.first_result_at: Optional[float] = None
        self._t_start: Optional[float] = None
        self._is_start_pending = True

    # ------------------------------------------------------------------ #

    async def connect(self, timeout: float = 20.0) -> dict:
        """建连并完成 hello/ready 握手。返回 ready 消息。"""
        self.ws = await websockets.connect(self.url, max_size=WS_MAX_SIZE)
        hello = {
            "type": "hello",
            "service": "asr_frontend",
            "connection_id": self.connection_id,
            "options": {
                "enable_speech_enhancement": self.enable_se,
                "enable_aec": self.enable_aec,
                "enable_speech_separation": False,
            },
        }
        await self.ws.send(json.dumps(hello))
        raw = await asyncio.wait_for(self.ws.recv(), timeout=timeout)
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        ready = json.loads(raw)
        if ready.get("type") != "ready":
            raise RuntimeError(f"AEC 服务拒绝 hello: {ready!r}")
        self.ready = True
        self._t_start = time.perf_counter()
        logger.info("AEC 已连接: %s (connection_id=%s)", self.url,
                    ready.get("connection_id"))
        return ready

    # ------------------------------------------------------------------ #

    async def push(self, nearend: np.ndarray, farend: Optional[np.ndarray] = None,
                   is_end: bool = False, flush: bool = False) -> None:
        """送一个 chunk。

        ``nearend`` / ``farend`` 均为 ``(C, T)`` float32（T ≤ max_chunk 时最自然）。
        ``farend=None`` 时**自动补等长静音**（见模块 docstring 的三条理由）。
        """
        if self.ws is None:
            raise RuntimeError("connect() 未调用")
        if nearend.ndim != 2:
            raise ValueError(f"nearend 必须是 (C,T)，得到 {nearend.shape}")
        t = nearend.shape[1]
        if farend is None:
            farend = np.zeros_like(nearend)
        elif farend.shape != nearend.shape:
            raise ValueError(
                f"farend shape {farend.shape} 与 nearend {nearend.shape} 不一致"
            )

        slots, payload = pack_arrays([("nearend", nearend), ("farend", farend)])
        header = {
            "type": "chunk",
            "is_start": self._is_start_pending,
            "is_end": is_end,
            "flush_buffer": flush,
            "sampling_rate": SR,
            "slots": slots,
        }
        await self.ws.send(encode_frame(header, payload))
        self._is_start_pending = False
        self.chunks_sent += 1
        self.samples_sent += t

    async def recv_loop(self, on_audio) -> None:
        """接收循环。``on_audio(np.ndarray)`` 对每个输出段回调一次。

        调用方需与发送并发（``websockets`` 同连接不能同时 send/recv 由
        不同任务驱动时的竞争——本类的 push 与 recv_loop 分属不同任务，
        是官方客户端验证过的双线程模式）。
        """
        if self.ws is None:
            raise RuntimeError("connect() 未调用")
        try:
            while not self.closed:
                raw = await self.ws.recv()
                if isinstance(raw, str):
                    msg = json.loads(raw)
                    mtype = msg.get("type")
                    if mtype == "stream_end":
                        logger.info("AEC: stream_end（chunks_sent=%d）", self.chunks_sent)
                        return
                    if mtype == "error":
                        self.error = f"{msg.get('code')}: {msg.get('message')}"
                        logger.error("AEC 服务错误: %s", self.error)
                        return
                    logger.debug("AEC text: %s", msg)
                    continue

                try:
                    segments = parse_result_frame(raw)
                except ProtocolError as exc:
                    logger.error("AEC result 帧解析失败: %s", exc)
                    continue
                if self.first_result_at is None:
                    self.first_result_at = time.perf_counter()
                self.result_frames += 1
                for seg in segments:
                    seg = np.asarray(seg)
                    self.samples_recv += seg.size
                    on_audio(seg)
        except websockets.ConnectionClosed as exc:
            if not self.closed:
                self.error = f"连接意外关闭: {exc}"
                logger.error("AEC %s", self.error)
        except asyncio.CancelledError:
            raise

    # ------------------------------------------------------------------ #

    async def close(self, send_end: bool = True) -> None:
        """优雅关闭：发 is_end 触发服务端 flush + 状态复位，再关连接。"""
        if self.ws is None:
            return
        self.closed = True
        if send_end and self.chunks_sent > 0:
            try:
                silent = np.zeros((1, self.max_chunk), dtype=np.float32)
                await asyncio.wait_for(
                    self.push(silent, is_end=True), timeout=5.0
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("AEC 发送 is_end 失败（忽略）: %s", exc)
        try:
            await self.ws.close()
        except Exception:  # noqa: BLE001
            pass
        logger.info(
            "AEC 已关闭: sent=%d(%.1fs) recv=%d(%.1fs) frames=%d",
            self.chunks_sent, self.samples_sent / SR,
            self.samples_recv, self.samples_recv / SR, self.result_frames,
        )

    def first_result_latency_ms(self) -> Optional[float]:
        if self.first_result_at is None or self._t_start is None:
            return None
        return (self.first_result_at - self._t_start) * 1000.0

    def sample_ratio(self) -> Optional[float]:
        """输出/输入 样本比。偏离 1.0 说明有时间拉伸。"""
        if self.samples_sent == 0:
            return None
        return self.samples_recv / self.samples_sent
