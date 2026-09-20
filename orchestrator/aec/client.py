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
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---- 帧协议实现 ---------------------------------------------------------- #
# 默认用本仓库内联的那份（``orchestrator/aec/_protocol.py``，自 speech_frontend
# 原样拷贝），这样部署不必再同步那个仓库。它带一段 round-trip 自检。
#
# 上游是该协议的**唯一权威**，所以留一个开关：设 ORCH_AEC_PROTOCOL=upstream
# 就切回直接 import speech_frontend 的 webserver.protocol（内联前的老行为）。
# 内联自检失败时也会提示这个兜底。
_PROTOCOL_SOURCE = (os.environ.get("ORCH_AEC_PROTOCOL") or "local").strip().lower()

_SF_ROOT = Path(__file__).resolve().parents[3] / "speech_frontend"


def _load_protocol_upstream():
    """从 speech_frontend 仓库取（老行为；需要该仓库在同级目录）。"""
    if _SF_ROOT.is_dir() and str(_SF_ROOT) not in sys.path:
        sys.path.insert(0, str(_SF_ROOT))
    from webserver.protocol import (  # type: ignore
        ProtocolError,
        encode_frame,
        pack_arrays,
        parse_result_frame,
    )
    return ProtocolError, encode_frame, pack_arrays, parse_result_frame


def _load_protocol_local():
    """用本仓库内联的那份。"""
    from . import _protocol
    if not _protocol.roundtrip_ok:
        # 自检失败 = 这份拷贝与「能被正确解析」的语义不符，宁可起不来
        raise ImportError(
            f"内联的 AEC 帧协议未通过 round-trip 自检"
            f"（{_protocol.roundtrip_error}）。\n"
            f"  这份拷贝在 {Path(__file__).with_name('_protocol.py')}，"
            f"源自 speech_frontend 的 webserver/protocol.py。\n"
            f"  若上游确实改了协议：按 _protocol.py 顶部的说明重新同步；\n"
            f"  若想先用权威实现跑起来：设 ORCH_AEC_PROTOCOL=upstream"
            f"（需要同级有 speech_frontend 仓库）。")
    return (_protocol.ProtocolError, _protocol.encode_frame,
            _protocol.pack_arrays, _protocol.parse_result_frame)


try:
    (_ProtocolError, _encode_frame, _pack_arrays,
     _parse_result_frame) = (
        _load_protocol_upstream() if _PROTOCOL_SOURCE == "upstream"
        else _load_protocol_local())
except ImportError as exc:
    raise ImportError(
        f"加载 AEC 帧协议失败（ORCH_AEC_PROTOCOL={_PROTOCOL_SOURCE}）: {exc}") from exc

# 对外保持与内联前一致的四个符号名
ProtocolError = _ProtocolError
encode_frame = _encode_frame
pack_arrays = _pack_arrays
parse_result_frame = _parse_result_frame

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

        self.ws = None
        self.ready = False
        self.closed = False
        self.error: Optional[str] = None
        # 收尾状态：drain() 发 is_end 后置 _drained；recv_loop 收到
        # stream_end 后置 _stream_ended（drain 轮询它来判断尾部是否收全）
        self._drained = False
        self._stream_ended = False

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
                        self._stream_ended = True
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

    async def drain(self, timeout: float = 10.0) -> bool:
        """发 ``is_end`` 触发服务端 flush，并等 ``stream_end`` 回来。

        ⚠️ 与 ASR 同理：发完 ``is_end`` **不能立刻关连接** —— 服务端还要
        把缓冲区里的尾部推理出来（AEC 是滑窗，尾部窗口尚未输出）。立刻关
        会丢掉最后一窗，表现为样本比略小于 1（如 0.966）。

        拆成两步：``drain()`` 收尾（可慢），``close()`` 断开（必须快）。
        返回是否收到 ``stream_end``。
        """
        if self.ws is None or self._drained:
            return False
        self._drained = True
        if self.chunks_sent > 0:
            try:
                silent = np.zeros((1, self.max_chunk), dtype=np.float32)
                await asyncio.wait_for(self.push(silent, is_end=True), timeout=5.0)
            except Exception as exc:  # noqa: BLE001
                logger.warning("AEC 发送 is_end 失败: %s", exc)
                return False
        # 等 stream_end（recv_loop 收到时置 _stream_ended）
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            if self._stream_ended:
                logger.info("AEC 已收到 stream_end（%.1fs）", time.monotonic() - t0)
                return True
            await asyncio.sleep(0.05)
        logger.warning("AEC 等待 stream_end 超时（%.1fs）", timeout)
        return False

    async def close(self, send_end: bool = False) -> None:
        """关闭连接。**必须快** —— 收尾请先调 ``drain()``。"""
        if self.ws is None:
            return
        self.closed = True
        try:
            await self.ws.close()
        except Exception:  # noqa: BLE001
            pass
        logger.info(
            "AEC 已关闭: sent=%d(%.1fs) recv=%d(%.1fs) frames=%d 样本比=%.4f",
            self.chunks_sent, self.samples_sent / SR,
            self.samples_recv, self.samples_recv / SR, self.result_frames,
            self.samples_recv / self.samples_sent if self.samples_sent else 0.0,
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
