"""TTS 服务客户端（gRPC）。**两种模式**：

  · :meth:`TtsClient.synthesize` —— **整段合成**（``inference``，unary-stream）。
    LLM 整条回复产完后一次性合成。首帧实测 **508ms**，占总耗时 78%。
  · :meth:`TtsClient.open_stream` —— **双向流式**（``stream_inference``，
    stream-stream）。LLM 文本 delta 一到就喂，TTS 音频一出就下发，
    首声降到「首个 delta + TTS 首帧」。实测首帧 **381ms**。

**为什么两者都用同步 stub + 专用线程，不用 grpc.aio**：
``protos/tts_pb2_grpc.py`` 生成的是**同步 stub**（``channel.stream_stream``），
配 aio channel 依赖生成代码恰好兼容，比较脆。整段合成阻塞几百毫秒本来就要
丢线程池；流式这条路把**输入侧**（阻塞等 LLM 下一个 delta）放到专用线程，
**输出侧**仍回到 asyncio 事件循环，所以两边都不卡 loop。

**流式的分句是服务端做的** —— ``tts/grpc_server.py`` 的 ``split_stream_text``
会把原始文本转给独立的 ``STREAM_SPLITTER`` 服务切句。所以客户端**不必分句、
不必攒批**，来一个字发一个字即可。

**实测结果**（见 ``orchestrator/tests/README.md``）：

  - PCM = **int16 @ 24kHz**（两种模式一致）
  - ``CHAR_TIME_MAP`` 提供字级时间戳 ``[[字符, 起始秒, 结束秒], ...]``
  - ⚠️ 流式下 ``sentence_index`` 实测**恒为 -1**，不能拿它做判断；
    收尾只看 ``inference_end``
"""
from __future__ import annotations

import asyncio
import json
import logging
import queue
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncIterator, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

TTS_SR = 24000
# 照抄示例：长文本会超默认接收上限
GRPC_OPTIONS = [("grpc.max_receive_message_length", 4605632 * 2)]

# TTS 仓库路径（proto 所在）
_TTS_ROOT = Path(__file__).resolve().parents[3] / "TTS"
if _TTS_ROOT.is_dir() and str(_TTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_TTS_ROOT))


@dataclass
class TtsResult:
    pcm: np.ndarray                       # int16 @ 24kHz
    char_time_map: Optional[list] = None  # [[char, start_s, end_s], ...]
    first_frame_s: float = 0.0
    total_s: float = 0.0
    n_frames: int = 0

    @property
    def duration_s(self) -> float:
        return len(self.pcm) / TTS_SR


class TtsStream:
    """一次**双向流式**合成会话（``stream_inference``）。

    用法::

        st = client.open_stream(loop)
        st.feed("米家智能")          # 可以反复喂，来多少喂多少
        st.feed("按摩椅。")
        st.end()                     # 文本发完
        async for pcm in st.audio_iter():   # int16 @24k，逐块
            ...

    **线程模型**：gRPC 是阻塞式的，跑在专用线程里；线程阻塞在「等下一个
    delta」和「等服务端回包」上。音频回包经 ``call_soon_threadsafe`` 投回
    asyncio 事件循环，所以调用方（asyncio 侧）始终不阻塞。

    **分句由服务端做**，客户端不必攒批 —— 喂单个字也可以。

    ⚠️ ``sentence_index`` 实测恒为 -1，收尾只能看 ``inference_end``。
    """

    #: 音频队列的结束哨兵（服务端 ``inference_end`` 或出错）
    _SENTINEL = object()

    def __init__(self, stub, loop: asyncio.AbstractEventLoop,
                 tts_type: str, speaker_id: Optional[str],
                 speaker_vector_b64: Optional[str] = None,
                 timeout: float = 60.0, infer_id: str = "orch-stream") -> None:
        self._stub = stub
        self._loop = loop
        self._timeout = timeout
        self._infer_id = infer_id
        self._tts_type = tts_type
        self._speaker_id = speaker_id
        self._speaker_vector_b64 = speaker_vector_b64

        # 输入侧：asyncio 线程 put，gRPC 线程 get
        self._req_q: "queue.Queue" = queue.Queue()
        # 输出侧：gRPC 线程 call_soon_threadsafe put，asyncio 线程 get
        self._audio_q: asyncio.Queue = asyncio.Queue()

        self._sent_first = False      # is_first 只置第一条
        self._ended = False           # end() 已调用
        self._aborted = False
        self._worker: Optional[threading.Thread] = None
        self._error: Optional[str] = None
        self._lock = threading.Lock()

        # 统计
        self.chunks_out = 0
        self.samples_out = 0
        self.char_time_map: Optional[list] = None
        self.first_chunk_s: float = 0.0
        self.started_at = time.perf_counter()

        self._worker = threading.Thread(
            target=self._run, name="tts-stream", daemon=True)
        self._worker.start()

    # ------------------------------------------------------------------ #

    @property
    def error(self) -> Optional[str]:
        return self._error

    @property
    def aborted(self) -> bool:
        return self._aborted

    def feed(self, text: str, is_last: bool = False) -> None:
        """喂一段文本。线程安全，非阻塞。

        ``is_last=True`` 等价于先 feed 再 :meth:`end`。
        """
        if self._aborted:
            return
        text = text or ""
        if text or is_last:
            self._req_q.put((text, is_last))

    def end(self) -> None:
        """文本发完，让服务端收尾（对端见到 ``is_last`` 会补 ``inference_end``）。"""
        if self._ended or self._aborted:
            return
        self._ended = True
        self._req_q.put(("", True))

    async def audio_iter(self) -> AsyncIterator[np.ndarray]:
        """逐块产出 int16 @24kHz PCM。服务端收尾或出错时结束。"""
        while True:
            item = await self._audio_q.get()
            if item is self._SENTINEL:
                return
            yield item

    async def abort(self, timeout: float = 3.0) -> None:
        """打断：停止喂文本、让 RPC 尽快结束、等线程退出。

        ⚠️ 必须在**新一句开始之前**把旧流收干净 —— 否则旧流的音频会继续
        投进队列，被新的消费循环读到（串音）。
        """
        if self._aborted:
            return
        self._aborted = True
        # 用哨兵唤醒阻塞在 get() 的生成器线程，让它退出循环
        self._req_q.put(None)
        await asyncio.to_thread(self._join, timeout)

    def _join(self, timeout: float) -> None:
        if self._worker is not None and self._worker.is_alive():
            self._worker.join(timeout=timeout)

    # ------------------------------------------------------------------ #

    def _build_request(self, text: str, is_last: bool):
        from protos import tts_pb2  # type: ignore

        is_first = not self._sent_first
        self._sent_first = True
        kwargs = {}
        if self._speaker_vector_b64:
            import base64
            kwargs["speaker_vector"] = base64.b64decode(self._speaker_vector_b64)
        else:
            kwargs["speaker_id"] = str(self._speaker_id)
        return tts_pb2.Text(
            text=text,
            tts_type=self._tts_type,
            is_first=is_first,
            is_last=is_last,
            return_sep=False,
            flush_buffer=False,
            secondary_style_id=getattr(
                tts_pb2.Text.SECONDARY_STYLE_ID, "jiangpin"
            ),
            **kwargs,
        )

    def _request_iter(self):
        """生成器线程侧：阻塞取文本，直到 end() / abort()。"""
        while True:
            item = self._req_q.get()
            if item is None:        # abort 哨兵
                return
            text, is_last = item
            try:
                yield self._build_request(text, is_last)
            except Exception as exc:  # noqa: BLE001
                self._fail(f"构造请求失败: {exc}")
                return
            if is_last:
                return

    def _put_audio(self, pcm: np.ndarray) -> None:
        """gRPC 线程 → asyncio 队列。"""
        try:
            self._loop.call_soon_threadsafe(self._audio_q.put_nowait, pcm)
        except RuntimeError:
            # loop 已关闭（会话收尾竞态）：丢弃即可
            pass

    def _finish(self) -> None:
        try:
            self._loop.call_soon_threadsafe(
                self._audio_q.put_nowait, self._SENTINEL)
        except RuntimeError:
            pass

    def _fail(self, msg: str) -> None:
        with self._lock:
            if self._error is None:
                self._error = msg
                logger.warning("TTS 流式会话出错: %s", msg)

    def _run(self) -> None:
        """专用线程：跑阻塞的 stream_inference，把音频投回 loop。"""
        try:
            t0 = time.perf_counter()
            for r in self._stub.stream_inference(
                self._request_iter(),
                metadata=[("grpc-infer-id", self._infer_id)],
                timeout=self._timeout,
            ):
                if self._aborted:
                    break
                if r.data_type == 0:            # AUDIO
                    if not r.data:
                        continue
                    if self.first_chunk_s == 0.0:
                        self.first_chunk_s = time.perf_counter() - t0
                    pcm = np.frombuffer(r.data, dtype=np.int16)
                    self.chunks_out += 1
                    self.samples_out += pcm.size
                    self._put_audio(pcm.copy())
                elif r.data_type == 1:          # CHAR_TIME_MAP
                    try:
                        self.char_time_map = json.loads(r.data)
                    except Exception:  # noqa: BLE001
                        pass
                # ⚠️ sentence_index 实测恒为 -1，不能用来判断任何东西
                if r.inference_end:
                    break
        except Exception as exc:  # noqa: BLE001
            if not self._aborted:
                self._fail(f"{type(exc).__name__}: {exc}")
        finally:
            self._finish()


class TtsClient:
    """一路 TTS 会话。

    ``synthesize`` 是协程，内部把阻塞的 gRPC 调用丢进线程池，
    不阻塞 asyncio loop。
    """

    #: 支持 :meth:`open_stream`（执行器据此决定走不走流式）
    supports_streaming = True

    def __init__(self, host: str, port: int, timeout: float = 60.0,
                 tts_type: str = "mltts", speaker_id: str = "17",
                 secondary_style: str = "jiangpin") -> None:
        self.host = host
        self.port = port
        self.timeout = timeout
        self.default_tts_type = tts_type
        self.default_speaker_id = speaker_id
        self.secondary_style = secondary_style

        self._channel = None
        self._stub = None
        self.closed = False
        # 收尾竞态防护：_closing 阻止新请求；_inflight 让 close() 能等在飞
        # 的合成跑完，而不是直接关 channel 把 RPC 打断
        self._closing = False
        self._inflight = 0
        self.calls = 0
        self.total_audio_s = 0.0
        self.last_total_s = 0.0

    # ------------------------------------------------------------------ #

    def _ensure_stub(self):
        if self._stub is not None:
            return self._stub
        import grpc
        from protos import tts_pb2_grpc  # type: ignore
        self._channel = grpc.insecure_channel(
            f"{self.host}:{self.port}", options=GRPC_OPTIONS
        )
        self._stub = tts_pb2_grpc.TTSStub(self._channel)
        logger.info("TTS 已连接: %s:%s", self.host, self.port)
        return self._stub

    # ------------------------------------------------------------------ #

    def _synth_blocking(self, text: str, tts_type: str,
                        speaker_id: Optional[str],
                        speaker_vector_b64: Optional[str]) -> TtsResult:
        """阻塞式合成（在线程池里跑）。"""
        import time
        from protos import tts_pb2  # type: ignore

        stub = self._ensure_stub()
        t0 = time.perf_counter()

        kwargs = {}
        if speaker_vector_b64:
            import base64
            kwargs["speaker_vector"] = base64.b64decode(speaker_vector_b64)
        elif speaker_id:
            kwargs["speaker_id"] = str(speaker_id)
        else:
            kwargs["speaker_id"] = str(self.default_speaker_id)

        req = tts_pb2.Text(
            text=text,
            tts_type=tts_type,
            is_first=True,
            is_last=True,
            return_sep=False,
            secondary_style_id=getattr(
                tts_pb2.Text.SECONDARY_STYLE_ID, self.secondary_style
            ),
            **kwargs,
        )

        buf = bytearray()
        char_time_map = None
        first_frame_s = 0.0
        n_frames = 0
        for r in stub.inference(req, metadata=[("grpc-infer-id", "orch")],
                                timeout=self.timeout):
            n_frames += 1
            if first_frame_s == 0.0:
                first_frame_s = time.perf_counter() - t0
            if r.data_type == 0:        # AUDIO
                buf.extend(r.data)
            elif r.data_type == 1:      # CHAR_TIME_MAP
                try:
                    char_time_map = json.loads(r.data)
                except Exception:  # noqa: BLE001
                    char_time_map = None

        total_s = time.perf_counter() - t0
        pcm = np.frombuffer(bytes(buf), dtype=np.int16)
        return TtsResult(pcm=pcm, char_time_map=char_time_map,
                         first_frame_s=first_frame_s, total_s=total_s,
                         n_frames=n_frames)

    async def synthesize(self, text: str, tts_type: Optional[str] = None,
                         speaker_id: Optional[str] = None,
                         speaker_vector_b64: Optional[str] = None
                         ) -> Optional[np.ndarray]:
        """整段合成。返回 int16 @ 24kHz 的 1-D 数组，失败返回 None。

        ⚠️ 会话收尾期间的竞态：drain 时 OmniLLM 仍在产出，可能触发新的
        Speak，而此时 ``close()`` 已关掉 gRPC channel → RPC 被中断
        （表现为 ``_MultiThreadedRendezvous ... terminated``）。
        这里用 ``_closing`` 标志显式拒绝关停后的请求，避免误报为错误。
        """
        if self.closed or self._closing:
            logger.debug("TTS 已停止，忽略合成请求（%d 字）", len(text or ""))
            return None
        text = (text or "").strip()
        if not text:
            return None
        loop = asyncio.get_running_loop()
        self._inflight += 1
        try:
            res = await loop.run_in_executor(
                None, self._synth_blocking, text,
                tts_type or self.default_tts_type,
                speaker_id, speaker_vector_b64,
            )
        finally:
            self._inflight -= 1
        self.calls += 1
        self.total_audio_s += res.duration_s
        self.last_total_s = res.total_s
        logger.debug(
            "TTS: %d 字 -> %.2fs 音频，首帧 %.0fms，总 %.0fms（%d 帧）",
            len(text), res.duration_s, res.first_frame_s * 1000,
            res.total_s * 1000, res.n_frames,
        )
        return res.pcm

    def open_stream(self, tts_type: Optional[str] = None,
                    speaker_id: Optional[str] = None,
                    speaker_vector_b64: Optional[str] = None,
                    infer_id: str = "orch-stream") -> Optional["TtsStream"]:
        """开一次**双向流式**合成会话（非阻塞，立即返回）。

        调用方拿到 :class:`TtsStream` 后反复 ``feed()`` 文本、再从
        ``audio_iter()`` 收音频。收尾/打断见该类的 ``end()`` / ``abort()``。

        关停中（``_closing``/``closed``）返回 ``None`` —— 与 ``synthesize``
        同一套竞态防护。流在飞期间也算 ``_inflight``，否则 ``close()`` 会
        在流还没收干净时就关掉 channel。
        """
        if self.closed or self._closing:
            logger.debug("TTS 已停止，拒绝开流式会话")
            return None
        stub = self._ensure_stub()
        loop = asyncio.get_running_loop()
        self._inflight += 1
        try:
            st = TtsStream(
                stub, loop,
                tts_type=tts_type or self.default_tts_type,
                speaker_id=speaker_id or self.default_speaker_id,
                speaker_vector_b64=speaker_vector_b64,
                timeout=self.timeout,
                infer_id=infer_id,
            )
        except Exception:
            self._inflight -= 1
            raise
        self.calls += 1
        # 流结束后自动减 inflight：包一层，不侵入 TtsStream 自身
        orig_finish = st._finish

        def _finish_counted() -> None:
            try:
                orig_finish()
            finally:
                self._inflight -= 1
                self.total_audio_s += st.samples_out / TTS_SR

        st._finish = _finish_counted  # type: ignore[method-assign]
        return st

    async def synthesize_full(self, text: str, **kw) -> Optional[TtsResult]:
        """同 ``synthesize`` 但返回完整结果（含字级时间戳）。"""
        if self.closed or self._closing:
            return None
        text = (text or "").strip()
        if not text:
            return None
        loop = asyncio.get_running_loop()
        self._inflight += 1
        try:
            res = await loop.run_in_executor(
                None, self._synth_blocking, text,
                kw.get("tts_type") or self.default_tts_type,
                kw.get("speaker_id"), kw.get("speaker_vector_b64"),
            )
        finally:
            self._inflight -= 1
        self.calls += 1
        self.total_audio_s += res.duration_s
        return res

    # ------------------------------------------------------------------ #

    async def close(self) -> None:
        """关闭。

        先置 ``_closing`` 拒绝新请求，再**等在飞的合成跑完**，最后才关
        channel —— 否则正在进行的 RPC 会被打断，表现为
        ``_MultiThreadedRendezvous ... terminated`` 的假错误
        （会话收尾时 OmniLLM 仍在产出、可能触发新的 Speak）。
        """
        if self.closed:
            return
        self._closing = True
        if self._inflight > 0:
            deadline = time.monotonic() + 15.0
            while self._inflight > 0 and time.monotonic() < deadline:
                await asyncio.sleep(0.1)
            if self._inflight > 0:
                logger.warning("TTS 关闭时仍有 %d 个合成在飞", self._inflight)
        self.closed = True
        if self._channel is not None:
            try:
                self._channel.close()
            except Exception:  # noqa: BLE001
                pass
        logger.info("TTS 已关闭: %d 次合成，合计 %.1fs 音频",
                    self.calls, self.total_audio_s)


class MockTtsClient:
    """测试替身：不连服务，产一段确定性的正弦音频。

    用于无 TTS 服务时的端到端链路验证（如本地开发）。
    """

    def __init__(self, ms_per_char: float = 150.0) -> None:
        self.ms_per_char = ms_per_char
        self.closed = False
        self._closing = False      # 与 TtsClient 接口对齐
        self._inflight = 0
        self.calls = 0
        self.total_audio_s = 0.0

    async def synthesize(self, text: str, **kw) -> Optional[np.ndarray]:
        if self.closed or self._closing or not (text or "").strip():
            return None
        n = int(TTS_SR * len(text) * self.ms_per_char / 1000.0)
        n = max(n, TTS_SR // 4)
        t = np.arange(n, dtype=np.float64) / TTS_SR
        wave = 0.25 * np.sin(2 * np.pi * 420 * t)
        env = np.minimum(1.0, np.minimum(t * 20, (n / TTS_SR - t) * 20))
        pcm = (wave * np.maximum(env, 0) * 32767).astype(np.int16)
        self.calls += 1
        self.total_audio_s += len(pcm) / TTS_SR
        return pcm

    async def close(self) -> None:
        self.closed = True
