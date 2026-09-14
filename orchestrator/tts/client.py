"""TTS 服务客户端（gRPC，整段合成）。

**调用模式照抄 ``TTS/tests/grpc_client.py`` 的 ``inference()``。**

**为什么用同步 stub + run_in_executor，不用 grpc.aio**：
TTS 是整段合成，调用本身就要阻塞几百毫秒等全部 PCM 回来，期间没有别的
活可干；而 ``protos/tts_pb2_grpc.py`` 生成的是**同步 stub**
（``channel.unary_stream``），配 aio channel 依赖生成代码恰好兼容，比较脆。

**实测结果**（见 ``orchestrator/tests/README.md``）：

  - PCM = **int16 @ 24kHz**
  - 首帧延迟 **508ms**，总耗时 649ms（RTF 0.144）—— 首帧占总耗时 78%
  - ``CHAR_TIME_MAP`` 提供字级时间戳 ``[[字符, 起始秒, 结束秒], ...]``
  - ⚠️ 首帧占比高说明整段合成有优化空间；后续可切 ``stream_inference``
    （按句 client-streaming）把首字延迟降到句级。接口按可切换设计。
"""
from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

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


class TtsClient:
    """一路 TTS 会话。

    ``synthesize`` 是协程，内部把阻塞的 gRPC 调用丢进线程池，
    不阻塞 asyncio loop。
    """

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
