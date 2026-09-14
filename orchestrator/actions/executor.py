"""执行下游返回的 action。

本阶段支持：
  - ``Speak``  → 调 TTS 合成 → 送浏览器 → 同时落进 RefTrack（供 AEC）
  - ``Cancel`` → **原子地**停播放器 + truncate RefTrack
  - ``SendToOmni`` / ``Emit``

``Cancel`` 的三件事必须原子：否则 AEC 会拿着没播出的音频当参考，
主动误适配去追一个不存在的回声 —— 比不给参考更糟。
"""
from __future__ import annotations

import logging
import time
import uuid
from typing import TYPE_CHECKING, Optional

import numpy as np

from ..protocol import TtsAudio, TtsCancel, TtsEnd, TtsStart
from ..downstream.interface import Cancel, Emit, SendToOmni, Speak

if TYPE_CHECKING:
    from ..session import OrchestratorSession

logger = logging.getLogger(__name__)

TTS_SR = 24000
# 播放提前量（与 MiniCPM-o-Demo 的 config.json 一致）
PLAYBACK_DELAY_MS = 200


class ActionExecutor:
    """把下游 action 变成实际副作用。"""

    def __init__(self, playback_delay_ms: int = PLAYBACK_DELAY_MS) -> None:
        self.playback_delay_ms = playback_delay_ms
        self._current_response_id: Optional[str] = None
        self._tts_seq = 0
        self.speaks_done = 0
        self.cancels_done = 0
        # 播放起点的墙钟锚（用于没有回执时的退化路径）
        self._playback_anchor_ctx: Optional[float] = None

    # ------------------------------------------------------------------ #

    async def execute(self, act, session: "OrchestratorSession") -> None:
        if isinstance(act, Speak):
            await self._speak(act, session)
        elif isinstance(act, Cancel):
            await self._cancel(act, session)
        elif isinstance(act, SendToOmni):
            await self._send_to_omni(act, session)
        elif isinstance(act, Emit):
            logger.info("[emit:%s] %s", act.channel, act.payload)
        else:
            logger.warning("未知 action: %r", act)

    # ------------------------------------------------------------------ #

    async def _speak(self, act: Speak, session: "OrchestratorSession") -> None:
        if session.tts is None:
            logger.warning("Speak 但未配置 TTS 客户端，忽略")
            return

        response_id = uuid.uuid4().hex[:8]
        self._current_response_id = response_id
        self._tts_seq = 0

        await session.send_to_client(TtsStart(
            response_id=response_id, text=act.text, sample_rate=TTS_SR,
        ))

        try:
            pcm24 = await session.tts.synthesize(
                act.text, tts_type=act.tts_type, speaker_id=act.speaker_id,
                speaker_vector_b64=act.speaker_vector_b64,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("TTS 合成失败: %s", exc)
            await session.send_to_client(TtsEnd(response_id=response_id))
            return

        if pcm24 is None or len(pcm24) == 0:
            logger.warning("TTS 返回空音频")
            await session.send_to_client(TtsEnd(response_id=response_id))
            return

        # 分块发送（每块 0.5s，便于浏览器调度并尽早开始播放）
        chunk = TTS_SR // 2
        for i in range(0, len(pcm24), chunk):
            part = pcm24[i:i + chunk]
            await session.send_to_client(
                TtsAudio.from_int16(part, response_id, self._tts_seq)
            )
            self._tts_seq += 1

        await session.send_to_client(TtsEnd(response_id=response_id))

        # ⚠️ 关键：把同一份 PCM 落进参考轨，供 AEC 使用。
        # 播放延迟 = playback_delay_ms，所以参考相对当前时钟在"未来"。
        if session.ref_track is not None:
            ctx_time = self._next_ctx_time(session)
            session.ref_track.place(response_id, 0, pcm24.astype(np.int16), ctx_time)
            logger.debug("ref 落位 response=%s ctx_time=%.3f len=%d",
                         response_id, ctx_time, len(pcm24))

        self.speaks_done += 1
        logger.info("TTS 完成 response=%s 文本 %d 字 音频 %.2fs",
                    response_id, len(act.text), len(pcm24) / TTS_SR)

    def _next_ctx_time(self, session: "OrchestratorSession") -> float:
        """计算本次播放的 AudioContext 时刻。

        真实场景由浏览器回执提供；这里按"现在 + 播放提前量"估算，
        使参考轨落在未来（这正是我们要的：ref 写在 mic 之前，
        read() 总能读到真数据）。
        """
        now_s = session.clock.seconds()
        return now_s + self.playback_delay_ms / 1000.0

    # ------------------------------------------------------------------ #

    async def _cancel(self, act: Cancel, session: "OrchestratorSession") -> None:
        """中断播报。三件事原子做。"""
        rid = act.response_id or self._current_response_id
        if rid is None:
            return
        # ① 通知浏览器停止并清空播放器
        await session.send_to_client(TtsCancel(response_id=rid, reason=act.reason))
        # ② 截断参考轨（未播出的部分）
        if session.ref_track is not None:
            n = session.ref_track.truncate(rid)
            if n:
                logger.info("取消 %s：截断 ref %d 采样（%.0fms）",
                            rid, n, n / 16000 * 1000)
        # ③ 清当前指针
        if self._current_response_id == rid:
            self._current_response_id = None
        self.cancels_done += 1

    async def _send_to_omni(self, act: SendToOmni,
                            session: "OrchestratorSession") -> None:
        if session.omni is None:
            return
        try:
            if act.text:
                await session.omni.send_text(act.text,
                                             force_listen=act.force_listen)
        except Exception as exc:  # noqa: BLE001
            logger.warning("SendToOmni 失败: %s", exc)
