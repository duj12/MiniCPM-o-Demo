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
        # 当前这句**预计播完**的会话采样位置。用来判断新回复要不要打断旧句：
        # 它不能靠 playback.started/ended 判断 —— 那些只反映"音频送达"，
        # 不反映"播完"（整段是排程播放的，送达时可能才刚起播）。
        self._current_play_until: int = 0
        self._tts_seq = 0
        self.speaks_done = 0
        self.cancels_done = 0
        # 在飞的合成数：会话收尾要等它归零，否则 close() 会打断 RPC
        self._speak_inflight = 0
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
        """合成并发送 TTS 音频。

        ``_speak_inflight`` 覆盖**整个方法**（合成 + 分帧发送），不只是
        合成 —— 收尾时必须等音频真正送达浏览器，否则 close() 会把它截断，
        表现为「合成成功但客户端一帧没收到」。
        """
        if session.tts is None:
            logger.warning("Speak 但未配置 TTS 客户端，忽略")
            return
        # ⚠️ 只在**已关闭**时跳过，不能在 drain（收尾）期间跳过 ——
        # 收尾恰恰是要把最后那句合成出来的时机。
        # close() 之后 TTS 的 gRPC channel 已关，再发起只会得到假错误。
        if session.closed:
            logger.info("会话已关闭，跳过 TTS 合成（%d 字）", len(act.text or ""))
            return
        self._speak_inflight += 1
        try:
            await self._speak_inner(act, session)
        finally:
            self._speak_inflight -= 1

    async def _speak_inner(self, act: Speak,
                           session: "OrchestratorSession") -> None:

        # ⚠️ **新回复要打断仍在播放的旧句** —— 否则参考轨会乱。
        #
        # 真机现象：上一句还没播完就插话，新回复的回声**完全消不掉**、
        # 被 ASR 整段识别。根因是四个缺陷叠加：
        #   ① 前端 `PcmPlayer.stop()` 是空操作：gain 设 0 又立刻设回 1，
        #      且已 `node.start()` 排程的 buffer **无法取消**，旧音频照播；
        #   ② 前端收到 `tts.start` 时 `nextAt = max(nextAt, now+0.2)` ——
        #      `nextAt` 还停在旧句末尾，于是新句被排到旧句**之后**；
        #   ③ 服务端把新句落位在"当前时刻 + 提前量"，与②的实际排程不符；
        #   ④ 旧句的排程音频还在播 → 实际播出的是旧句，参考轨写的却是新句。
        #   → mic 里的回声来自旧句，farend 是新句，两者完全无关，消不掉。
        #
        # 所以打断必须**由服务端在开新句前主动做**，且四件事原子完成。
        self._interrupt_current(session, reason="superseded")

        response_id = uuid.uuid4().hex[:8]
        self._current_response_id = response_id
        self._tts_seq = 0

        await session.send_to_client(TtsStart(
            response_id=response_id, text=act.text, sample_rate=TTS_SR,
        ))

        import time as _time
        t_synth0 = _time.monotonic()
        try:
            pcm24 = await session.tts.synthesize(
                act.text, tts_type=act.tts_type, speaker_id=act.speaker_id,
                speaker_vector_b64=act.speaker_vector_b64,
            )
        except Exception as exc:  # noqa: BLE001
            # 会话收尾时的中断不是真错误（sess.closed 已置），降级为 info
            lvl = logger.info if session.closed else logger.error
            lvl("TTS 合成未完成: %s: %s", type(exc).__name__, str(exc)[:200])
            if not session.closed:
                await session.send_to_client(TtsEnd(response_id=response_id))
            return
        # 指标：合成耗时与音频时长
        if session.metrics is not None:
            session.metrics.tts_total.record((_time.monotonic() - t_synth0) * 1000)
            session.metrics.inc("tts_calls")

        if pcm24 is None or len(pcm24) == 0:
            logger.warning("TTS 返回空音频")
            await session.send_to_client(TtsEnd(response_id=response_id))
            return

        # ⚠️ 记录**送第一块音频时**的会话采样位置 —— 这是参考轨落位的
        # 基准。浏览器会在收到首块后 delay 开始播放，所以播放起点 =
        # 此刻 + playback_delay。用会话时钟自算，**不混用浏览器 ctx_time**
        # （混用会导致写入位置变成大负数，参考轨读出来全是 0 —— 实测踩过）。
        t_send = session.clock.now()

        # 分块发送（每块 0.5s，便于浏览器调度并尽早开始播放）
        chunk = TTS_SR // 2
        for i in range(0, len(pcm24), chunk):
            part = pcm24[i:i + chunk]
            await session.send_to_client(
                TtsAudio.from_int16(part, response_id, self._tts_seq)
            )
            self._tts_seq += 1

        await session.send_to_client(TtsEnd(response_id=response_id))

        # 把同一份 PCM 落进参考轨，供 AEC 使用
        if session.ref_track is not None:
            at_sample = t_send + int(
                self.playback_delay_ms * session.clock.sr / 1000.0
            )
            session.ref_track.place(
                response_id, 0, pcm24.astype(np.int16), at_sample
            )
            logger.debug(
                "ref 落位 response=%s at_sample=%d（会话位置，+%dms 播放提前）len=%d",
                response_id, at_sample, self.playback_delay_ms, len(pcm24),
            )

        # 记录这句预计播完的位置（落位起点 + 音频长度）—— 见 _current_play_until
        if session.ref_track is not None:
            self._current_play_until = at_sample + int(
                len(pcm24) * session.clock.sr / TTS_SR)

        self.speaks_done += 1
        if session.metrics is not None:
            session.metrics.inc("tts_audio_s", int(len(pcm24) / TTS_SR))
        logger.info("TTS 完成 response=%s 文本 %d 字 音频 %.2fs",
                    response_id, len(act.text), len(pcm24) / TTS_SR)

    def ref_snapshot(self, session: "OrchestratorSession") -> dict:
        """参考轨状态快照（排查用）。

        生产环境"回声没消掉"的第一件事就是看这里：``read_peak`` 为 0
        说明 farend 是静音，AEC 什么都没得消。
        """
        rt = session.ref_track
        if rt is None:
            return {"enabled": False}
        lo, hi = rt.buf.written_span()
        t = session.clock.now()
        probe = rt.read(max(0, t - 1600), 1600) if t > 1600 else rt.read(0, 1600)
        return {
            "enabled": True,
            "written_span": [lo, hi],
            "now": t,
            "delay_samples": rt.delay_samples,
            "delay_ms": round(rt.delay_samples / rt.sr * 1000, 1),
            "read_peak": round(float(np.abs(probe).max()), 4),
            "active": rt.is_active(t),
        }

    # ------------------------------------------------------------------ #

    def _interrupt_current(self, session: "OrchestratorSession",
                           reason: str) -> Optional[str]:
        """打断当前播报，并让参考轨与实际播出重新对齐。**同步、原子**。

        返回被打断的 response_id（没有可打断的则 None）。

        ⚠️ **截断点不能取"当前时刻"**，也别取"起播时刻" —— 两者都会错：

        设本句的落位区间是 ``[started, end)``：
          · 取"当前时刻" → 时钟还停在**句子开头之前**（`tts.end` 到达时
            浏览器往往才刚起播），会留下整段没播的音频当参考 → AEC 去追
            一个不存在的回声。
          · 取"起播时刻" → 等于把**整句都清掉**（连已经播出去的那段也没
            了），而那段是真的有回声的 → 有回声、没 farend，照样消不掉。

        正确的是：**保留 ``[started, now)``，清掉 ``[max(now, started), end)``**。
        即截断点 = ``max(now, started)``：
          · 还没起播（now ≤ started）→ 从 started 清 → 整句清干净
          · 已播到中途（now > started）→ 从 now 清 → 已播部分保留

        参考轨清理由 `RefTrack.truncate` 保证**不误伤其他 response**
        （它只清本 response 自己落位的区间）。
        """
        rid = self._current_response_id
        if rid is None:
            return None
        now = session.clock.now()
        started = (session.ref_track.started_at(rid)
                   if session.ref_track is not None else None)
        # 保留 [started, now)，清 [max(now, started), end)
        from_sample = max(now, started) if started is not None else now

        n = 0
        if session.ref_track is not None:
            n = session.ref_track.truncate(rid, from_sample=from_sample)
        logger.info(
            "打断 %s（%s）：ref 清 %d 采样（%.0fms），截断点=%d"
            "（now=%d，起播=%s）",
            rid, reason, n, n / 16000 * 1000, from_sample, now,
            started if started is not None else "未知",
        )
        self._current_response_id = None
        self._current_play_until = 0
        self.cancels_done += 1
        return rid

    async def _cancel(self, act: Cancel, session: "OrchestratorSession") -> None:
        """中断播报（下游显式请求）。四件事原子做。"""
        rid = act.response_id or self._current_response_id
        if rid is None:
            return
        # ① 通知浏览器停止并清空播放器，同时**重置排程指针** ——
        #    否则下一个 tts.start 的 nextAt 还停在旧句末尾，新句会被排到
        #    旧句之后（实测正是这个导致新回复的回声对不上）。
        await session.send_to_client(TtsCancel(response_id=rid,
                                               reason=act.reason))
        # ② 截断参考轨（未播出的部分），并复位当前指针
        if act.response_id and act.response_id != self._current_response_id:
            # 指定了别的 response：只清那一段，不动当前句
            if session.ref_track is not None:
                session.ref_track.truncate(act.response_id)
        else:
            self._interrupt_current(session, reason=act.reason or "cancel")

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
