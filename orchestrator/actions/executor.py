"""执行下游返回的 action。

本阶段支持：
  - ``Speak``  → 调 TTS 合成 → 送浏览器 → 同时落进 RefTrack（供 AEC）
  - ``Cancel`` → **原子地**停播放器 + truncate RefTrack
  - ``SendToOmni`` / ``Emit``

``Cancel`` 的三件事必须原子：否则 AEC 会拿着没播出的音频当参考，
主动误适配去追一个不存在的回声 —— 比不给参考更糟。
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import TYPE_CHECKING, Optional

import numpy as np

from ..protocol import TtsAudio, TtsCancel, TtsDelta, TtsEnd, TtsStart
from ..downstream.interface import Cancel, Emit, SendToOmni, Speak

if TYPE_CHECKING:
    from ..session import OrchestratorSession

logger = logging.getLogger(__name__)

TTS_SR = 24000
# 播放提前量：浏览器收到 tts.start 后承诺在「现在 + 这个值」起播。
# 由服务端经 TtsStart.lead_ms / SessionReady.lead_ms 下发，前端不写死
# （写死过一次，两边漂了）。太小会让首块音频赶不上排程点，太大则加延迟。
PLAYBACK_DELAY_MS = 200
# 等浏览器 armed 承诺的上限。
#
# ⚠️ 不能随便调大：这个等待是在**音频已经发出去之后**进行的，而参考轨
#    必须赶在回声返回之前落位。可用预算是 `lead + D ≈ 400ms`（见
#    `_speak_inner` 里的时序说明），所以超时上限必须明显小于它。
#    超时说明前端没实现/回执丢了 —— 退回预测落位，不能干等。
ARM_TIMEOUT_MS = 250


class ActionExecutor:
    """把下游 action 变成实际副作用。"""

    def __init__(self, playback_delay_ms: int = PLAYBACK_DELAY_MS,
                 arm_timeout_ms: int = ARM_TIMEOUT_MS) -> None:
        self.playback_delay_ms = playback_delay_ms
        self.arm_timeout_ms = arm_timeout_ms
        self._current_response_id: Optional[str] = None
        # 当前这句**预计播完**的会话采样位置。用来判断新回复要不要打断旧句：
        # 它不能靠 playback.started/ended 判断 —— 那些只反映"音频送达"，
        # 不反映"播完"（整段是排程播放的，送达时可能才刚起播）。
        self._current_play_until: int = 0
        #: 当前这句的**原文**。用于「同一条模板被下发了两次」的去重 ——
        #: 见 `_is_duplicate_speak`。
        self._current_text: str = ""
        self._tts_seq = 0
        self.speaks_done = 0
        self.cancels_done = 0
        # 在飞的合成数：会话收尾要等它归零，否则 close() 会打断 RPC
        self._speak_inflight = 0
        # 播放起点的墙钟锚（用于没有回执时的退化路径）
        self._playback_anchor_ctx: Optional[float] = None
        # ---- 流式合成状态 ----
        # 当前开着的 TTS 流与它的消费任务。同一时刻只允许一路：
        # 新回复开始前会把旧的 abort 掉（见 _open_stream）。
        self._stream = None                       # TtsStream | None
        self._stream_id: str = ""                 # 当前流的 stream_id
        self._stream_task: Optional[asyncio.Task] = None
        self._stream_text: list = []              # 累计全文（收尾时给 tts.end）
        #: 当前流是否已发过结束信号。发过之后不能再 feed（见 _speak_stream）
        self._stream_ended = False

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

    def _is_duplicate_speak(self, act: Speak) -> bool:
        """这条 Speak 是不是"**正在播的那句**"的重复？—— 是则跳过。

        ## 为什么会有重复

        IC 判 ``GREET`` / ``UTTER`` 时会做**两件事**（见 interactioncore 的
        ``runtime.py._notify_sinks``）：

          ① 把文案塞进 Action 一起返回 → 编排服务 `downstream` 收到后
             `return [Speak(text)]` 播一遍
          ② **另外**调 ``expression_sink.play_template()`` →
             ``POST /v1/speak`` 再播一遍

        ⚠️ ② **先于** ① 到达（``_notify_sinks`` 在 `return action` 之前调用），
        所以实际顺序是：② 开始播 → ① 到达 → 打断②重新播 → 听感是
        **同一句被说了两遍/带一次重启感**。

        这个重复在**今天下午之前一直看不见** —— 因为那时 IC 的 HTTP 通路
        被证书问题切断（② 静默失败），只播了 ①，**恰好是对的**。
        证书修好后 ② 恢复，重复才暴露出来。

        ## 为什么是"比对正在播的"而不是"记住播过什么"

        ⚠️ **不能**做成"同样的文本播过就跳过" —— 那样会吞掉合法的
        **重复迎宾**：会话 END 后（``clear_session_on_end`` 会清
        ``greet_spoken``），人再回来时 IC 会合理地**再判一次 GREET**，
        文案与上次**完全相同**。按历史去重会把这次迎宾吃掉。

        所以判据限定为「与**当前正在播的这句**相同」—— 只挡"同一时刻的
        重复下发"，对"END 之后再迎宾"没有影响（那时 `_current_text` 已被
        `_interrupt_current` 或新一轮 Speak 覆盖/清空）。

        ## 已知的边界

        若 IC 真的**连着**下发两条一模一样的 Speak（不是①②重复，而是
        有意说两遍），这里会吞掉第二条。当前 IC 的 GREET/UTTER 都有消费
        保护（``greet_spoken`` / ``consumed_turn``），**不会**发生这种情况。
        """
        text = (act.text or "").strip()
        if not text:
            return False
        playing = (self._current_text or "").strip()
        if playing and text == playing:
            logger.info("跳过重复播报（与正在播的相同，%d 字）：%s",
                        len(text), text[:40])
            return True
        return False

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
        if self._is_duplicate_speak(act):
            return
        # 流式增量：只把文本喂进流就返回。**绝不能在这里等合成** ——
        # run_downstream 是串行 await 的，占住它会让后续 delta 进不来，
        # 而本次合成正等着那些 delta → 自锁。
        if act.stream_id:
            await self._speak_stream(act, session)
            return
        self._speak_inflight += 1
        try:
            await self._speak_inner(act, session)
        finally:
            self._speak_inflight -= 1

    # ------------------------------------------------------------------ #
    #  流式合成
    # ------------------------------------------------------------------ #

    async def _speak_stream(self, act: Speak,
                            session: "OrchestratorSession") -> None:
        """处理一个流式增量 Speak。**必须快速返回**（见上面的死锁说明）。"""
        if not getattr(session.tts, "supports_streaming", False):
            # 客户端不支持流式：退化成整段合成（每个增量各合成一次，
            # 会有点碎，但不会静默丢失）
            logger.warning("TTS 客户端不支持流式，增量文本退化为整段合成")
            return
        if self._stream is None or self._stream_id != act.stream_id:
            if self._stream is not None:
                # stream_id 变了（旧流没收尾就来了新的）：把旧的收干净
                await self._close_stream(session, reason="superseded")
            ok = await self._open_stream(act, session)
            if not ok:
                return
        if act.text:
            # ⚠️ 已 end 的流**绝不能再喂** —— 喂了文本永远等不到对应的音频
            # （服务端的输入序列已经结束），表现为这一轮的文本静默丢失。
            # 正常路径下 stream_id 每轮唯一，不会走到这；这里是兜底，
            # 防的是下游stream_id 复用/错配。
            if self._stream_ended:
                logger.warning(
                    "流 %s 已收尾，丢弃迟到的文本（%d 字）—— 多半是 "
                    "stream_id 被复用了", act.stream_id, len(act.text))
            else:
                self._stream_text.append(act.text)
                # 先推给 UI，再喂 TTS —— 字幕要**跟着 LLM 生成**走，
                # 而不是等 `tts.end`（那样整轮说完才显示，等于没有流式）
                rid = self._current_response_id
                if rid:
                    try:
                        await session.send_to_client(
                            TtsDelta(response_id=rid, text=act.text))
                    except Exception as exc:  # noqa: BLE001
                        logger.debug("发送 tts.delta 失败（忽略）: %s", exc)
                self._stream.feed(act.text)
        if act.is_final and not self._stream_ended:
            self._stream_ended = True
            self._stream.end()

    async def _open_stream(self, act: Speak,
                           session: "OrchestratorSession") -> bool:
        """开一路新的 TTS 流，并起后台任务消费它的音频。"""
        # 打断上一句：与整段合成同一条路径（浏览器掐断 + 参考轨截断）
        await self._interrupt_current(session, reason="superseded")
        # 上一句的流任务要等它退干净，否则旧音频会串进新的消费循环
        await self._await_stream_task()

        response_id = uuid.uuid4().hex[:8]
        self._current_response_id = response_id
        self._tts_seq = 0
        self._stream_text = []
        self._current_text = act.text or ""

        # ⚠️ 流式下此刻还不知道全文，text 只能给空 —— 字幕靠 tts.end 补
        await session.send_to_client(TtsStart(
            response_id=response_id, text="", sample_rate=TTS_SR,
            lead_ms=self.playback_delay_ms,
        ))
        arm = self._arm(session, response_id)
        stream = session.tts.open_stream(
            tts_type=act.tts_type, speaker_id=act.speaker_id,
            speaker_vector_b64=act.speaker_vector_b64,
        )
        if stream is None:
            self._release_arm(session, response_id)
            logger.warning("TTS 流式会话开启失败（服务已关停？）")
            return False
        self._stream = stream
        self._stream_id = act.stream_id
        self._stream_ended = False
        self._speak_inflight += 1
        self._stream_task = asyncio.create_task(
            self._consume_stream(session, response_id, arm, stream))
        session.stats["tts_streams"] = session.stats.get("tts_streams", 0) + 1
        return True

    async def _consume_stream(self, session: "OrchestratorSession",
                              response_id: str, arm, stream) -> None:
        """后台任务：收流式音频 → 下发浏览器 → **逐块**落位参考轨。

        与整段合成的关键差异：参考轨不再是「整段一次 place」，而是每收到
        一块就续着写。这是安全的 —— RefTrack 的 ``StatefulResampler`` 跨
        ``place()`` 保持相位，逐块落位与整段落位**逐采样一致**（已实测）。
        """
        seq = 0
        at_sample: Optional[int] = None
        n_chunks = 0
        t0 = time.monotonic()
        try:
            async for pcm24 in stream.audio_iter():
                if session.closed:
                    break
                await session.send_to_client(
                    TtsAudio.from_int16(pcm24, response_id, seq))
                seq += 1
                n_chunks += 1
                if at_sample is None:
                    # 第一块到达才要锚点 —— 浏览器的 armed 承诺是在第一块
                    # 音频到达时给出的（见 PcmPlayer._armNow 的说明）
                    at_sample = await self._resolve_play_at(
                        session, response_id, arm)
                if session.ref_track is not None and at_sample is not None:
                    ch = session.ref_track.place(
                        response_id, seq, pcm24, at_sample)
                    # 用 place 返回的长度累加（重采样有取整，不能自己乘比例）
                    at_sample = ch.t1
                    self._current_play_until = ch.t1
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.warning("流式合成消费异常: %s", exc)
        finally:
            # 收尾：发 tts.end（带全文，字幕用），释放 inflight
            try:
                if not session.closed:
                    await session.send_to_client(TtsEnd(
                        response_id=response_id, text="".join(self._stream_text)))
            except Exception as exc:  # noqa: BLE001
                logger.debug("发送 tts.end 失败（忽略）: %s", exc)
            self._speak_inflight -= 1
            self.speaks_done += 1
            logger.info(
                "TTS 流式完成: %s 共 %d 块 / %d 字，耗时 %.0fms，首块 %.0fms%s",
                response_id, n_chunks, len("".join(self._stream_text)),
                (time.monotonic() - t0) * 1000,
                stream.first_chunk_s * 1000,
                f"，错误={stream.error}" if stream.error else "",
            )

    async def _close_stream(self, session: "OrchestratorSession",
                            reason: str = "") -> None:
        """结束当前流（收尾或打断），并等消费任务退出。"""
        st, self._stream = self._stream, None
        self._stream_id = ""
        self._stream_ended = False
        if st is not None:
            st.end()          # 幂等；已 end 过就什么都不做
            await st.abort()  # 保证线程退出、音频不再投递
        await self._await_stream_task()
        if reason:
            logger.debug("TTS 流已结束（%s）", reason)

    async def _await_stream_task(self) -> None:
        """等消费任务退出。**不能直接 cancel** —— 那会跳过 tts.end 下发。"""
        task, self._stream_task = self._stream_task, None
        if task is None or task.done():
            return
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=5.0)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            task.cancel()
            logger.debug("流式消费任务未在 5s 内退出，已取消")

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
        # 所以打断必须**由服务端在开新句前主动做**，且两件事都要做：
        #   ① 给浏览器发 tts.cancel（前端据此**掐断已排程的音频源**）
        #   ② 截断参考轨
        # ⚠️ 早先只做了② —— 参考轨清对了，但浏览器那边**照旧播完**，
        #    表现为"打断后当前语音不停，然后 ASR 开始识别还在播的音频"。
        await self._interrupt_current(session, reason="superseded")

        response_id = uuid.uuid4().hex[:8]
        self._current_response_id = response_id
        self._tts_seq = 0
        self._current_text = act.text or ""

        # ⚠️ tts.start 必须在**合成之前**发 —— 浏览器要据此承诺起播时刻，
        #    而这段承诺的往返正好藏在 TTS 的合成耗时里（实测首帧 508ms）。
        #    放到合成之后发就会白加一个网络往返。
        await session.send_to_client(TtsStart(
            response_id=response_id, text=act.text, sample_rate=TTS_SR,
            lead_ms=self.playback_delay_ms,
        ))
        arm = self._arm(session, response_id)

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
            self._release_arm(session, response_id)
            await session.send_to_client(TtsEnd(response_id=response_id))
            return

        # ---- 先发音频，再等浏览器的「承诺」----
        #
        # ⚠️ **顺序不能反**。浏览器的起播时刻是在**第一块音频到达时**才
        #    算出来的（`ctx.currentTime + 提前量`）。早先让它收到 tts.start
        #    就承诺 —— 而 tts.start 在合成之前发，等音频真正到达时承诺的
        #    时刻已经过去 340ms，浏览器只能"马上播"，参考轨却仍按那个过期
        #    时刻落位，结果**比实际回声早 340ms**（真机实测：回声完全消不掉）。
        #    所以必须先把音频推出去，承诺才有意义。
        chunk = TTS_SR // 2
        for i in range(0, len(pcm24), chunk):
            part = pcm24[i:i + chunk]
            await session.send_to_client(
                TtsAudio.from_int16(part, response_id, self._tts_seq)
            )
            self._tts_seq += 1
            if arm is not None and i == 0:
                # ⚠️ 首块发完立刻让出事件循环，让 armed 回执能回来。
                #    否则整段音频（长回复可能上百块）发完才轮到处理回执，
                #    白白晚一个 RTT —— 而参考轨必须赶在回声之前落位
                #    （回声在承诺时刻 + D 之后才到，D 是几百毫秒量级）。
                await asyncio.sleep(0)

        await session.send_to_client(TtsEnd(response_id=response_id))

        # ---- 参考轨落位：用浏览器承诺的时刻，不猜 ----
        #
        # 时序余量（决定这个"先发后等"能不能成立）：
        #   承诺时刻 = 浏览器收到首块音频的时刻 + lead（默认 200ms）
        #   回声到达麦克风 = 承诺时刻 + D（设备声学延迟，约 200ms）
        # 我们要在**回声到达之前**把参考轨写好，所以可用的预算是
        # `lead + D ≈ 400ms`，而这里只花了「首块 → ack 回来」一个 RTT
        # （局域网几十毫秒）。余量充足。
        # 若 lead 被调得很小、或 D 极小、或 RTT 异常大，参考轨就会写晚 ——
        # 表现为该句开头一小段回声没有参考。`started` 回执里的偏差会暴露。
        at_sample = await self._resolve_play_at(session, response_id, arm)
        if session.ref_track is not None:
            session.ref_track.place(
                response_id, 0, pcm24.astype(np.int16), at_sample
            )
            # 记录这句预计播完的位置（落位起点 + 音频长度）
            self._current_play_until = at_sample + int(
                len(pcm24) * session.clock.sr / TTS_SR)

        self.speaks_done += 1
        if session.metrics is not None:
            session.metrics.inc("tts_audio_s", int(len(pcm24) / TTS_SR))
        logger.info("TTS 完成 response=%s 文本 %d 字 音频 %.2fs 落位于 %d"
                    "（%.2fs，来源=%s）",
                    response_id, len(act.text), len(pcm24) / TTS_SR,
                    at_sample, at_sample / session.clock.sr, session.anchor_source)

    # ------------------------------------------------------------------ #

    # 打断时在估算的播出位置之后**多留**的余量（采样）。见
    # `_interrupt_current` 里对偏差方向的说明：宁多勿少。
    INTERRUPT_MARGIN_SAMPLES = int(0.30 * 16000)

    def _estimate_played_to(self, session: "OrchestratorSession",
                            fallback: int) -> int:
        """估计"已经播到哪了"（会话采样位置），偏小。

        有锚点时用 ``clock.ctx_now_estimate()`` 换算（反映浏览器的真实
        播放位置）；没有锚点（旧客户端 / context 挂起）就退回
        ``clock.now()`` —— 那也是偏小的，方向是对的。
        """
        ctx = None
        try:
            ctx = session.clock.ctx_now_estimate()
        except Exception:  # noqa: BLE001
            ctx = None
        if ctx is None:
            return fallback
        at = session.clock.ctx_to_sample(ctx)
        return fallback if at is None else at

    def _arm(self, session: "OrchestratorSession", response_id: str):
        """注册 armed 等待点；客户端没有该能力位时返回 None（不等）。

        「能力位」这层开关让旧客户端与所有存量测试完全走原路径，零回归。
        """
        if "playback_anchor" not in getattr(session, "client_caps", ()):
            return None
        return session.arm_waiter(response_id)

    @staticmethod
    def _release_arm(session: "OrchestratorSession", response_id: str) -> None:
        try:
            session.release_waiter(response_id)
        except Exception:  # noqa: BLE001
            pass

    async def _resolve_play_at(self, session: "OrchestratorSession",
                               response_id: str, arm) -> int:
        """算出参考轨该落位的会话采样位置。

        优先用浏览器的 armed 承诺（精确）；拿不到则退回预测（= 旧行为，
        但会在 stats 里标成 ``predicted`` 并告警）。
        """
        if arm is not None:
            at = await session.wait_armed(
                response_id, self.arm_timeout_ms / 1000.0)
            session.release_waiter(response_id)
            if at is not None:
                return at
            session.anchor_source = "predicted"
            logger.warning(
                "未收到 %s 的 armed 承诺（%dms 超时）—— 退回预测落位，"
                "参考轨会有网络/浏览器抖动级别的错位，回声对齐不准",
                response_id, self.arm_timeout_ms,
            )
        else:
            session.anchor_source = "predicted"
        return session.clock.now() + int(
            self.playback_delay_ms * session.clock.sr / 1000.0)

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

    async def _interrupt_current(self, session: "OrchestratorSession",
                                 reason: str,
                                 notify: bool = True) -> Optional[str]:
        """打断当前播报：**通知浏览器停** + 截断参考轨，让两者重新对齐。

        返回被打断的 response_id（没有可打断的则 None）。

        ⚠️ 两件事缺一不可：
          · 只截断参考轨不发 `tts.cancel` → 浏览器照旧把旧句播完，
            而参考轨已经清空 → 还在播的音频没有 farend → ASR 全识别到
            （真机实测现象："打断后当前语音不停，然后开始识别正在播的音频"）
          · 只发 `tts.cancel` 不截断参考轨 → 没播出的音频留在轨上，
            AEC 去追一个不存在的回声，比不给参考更糟

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
        # ⓪ 流式：先把流掐掉，**再**做下面那些簿记。
        #    顺序很重要 —— 流还在投音频的话，截断完之后它又会往轨上写，
        #    参考轨就带着"已经不该播的音频"继续跑（AEC 追不存在的回声）。
        #    这里不 await 消费任务退出（那是 _await_stream_task 的事，
        #    由 _open_stream 在开新流前做）—— 只保证流立刻停止产出。
        if self._stream is not None:
            st, self._stream = self._stream, None
            self._stream_id = ""
            self._stream_ended = False
            st.end()
            await st.abort()
            logger.debug("打断时已中止 TTS 流")

        rid = self._current_response_id
        if rid is None:
            return None
        # ① 先让浏览器停 —— 必须在截断之前发，前端收到后掐断已排程的源，
        #    这样"参考轨清掉的那段"与"实际没播出的那段"才是同一段。
        if notify:
            try:
                await session.send_to_client(
                    TtsCancel(response_id=rid, reason=reason))
            except Exception as exc:  # noqa: BLE001
                logger.warning("发送 tts.cancel 失败: %s", exc)
        now = session.clock.now()
        started = (session.ref_track.started_at(rid)
                   if session.ref_track is not None else None)
        # 临时截断点 =「已经播到哪」的一个**偏小**估计，保留 [started, cut)。
        #
        # ⚠️ 这里的偏差方向很关键，别搞反：`resize`（前端回执到达后的精确
        # 校正）**只能缩短**参考轨（ref_track.py 的 `cut = lo + keep` 只会
        # 往后清）。所以临时截断必须**少清** —— 留多了能由 resize 精确剪掉，
        # **留少了永远补不回来**，那就是"有回声、没 farend"，AEC 直接失效。
        #
        # 校准后的做法：按锚点估算浏览器此刻的播出位置（比 `clock.now()`
        # 准得多 —— 后者是"已 ingest 的采样数"，系统性落后 100~300ms），
        # 再往后**多留** 300ms 余量。对 200ms 的容忍窗来说这点尾部残留
        # 无所谓（resize 会剪掉，且 AEC 对尾部几十毫秒的多余参考不敏感），
        # 而少留就是硬伤。
        cut = self._estimate_played_to(session, now) + self.INTERRUPT_MARGIN_SAMPLES
        if started is not None:
            cut = max(cut, started)      # 还没起播 → 整句清干净
        from_sample = cut

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
        # ⚠️ **必须一起清** —— 它是 `_is_duplicate_speak` 的判据。
        #    不清的话：打断后 IC 再下发同一句（比如 END 后人回来重新迎宾、
        #    文案相同）会被误当成"正在播的那句"而**吞掉**。
        self._current_text = ""
        self.cancels_done += 1
        # 打断 = 立刻不再出声，**不等浏览器回执**（那要绕一圈才回来）。
        # IC 的 SOP 07 靠这个判"用户抢话后已经停播"。
        #
        # ⚠️ 这是 `playback_active` 的**唯一例外**：常规路径由浏览器按真实
        #    播放进度报 playing/stopped（见 session.on_playback_receipt），
        #    但打断是**动作**而不是观察 —— 我们主动把源掐了，不必等回执。
        #    浏览器随后也会报一次 stopped，IC 那边是幂等覆盖，无害。
        try:
            session._notify_playback_active(False)
        except Exception:  # noqa: BLE001
            pass
        return rid

    async def _cancel(self, act: Cancel, session: "OrchestratorSession") -> None:
        """中断播报（下游显式请求）。通知浏览器 + 截断参考轨，两者原子做。"""
        if act.response_id and act.response_id != self._current_response_id:
            # 指定了**别的** response（不是当前这句）：只清那一段，不动当前句
            await session.send_to_client(TtsCancel(response_id=act.response_id,
                                                   reason=act.reason))
            if session.ref_track is not None:
                session.ref_track.truncate(act.response_id)
            self.cancels_done += 1
            return
        # 打断当前正在播的这句（_interrupt_current 内部会发 tts.cancel）
        await self._interrupt_current(session,
                                      reason=act.reason or "cancel")

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
