"""会话编排 —— 把所有流水线串在一条时间轴上。

一个 ``OrchestratorSession`` 对应一路浏览器连接。它拥有：

  · SampleClock       —— 唯一时间基准（mic ingest 是唯一写入者）
  · AecClient         —— 音频清洗
  · AsrClient         —— 语音识别（结果给 downstream）
  · OmniClient        —— 音视频理解（原生音频 + 1fps 视频）
  · RefTrack          —— TTS 参考轨（AEC 的 farend）
  · TtsClient         —— 文本转语音
  · FaceWorker        —— 人脸/唇动（独立线程）
  · Downstream        —— 决策接口（本阶段用 passthrough 桩）

**音频链路顺序**：``mic → AEC → {ASR, OmniLLM}``。
AEC 只调一次，扇出在其后；**绝不两路各自调 AEC**（它是有状态流式算法，
状态会被破坏）。
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from typing import Any, Deque, Dict, List, Optional

import numpy as np

from .clock import SR, AudioFrame, SampleClock
from .protocol import MIC_CHUNK

logger = logging.getLogger(__name__)


class OrchestratorSession:
    """一路会话的编排器。

    ⚠️ 本阶段的实现边界：downstream 用 deterministic 桩（把 ASR 最终结果
    直接作为 Speak 文本），让「说话 → ASR → TTS → 播放 → AEC」能端到端
    验证。Policy/Agent 接入时只替换 downstream 实例。
    """

    def __init__(self, session_id: str, config: Dict[str, Any]) -> None:
        self.session_id = session_id
        self.config = config
        self.clock = SampleClock(SR)
        self.closed = False
        self.error: Optional[str] = None

        # 各组件由 main.py 注入（便于测试时替换）
        self.aec = None
        self.asr = None
        self.omni = None
        self.tts = None
        self.ref_track = None
        self.delay_tracker = None
        self.face_worker = None
        self.downstream = None
        self.executor = None
        self.send_to_client = None   # Callable[[Any], Awaitable[None]]

        # 音频扇出队列（AEC 输出 → ASR / OmniLLM）
        self._aec_out_q: asyncio.Queue = asyncio.Queue(maxsize=64)

        # downstream 事件队列（有界，满时按优先级丢弃）
        self._down_q: asyncio.Queue = asyncio.Queue(maxsize=512)

        # UI 显示消息队列（纯展示，满则丢；不影响控制流正确性）
        self._display_q: asyncio.Queue = asyncio.Queue(maxsize=128)

        # 人脸信号队列（人脸线程 → asyncio，解耦同步回调）
        self._face_q: asyncio.Queue = asyncio.Queue(maxsize=256)

        # 统计
        self.stats: Dict[str, Any] = {
            "audio_chunks_in": 0,
            "audio_samples_in": 0,
            "video_face_frames": 0,
            "video_omni_frames": 0,
            "aec_segments_out": 0,
            "aec_samples_out": 0,
            "down_dropped": 0,
            "ticks": 0,
        }
        self._t_wall0 = time.monotonic()
        # 供 barge-in 用的原始 mic 旁路（不经过 AEC 的环形缓冲）
        self._raw_recent: Deque[float] = deque(maxlen=8)

    # ------------------------------------------------------------------ #
    #  浏览器侧输入
    # ------------------------------------------------------------------ #

    async def on_audio(self, x: np.ndarray, t_ms: int = 0) -> None:
        """收到一段麦克风音频（float32 (1,T)，16kHz）。

        mic ingest 是**唯一**推进时钟的地方。时钟因此跟踪真实音频流，
        所有其他流都在它上面表达。
        """
        if self.closed:
            return
        if x.shape[1] != MIC_CHUNK:
            # 容忍非标准块：重采样/补齐，但绝不崩
            x = self._normalize_chunk(x)
        frame = self.clock.frame_of(x)
        self.stats["audio_chunks_in"] += 1
        self.stats["audio_samples_in"] += frame.n_samples

        # 原始 mic 旁路：barge-in 检测用，零额外延迟
        self._raw_recent.append(float(np.sqrt(np.mean(x ** 2))))

        # 送去清洗：有 AEC 走 AEC，否则直接扇出。
        # AEC 是**可选**的清洗环节，不是链路的一环 —— 缺它不影响正确性。
        if self.aec is not None:
            ref = self._ref_for(frame)
            await self.aec.push(frame.data, ref)
        else:
            await self._fanout_direct(frame.data)

    def _normalize_chunk(self, x: np.ndarray) -> np.ndarray:
        """把任意长度的块规整成 MIC_CHUNK 长度（必要时应改造为环形缓冲）。"""
        n = x.shape[1]
        if n > MIC_CHUNK:
            return x[:, :MIC_CHUNK]
        pad = np.zeros((x.shape[0], MIC_CHUNK - n), dtype=np.float32)
        return np.concatenate([x, pad], axis=1)

    async def _fanout_direct(self, x: np.ndarray) -> None:
        """无 AEC 时的直接扇出（AEC 未启用，或连接失败降级）。

        架构上 ASR/OmniLLM 的数据源不该强依赖 AEC 存在 —— AEC 是**可选
        的清洗环节**，不是链路的一环。缺少它时链路的正确性不受影响，
        只是回声没被消除。
        """
        seg = x.reshape(-1)
        if self.asr is not None:
            try:
                await self.asr.push(seg)
            except Exception as exc:  # noqa: BLE001
                logger.warning("ASR push 失败: %s", exc)
        if self.omni is not None:
            try:
                await self.omni.push_audio(seg)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Omni push 失败: %s", exc)

    def _ref_for(self, frame: AudioFrame) -> np.ndarray:
        """取该帧对应的 AEC 参考信号。

        决策 3：**始终传 farend**（空闲传等长静音），理由：
          1. 分支不切换 → 时间轴无跳变，时间基准是常量偏移
          2. 给静音是物理上诚实的 —— 确实没在播
        """
        if self.ref_track is None:
            return np.zeros_like(frame.data)
        ref = self.ref_track.read(frame.t0, frame.n_samples)
        return ref.reshape(1, -1).astype(np.float32)

    async def on_video_face(self, jpeg: bytes, t_ms: int = 0) -> None:
        """收到人脸用视频帧（25fps）。**绝不阻塞音频路径。**"""
        if self.closed:
            return
        self.stats["video_face_frames"] += 1
        if self.face_worker is not None:
            # put_nowait：队列满则丢最旧帧（新帧对唇动状态更有价值）
            self.face_worker.offer(jpeg, self.clock.now())

    async def on_video_omni(self, jpeg: bytes, t_ms: int = 0) -> None:
        """收到 OmniLLM 用视频帧（1fps）。"""
        if self.closed:
            return
        self.stats["video_omni_frames"] += 1
        if self.omni is not None:
            self.omni.offer_frame(jpeg)

    async def on_playback_receipt(self, response_id: str,
                                  phase: str, ctx_time: float,
                                  seq: int = 0) -> None:
        """播放回执 —— 驱动 AEC 参考轨的时钟。"""
        if self.closed or self.ref_track is None:
            return
        if phase == "started" and self.ref_track._anchor_ctx is None:
            # 首个回执建立锚点：ctx_time ↔ 当前采样位置
            self.ref_track.set_anchor(ctx_time, self.clock.now())
        elif phase in ("ended", "cancelled"):
            # ⚠️ 取消/结束时必须截断 ref 尾部：否则 AEC 会拿着没播出的
            # 音频当参考，主动误适配去追一个不存在的回声 —— 比不给更糟
            n = self.ref_track.truncate(response_id, from_ctx_time=ctx_time)
            if n:
                logger.debug("截断 ref %d 采样（%s）", n, phase)
        self._post_downstream_playback(response_id, phase, ctx_time, seq)

    def _post_downstream_playback(self, response_id: str, phase: str,
                                  ctx_time: float, seq: int) -> None:
        from .downstream.interface import PlaybackReceipt
        self.post_downstream(PlaybackReceipt(
            t=self.clock.now(), response_id=response_id,
            phase=phase, ctx_time=ctx_time, seq=seq,
        ))

    # ------------------------------------------------------------------ #
    #  AEC 输出扇出
    # ------------------------------------------------------------------ #

    async def run_aec_recv(self) -> None:
        """AEC 接收循环：把清洗后的音频入队，供 ASR / OmniLLM 消费。"""
        if self.aec is None:
            return
        loop = asyncio.get_running_loop()

        def on_audio(seg: np.ndarray) -> None:
            self.stats["aec_segments_out"] += 1
            self.stats["aec_samples_out"] += int(np.asarray(seg).size)
            # 从非 asyncio 上下文回投
            try:
                self._aec_out_q.put_nowait(np.asarray(seg).reshape(-1))
            except asyncio.QueueFull:
                self.stats["down_dropped"] += 1

        # AecClient.recv_loop 是 async，直接在 loop 里跑
        await self.aec.recv_loop(on_audio)

    async def run_asr_recv(self) -> None:
        """ASR 接收循环：原始 JSON → downstream 事件。

        时间戳换算：ASR 的毫秒时间戳原点是**其流起点**，``asr.stream_t0``
        标注该起点在会话采样轴上的位置。
        """
        if self.asr is None:
            return
        from .asr.client import parse_confidence, parse_final, parse_turnsense
        from .downstream.interface import (
            AsrFinal, AsrPartial, AsrTurnSense,
        )
        from .protocol import AsrDisplay

        def on_message(msg: dict) -> None:
            mode = str(msg.get("mode") or "")
            if mode == "turnsense":
                ts = parse_turnsense(msg)
                self.post_downstream(AsrTurnSense(
                    t=self.clock.now(),
                    label=ts["label"] or "invalid",
                    probabilities=ts["probabilities"],
                    segment_start_ms=ts["segment_start_ms"],
                    segment_end_ms=ts["segment_end_ms"],
                    speech_duration_s=ts["speech_duration_s"],
                ))
                return

            if mode == "2pass-online":
                text = msg.get("text", "")
                if text:
                    self.post_downstream(AsrPartial(
                        t=self.clock.now(), text=text,
                        confidence=parse_confidence(msg),
                        segment_id=self.asr.partials,
                    ))
                    self._send_display(AsrDisplay(
                        phase="partial", text=text,
                        t_ms=int(self.clock.seconds() * 1000),
                    ))
                return

            # 2pass-offline（最终结果）
            fin = parse_final(msg)
            t0 = self.asr.ms_to_sample(fin["start_ms"])
            t1 = self.asr.ms_to_sample(fin["end_ms"])
            if t1 <= t0:
                t1 = max(t0 + 1, self.clock.now())
            self.post_downstream(AsrFinal(
                t0=t0, t1=t1, text=fin["text"],
                confidence=fin["confidence"],
                tokens=fin["tokens"],
                token_times_ms=fin["token_times"],
                is_final=fin["is_final"],
            ))
            if fin["text"]:
                self._send_display(AsrDisplay(
                    phase="final", text=fin["text"],
                    t_ms=int(self.clock.seconds() * 1000),
                ))

        await self.asr.recv_loop(on_message)

    def _send_display(self, msg) -> None:
        """投递纯 UI 消息（不参与控制流）。

        ASR 的回调是同步的，不能在里面 await；入队后由 ``run_display``
        异步取出发送。队列满则丢弃 UI 消息（它不影响正确性）。
        """
        try:
            self._display_q.put_nowait(msg)
        except asyncio.QueueFull:
            self.stats["display_dropped"] = self.stats.get("display_dropped", 0) + 1

    async def run_face_signals(self) -> None:
        """把人脸线程的回调转成 downstream 事件 + UI 消息。

        ``FaceWorker`` 在专用线程里跑，回调是同步的 —— 这里用队列解耦，
        绝不阻塞人脸线程。
        """
        if self.face_worker is None:
            return
        from .downstream.interface import (
            FaceIdentity, FaceLipState, FaceWake,
        )
        from .protocol import FaceDisplay

        while not self.closed:
            try:
                item = await asyncio.wait_for(self._face_q.get(), timeout=0.5)
            except asyncio.TimeoutError:
                continue
            kind, ev = item
            if kind == "wake":
                self.post_downstream(FaceWake(
                    t=ev.t, phase=ev.phase, track_id=ev.track_id,
                    dwell_ms=ev.dwell_ms, mean_confidence=ev.mean_confidence,
                ))
                self._send_display(FaceDisplay(tracks=[{
                    "phase": ev.phase, "score": round(ev.mean_confidence, 3),
                    "box": ev.box,
                }]))
            elif kind == "lip":
                self.post_downstream(FaceLipState(
                    t0=ev.t0, t1=ev.t1, track_id=ev.track_id,
                    speaking=ev.speaking, lip_state=ev.lip_state,
                    confidence=ev.confidence,
                ))
            elif kind == "identity":
                self.post_downstream(FaceIdentity(
                    t=ev.t, track_id=0, person_id=ev.person_id,
                    uid=ev.uid, name=ev.name, similarity=ev.similarity,
                    is_enrolled=ev.is_enrolled,
                ))
                self._send_display(FaceDisplay(identity={
                    "name": ev.name, "uid": ev.uid,
                    "similarity": round(ev.similarity, 3),
                    "enrolled": ev.is_enrolled,
                }))

    def _face_cb(self, kind: str):
        """构造人脸线程用的同步回调（入队，不阻塞）。"""
        def cb(ev) -> None:
            try:
                self._face_q.put_nowait((kind, ev))
            except asyncio.QueueFull:
                self.stats["face_dropped"] = self.stats.get("face_dropped", 0) + 1
        return cb

    async def run_display(self) -> None:
        """UI 消息发送循环。"""
        while not self.closed:
            try:
                msg = await asyncio.wait_for(self._display_q.get(), timeout=0.5)
            except asyncio.TimeoutError:
                continue
            if self.send_to_client is not None:
                try:
                    await self.send_to_client(msg)
                except Exception as exc:  # noqa: BLE001
                    logger.debug("发送 UI 消息失败: %s", exc)

    async def run_fanout(self) -> None:
        """把 AEC 输出扇出给 ASR 与 OmniLLM。

        **AEC 只调一次，扇出在其后** —— 绝不两路各自调 AEC。
        """
        while not self.closed:
            try:
                seg = await asyncio.wait_for(self._aec_out_q.get(), timeout=0.5)
            except asyncio.TimeoutError:
                continue
            if self.asr is not None:
                try:
                    await self.asr.push(seg)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("ASR push 失败: %s", exc)
            if self.omni is not None:
                try:
                    await self.omni.push_audio(seg)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Omni push 失败: %s", exc)

    # ------------------------------------------------------------------ #
    #  downstream
    # ------------------------------------------------------------------ #

    def post_downstream(self, ev) -> None:
        """投递事件给下游。满时丢弃并计数（本阶段用无界策略：丢最旧）。"""
        try:
            self._down_q.put_nowait(ev)
        except asyncio.QueueFull:
            try:
                self._down_q.get_nowait()   # 丢最旧
                self._down_q.put_nowait(ev)
            except Exception:  # noqa: BLE001
                pass
            self.stats["down_dropped"] += 1

    async def run_downstream(self) -> None:
        """下游事件循环：取事件 → 调 downstream → 执行返回的 action。"""
        if self.downstream is None:
            return
        while not self.closed:
            try:
                ev = await asyncio.wait_for(self._down_q.get(), timeout=0.5)
            except asyncio.TimeoutError:
                continue
            try:
                actions = await self.downstream.on_event(ev)
            except Exception as exc:  # noqa: BLE001
                logger.exception("downstream.on_event 异常: %s", exc)
                continue
            for act in actions or []:
                await self.execute_action(act)

    async def execute_action(self, act) -> None:
        """执行下游返回的 action。"""
        if self.executor is not None:
            await self.executor.execute(act, self)
        else:
            logger.debug("无 executor，忽略 action: %s", act)

    async def run_tick(self, interval: float = 0.05) -> None:
        """心跳循环。下游在此实现超时逻辑。"""
        from .downstream.interface import Tick
        while not self.closed:
            await asyncio.sleep(interval)
            self.stats["ticks"] += 1
            self.post_downstream(Tick(t=self.clock.now()))

    # ------------------------------------------------------------------ #
    #  生命周期
    # ------------------------------------------------------------------ #

    async def start(self) -> None:
        if self.downstream is not None:
            from .downstream.interface import SessionContext
            await self.downstream.on_session_start(
                SessionContext(session_id=self.session_id, sample_rate=SR,
                               config=self.config)
            )
        if self.face_worker is not None:
            self.face_worker.start()

    async def drain(self, reason: str = "client_stop",
                    asr_timeout: float = 15.0) -> None:
        """优雅收尾：把残余数据推完、等最终结果。

        与 ``close()`` 分离的原因：收尾**可以慢**（等 ASR 跑完最终识别、
        等 OmniLLM 吐完残余），但 ``close()`` **必须快** —— 它会阻塞
        WS 处理协程的清理，被 cancel 会丢数据。
        """
        if self.closed:
            return
        logger.info("会话 %s 开始收尾（%s）", self.session_id, reason)
        # 顺序有讲究：先 drain AEC（它会把尾部窗口推给 ASR），
        # 再 drain ASR（吃下尾部后才会吐最终结果）。
        if self.aec is not None:
            try:
                await self.aec.drain(timeout=8.0)
            except Exception as exc:  # noqa: BLE001
                logger.warning("AEC drain 异常: %s", exc)
        # 把 OmniLLM 里不足 1s 的音频残余发出去
        if self.omni is not None:
            try:
                await self.omni.flush_audio()
            except Exception as exc:  # noqa: BLE001
                logger.debug("omni.flush_audio 异常: %s", exc)
        # 等 ASR 最终结果
        if self.asr is not None:
            try:
                await self.asr.drain(timeout=asr_timeout)
            except Exception as exc:  # noqa: BLE001
                logger.warning("ASR drain 异常: %s", exc)

    async def close(self, reason: str = "client_stop") -> None:
        if self.closed:
            return
        self.closed = True
        logger.info("会话 %s 关闭（%s）", self.session_id, reason)
        if self.downstream is not None:
            try:
                await self.downstream.on_session_end(reason)
            except Exception as exc:  # noqa: BLE001
                logger.warning("downstream.on_session_end 异常: %s", exc)
        # 每个组件的关闭都加超时，避免一个卡住拖死整个会话清理
        for comp, name in ((self.aec, "aec"), (self.asr, "asr"),
                           (self.omni, "omni"), (self.tts, "tts")):
            if comp is not None:
                try:
                    await asyncio.wait_for(comp.close(), timeout=8.0)
                except asyncio.TimeoutError:
                    logger.warning("%s 关闭超时（8s）", name)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("%s 关闭异常: %s", name, exc)
        if self.face_worker is not None:
            try:
                self.face_worker.stop()
            except Exception as exc:  # noqa: BLE001
                logger.warning("face_worker 关闭异常: %s", exc)

    def summary(self) -> str:
        wall = time.monotonic() - self._t_wall0
        lines = [
            f"会话 {self.session_id}: 墙钟 {wall:.1f}s，"
            f"时钟 {self.clock.seconds():.1f}s，"
            f"漂移 {self.clock.drift_samples()} 采样"
        ]
        for k, v in self.stats.items():
            lines.append(f"  {k}: {v}")
        if self.aec is not None:
            r = self.aec.sample_ratio()
            if r is not None:
                lines.append(f"  aec 样本比: {r:.4f}")
            lat = self.aec.first_result_latency_ms()
            if lat is not None:
                lines.append(f"  aec 首窗延迟: {lat:.0f}ms")
        return "\n".join(lines)
