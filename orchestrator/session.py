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

        # 指标（生产环境没有观测就是瞎子）
        from .metrics import SessionMetrics
        self.metrics = SessionMetrics(session_id)
        self._mark_audio_t0: Optional[float] = None

        # 声学延迟估计用的原始 mic 累积缓冲（**AEC 之前**的信号才有回声）
        self._raw_mic_buf: Optional[list] = []
        self._raw_mic_len = 0
        # ASR 触发的在途任务（turn_trigger="asr"）
        self._trigger_task: Optional[asyncio.Task] = None

        # 回声消除模式：browser | service | off（可被 session.start 覆盖）
        self.aec_mode = str(config.get("aec_mode") or "browser")
        # 声学延迟：优先用该设备的历史记录，没有则用配置默认值
        self._delay_store = None
        self._client_key = "default"
        self.delay_default_ms = float(config.get("aec_default_delay_ms") or 250.0)
        self.delay_adaptive = bool(config.get("aec_adaptive_delay", True))
        self.delay_source = "default"
        self._last_stats_push = 0.0
        # 播放提前量（前端起播预留的时间）—— 校准时要从中扣除
        self.playback_delay_ms = int(config.get("playback_delay_ms") or 200)
        # 进行中的校准会话
        self._cal = None
        # ASR 流式文本的累积缓冲。
        # 2pass-online 给的是**增量片段**（"今天"/"吃饭"/"了吗"），必须自己
        # 拼接；2pass-offline 到达时用它覆盖并清空。
        self._asr_online_text = ""

        # 实时 ERLE 统计（播放期间才有效）：mic 能量 vs AEC 输出能量。
        # 用来**实测**当前延迟配置到底消掉多少 —— 排障时最直接的依据。
        self._mic_e = 0.0
        self._aec_out_e = 0.0

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
        self.metrics.inc("audio_chunks")
        self.metrics.inc("audio_samples", frame.n_samples)
        # 实时 ERLE 的输入侧（仅播放期间累计，见 _update_erle）
        if (self.aec is not None and self.ref_track is not None
                and self.ref_track.is_active(frame.t0, lookahead=int(0.5 * SR))):
            self._mic_e += float(np.mean(frame.data ** 2)) * frame.n_samples

        # 原始 mic 旁路：barge-in 检测用，零额外延迟
        self._raw_recent.append(float(np.sqrt(np.mean(x ** 2))))

        # 校准中：优先喂校准器（它要的是原始 mic，且需要连续采集）
        self._feed_calibration(frame)
        # 声学延迟自适应估计（用原始 mic + raw ref，只在播放窗口内做）
        self._maybe_update_delay(frame)
        # 周期性链路诊断（每 5s）+ 状态推送（每 2s，供 UI 显示延迟）
        self._periodic_diag(frame)
        self.push_stats()

        # 送去清洗：有 AEC 走 AEC，否则直接扇出。
        # AEC 是**可选**的清洗环节，不是链路的一环 —— 缺它不影响正确性。
        if self.aec is not None:
            ref = self._ref_for(frame)
            self._trace_ref(ref, frame)
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

    def _trace_ref(self, ref: np.ndarray, frame: AudioFrame) -> None:
        """追踪送进 AEC 的 farend 是否真的非零。

        「回声消不掉」的第一件事就是看这里：如果 ``ref_nonzero`` 恒为 0，
        说明参考轨没送到，AEC 压根没得消 —— 而不是 AEC 算法不行。
        """
        self.stats["ref_push_total"] = self.stats.get("ref_push_total", 0) + 1
        rms = float(np.sqrt(np.mean(ref ** 2))) if ref.size else 0.0
        if rms > 1e-6:
            prev = self.stats.get("ref_push_nonzero", 0)
            self.stats["ref_push_nonzero"] = prev + 1
            if rms > self.stats.get("ref_rms_max", 0.0):
                self.stats["ref_rms_max"] = round(rms, 5)
            if prev == 0:
                # 首次出现非零参考 —— 最关键的转折点，必须打日志
                logger.info(
                    "[%s] 参考轨首次送出非零 farend：t0=%d rms=%.4f "
                    "（写入区间=%s D=%d）",
                    self.session_id, frame.t0, rms,
                    self.ref_track.buf.written_span() if self.ref_track else None,
                    self.ref_track.delay_samples if self.ref_track else -1,
                )
            elif self.stats["ref_push_nonzero"] % 100 == 0:
                logger.info(
                    "[%s] 已送出 %d 个非零 farend 块（峰值 rms=%.4f）",
                    self.session_id, self.stats["ref_push_nonzero"],
                    self.stats.get("ref_rms_max", 0.0),
                )

    # ------------------------------------------------------------------ #
    #  AEC 模式与声学延迟
    # ------------------------------------------------------------------ #

    def apply_delay_seed(self, client_key: str, store=None) -> float:
        """用该设备的历史延迟作为初值（冷启动即准，不必从头收敛）。

        返回采用的延迟（ms）。``source`` 记在 ``self.delay_source``：
        ``stored``（有历史记录）/ ``default``（用配置默认值）。
        """
        self._client_key = client_key or "default"
        self._delay_store = store
        ms, source = self.delay_default_ms, "default"
        if store is not None:
            ms, source = store.get(self._client_key)
        self.delay_source = source
        if self.ref_track is not None:
            self.ref_track.delay_samples = int(ms * SR / 1000.0)
        logger.info(
            "[%s] 声学延迟初值 %.0fms（来源=%s，client=%s）",
            self.session_id, ms, source, self._client_key,
        )
        return ms

    def set_aec_mode(self, mode: str) -> None:
        """运行时切换回声消除模式。

        ``browser`` 时不连云端 AEC（省一次云往返）；``service`` 时启用并
        按延迟预对齐。会话中途切换是允许的 —— 前端可以两个都试听再决定。
        """
        if mode not in ("browser", "service", "off"):
            logger.warning("[%s] 未知 AEC 模式 %r，忽略", self.session_id, mode)
            return
        if mode == self.aec_mode:
            return
        old, self.aec_mode = self.aec_mode, mode
        logger.info("[%s] AEC 模式切换: %s → %s", self.session_id, old, mode)
        if mode == "service" and self.aec is None:
            self._want_aec_client = True       # main.py 的任务会拉起连接
        self.push_stats(force=True)

    async def start_calibration(self) -> bool:
        """开始一次声学延迟校准（播放啁啾 + 采集回声）。

        与自适应的区别：**主动**发起，不依赖 TTS 播放窗口，用户点一下
        按钮即可；用的是宽带啁啾，互相关峰值比语音尖锐得多。
        """
        from .calibrate import CalibrationSession
        if getattr(self, "_cal", None) is not None and not self._cal.done:
            logger.info("[%s] 校准已在进行中", self.session_id)
            return False
        self._cal = CalibrationSession(sr=SR)
        chirp = self._cal.start()
        # 走**与 TTS 相同的播放通道** —— 否则测的是另一条链路的延迟
        rid = f"cal_{int(time.monotonic())}"
        from .protocol import TtsAudio, TtsEnd, TtsStart
        await self.send_to_client(TtsStart(
            response_id=rid, text="（正在校准回声延迟…）", sample_rate=24000))
        # 分块送，与 TTS 一致
        chunk = 24000 // 2
        seq = 0
        for i in range(0, len(chirp), chunk):
            await self.send_to_client(TtsAudio.from_int16(
                chirp[i:i + chunk], rid, seq))
            seq += 1
        await self.send_to_client(TtsEnd(response_id=rid))
        # 校准信号也进参考轨（这样 AEC 期间不会把啁啾当回声残留）
        if self.ref_track is not None:
            at = self.clock.now() + int(
                self.playback_delay_ms * SR / 1000.0)
            self.ref_track.place(rid, 0, chirp, at)
        logger.info("[%s] 校准信号已发出（%.1fs）", self.session_id,
                    len(chirp) / 24000)
        return True

    def _feed_calibration(self, frame: AudioFrame) -> None:
        """把原始麦克风喂给进行中的校准。"""
        cal = getattr(self, "_cal", None)
        if cal is None or cal.done:
            return
        cal.feed(frame.data.reshape(-1))
        if cal.ready():
            self._finish_calibration()

    def _finish_calibration(self) -> None:
        from .protocol import ErrorMsg, SessionStats
        cal = getattr(self, "_cal", None)
        if cal is None:
            return
        res = cal.finish(playback_delay_ms=self.playback_delay_ms)
        if res is None or not res.get("ok"):
            msg = (f"校准置信度低（峰值比 {res.get('peak_ratio') if res else '-'}）。"
                   "请确认环境安静、音量适中后重试。")
            logger.warning("[%s] %s", self.session_id, msg)
            self._send_display(ErrorMsg(code="calibrate_failed", message=msg))
            self._cal = None
            return
        d_ms = res["delay_ms"]
        if self.ref_track is not None:
            self.ref_track.delay_samples = int(d_ms * SR / 1000.0)
        self.stats["delay_measured"] = 1
        self.delay_source = "calibrated"
        if self._delay_store is not None:
            self._delay_store.put(self._client_key, d_ms, n_samples=10)
        logger.info("[%s] 校准完成：D=%.0fms（已保存，下次自动使用）",
                    self.session_id, d_ms)
        self.push_stats(force=True)
        self._cal = None

    def set_delay_ms(self, ms: float) -> None:
        """手动设置声学延迟（调试/对比用）。

        用途：不确定该用哪个 D 时，逐个试并观察 ``session.stats`` 里的
        ERLE —— 抑制最强的那个就是对的。比推理可靠。
        """
        if self.ref_track is None:
            return
        self.ref_track.delay_samples = int(max(0.0, ms) * SR / 1000.0)
        self.delay_source = "manual"
        self.stats["delay_measured"] = 1
        # 重置 ERLE 累计，让新配置的效果可独立观察
        self._mic_e = 0.0
        self._aec_out_e = 0.0
        logger.info("[%s] 手动设置声学延迟 = %.0f ms（ERLE 计数已重置）",
                    self.session_id, ms)
        self.push_stats(force=True)

    def current_delay_ms(self) -> float:
        if self.ref_track is None:
            return 0.0
        return self.ref_track.delay_samples / SR * 1000.0

    def push_stats(self, force: bool = False) -> None:
        """定期把链路状态推给前端（含声学延迟，供 UI 显示）。"""
        now = time.monotonic()
        if not force and now - self._last_stats_push < 2.0:
            return
        self._last_stats_push = now
        from .protocol import SessionStats
        total = self.stats.get("ref_push_total", 0)
        nz = self.stats.get("ref_push_nonzero", 0)
        d_ms = self.current_delay_ms()
        suggested = self.delay_default_ms
        # 已实测到就用实测值作为"建议设置"
        if self.stats.get("delay_measured"):
            suggested = d_ms
        self._send_display(SessionStats(
            aec_mode=self.aec_mode,
            delay_ms=round(d_ms, 1),
            delay_source=self.delay_source,
            delay_measured=bool(self.stats.get("delay_measured")),
            delay_samples=self.ref_track.delay_samples if self.ref_track else 0,
            suggested_delay_ms=round(suggested, 1),
            aec_active=(self.aec_mode == "service" and self.aec is not None),
            ref_nonzero_ratio=round(nz / total, 3) if total else 0.0,
            erle_db=(round(self.current_erle_db(), 1)
                     if self.current_erle_db() is not None else None),
        ))

    def _periodic_diag(self, frame: AudioFrame) -> None:
        """每 5 秒打一条链路诊断 —— 排查"回声没消掉"时这是第一现场。

        包含三件事：
          · 参考轨送出情况（ref_push_nonzero / 总数）→ 参考有没有到 AEC
          · 声学延迟 D → 远超 20ms 就说明云端 AEC 的有效窗口外
          · 送出的 farend 近期 RMS → 是静音还是真有信号
        """
        now = time.monotonic()
        if now - getattr(self, "_last_diag_at", 0.0) < 5.0:
            return
        self._last_diag_at = now

        total = self.stats.get("ref_push_total", 0)
        nz = self.stats.get("ref_push_nonzero", 0)
        rt = self.ref_track
        d = rt.delay_samples if rt else -1
        lo, hi = rt.buf.written_span() if rt else (None, None)
        erle = self.current_erle_db()
        logger.info(
            "[%s] 诊断: 音频块=%d ref推送=%d(非零 %d, %.0f%%) D=%d采样(%.0fms) "
            "ERLE=%s 写入区间=[%s,%s] now=%d omni触发=%d",
            self.session_id,
            self.stats.get("audio_chunks_in", 0), total, nz,
            (100.0 * nz / total) if total else 0.0,
            d, (d / SR * 1000) if d >= 0 else -1,
            f"{erle:.1f}dB" if erle is not None else "n/a",
            lo, hi, self.clock.now(),
            self.stats.get("omni_triggers", 0),
        )

    def current_erle_db(self) -> Optional[float]:
        """播放期间实测的回声抑制比（dB）。

        ``10·log10(mic能量 / AEC输出能量)``，只在有过播放窗口时有效。
        排障时看它最直接：**调 D 前后 ERLE 的变化**能立刻判断配置对不对。
        注意它含近端语音（故绝对值偏低），但**不同 D 之间的相对比较**
        是有效的。
        """
        if self._mic_e <= 0 or self._aec_out_e <= 0:
            return None
        return 10.0 * np.log10(self._mic_e / self._aec_out_e)

    def _maybe_update_delay(self, frame: AudioFrame) -> None:
        """在**播放窗口内**估计声学路径延迟 D 并喂给参考轨。

        为什么必须做：AEC 服务把 nearend/farend 按**相同偏移**推入，即
        假定两者已样本对齐；而扬声器→麦克风有物理延迟（数十到数百 ms）。
        不补偿的话，参考与回声分量错位，消不掉。

        ⚠️ 两个关键点：
          · 只在播放窗口估计 —— 空闲时 ref 是静音，估出来是垃圾
          · 用 **raw ref**（未补偿）与**原始 mic** 估计 —— 用补偿后的
            信号会自我抵消（补偿多少就测不出多少）
        """
        if self.ref_track is None or self.delay_tracker is None:
            return
        if self._raw_mic_buf is None:
            return
        # 播放窗口判断：当前或未来有参考在播
        if not self.ref_track.is_active(frame.t0, lookahead=int(0.5 * SR)):
            return

        # ⚠️ 窗口长度必须 **大于可能的最大延迟**，否则互相关搜不到：
        # mic 里的回声来自 D 之前的播放，若只读"当前窗"的 raw，两者
        # 没有重叠片段，argmax 会落在噪声上 —— 表现为恒报 0ms。
        # 取 WINDOW（1s）覆盖 0~1s 的延迟范围。
        WINDOW = SR
        self._raw_mic_buf.append(frame.data.reshape(-1))
        self._raw_mic_len += frame.n_samples
        if self._raw_mic_len < WINDOW:
            return
        mic = np.concatenate(self._raw_mic_buf)[-WINDOW:]

        # raw ref 要往前多读一个"最大延迟"的长度：
        # mic 窗 [t-W, t) 里的回声，其源在 raw 的 [t-W-D, t-D)。
        # 读 [t-W-Dmax, t) 这一整段，让互相关自己找偏移。
        dmax = int(self.delay_tracker.max_delay)
        raw = self.ref_track.read_raw(frame.t1 - WINDOW - dmax, WINDOW + dmax)
        self._raw_mic_buf = []
        self._raw_mic_len = 0
        if raw.shape[0] < WINDOW:
            return
        est = self.delay_tracker.estimate(mic, raw)
        if est is None:
            return
        if not self.delay_adaptive:
            return
        prev = self.ref_track.delay_samples
        self.ref_track.delay_samples = self.delay_tracker.delay
        self.stats["delay_measured"] = 1
        self.delay_source = "measured"
        changed = abs(self.ref_track.delay_samples - prev) >= 16  # >1ms
        if changed:
            logger.info(
                "[%s] 声学延迟自适应: %d → %d 采样（%.0fms），"
                "本次估计 %d（累计 %d 次样本）",
                self.session_id, prev, self.ref_track.delay_samples,
                self.ref_track.delay_samples / SR * 1000,
                est, self.delay_tracker.estimates,
            )
            self.push_stats(force=True)
            # 持久化：下次同一设备冷启动就用这个值，不必重新收敛
            if self._delay_store is not None and self.delay_tracker.estimates >= 3:
                self._delay_store.put(
                    self._client_key,
                    self.ref_track.delay_samples / SR * 1000.0,
                    n_samples=self.delay_tracker.estimates,
                )

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
        """播放回执。

        ⚠️ 浏览器的 ``ctx_time``（AudioContext 秒）与我们的会话采样时钟
        是**两个时钟域**。参考轨的落位由服务端按会话时钟自算（见
        executor），这里只在**取消/结束**时用"从现在起"的语义截断 ——
        那不需要跨域换算。

        早期实现用 ``ctx_time`` 做绝对对齐，导致写入位置变成大负数、
        参考轨读出来全是 0（AEC 的 farend 恒为静音）。不要退回那种做法。
        """
        if self.closed or self.ref_track is None:
            return
        if phase in ("ended", "cancelled"):
            # 取消/结束时必须截断 ref 尾部：否则 AEC 会拿着没播出的
            # 音频当参考，主动误适配去追一个不存在的回声 —— 比不给更糟
            n = self.ref_track.truncate(response_id,
                                        from_sample=self.clock.now())
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
            self.metrics.inc("aec_segments")
            # 实时 ERLE：累加 AEC 输出的能量，与同期 mic 能量比。
            # 播放期间才有意义（无回声时 ERLE≈0 反映的是纯近端）。
            self._aec_out_e += float(np.mean(np.asarray(seg) ** 2)) * np.asarray(seg).size
            # 首窗延迟（相对会话开始收音频）
            lat = self.aec.first_result_latency_ms() if self.aec else None
            if lat is not None and self.metrics.aec_first_window.count == 0:
                self.metrics.aec_first_window.record(lat)
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
                delta = msg.get("text", "")
                if delta:
                    # ⚠️ 2pass-online 是**增量片段**（"今天" / "吃饭" / "了吗"），
                    # 不是累积文本 —— 官方客户端也是自己 += 拼起来显示的
                    # （funasr_wss_client.py: `text_print_2pass_online += text`）。
                    # 必须自己累积，否则 UI 上只剩最后一个词。
                    self._asr_online_text += delta
                    self.post_downstream(AsrPartial(
                        t=self.clock.now(), text=self._asr_online_text,
                        confidence=parse_confidence(msg),
                        segment_id=self.asr.partials,
                    ))
                    self._send_display(AsrDisplay(
                        phase="partial", text=self._asr_online_text,
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
            # 最终结果到达：用它覆盖流式累积文本，并**清空缓冲**为下一段
            # 做准备（服务端也是这么做的：text_print_2pass_online = ""）
            self._asr_online_text = ""
            if fin["text"]:
                self._send_display(AsrDisplay(
                    phase="final", text=fin["text"],
                    t_ms=int(self.clock.seconds() * 1000),
                ))
                # ---- ASR 断句触发（turn_trigger="asr"）----
                # OmniLLM 一直在以 force_listen 累积视听上下文（"边听边看"），
                # 这里补一个触发 push 让它开口。视觉上下文是**连续流入**的，
                # 所以回复能带上截止到此刻的画面理解。
                if (self.omni is not None
                        and getattr(self.omni, "turn_trigger", "") == "asr"):
                    self._trigger_omni(fin["text"])

        # 接收循环（阻塞直到 is_final 或连接关闭）
        await self.asr.recv_loop(on_message)

    def _trigger_omni(self, text: str) -> None:
        """按 ASR 文本触发一次 OmniLLM 回复（异步，不阻塞 ASR 回调）。"""
        if self._trigger_task is not None and not self._trigger_task.done():
            # 上一次触发还没送出去 —— 排队即可（同一话轮内多次 final
            # 通常意味着 VAD 切段，合并成一次触发更自然）
            logger.debug("已有触发在途，合并本次 ASR final")
            return
        self.stats["omni_triggers"] = self.stats.get("omni_triggers", 0) + 1

        async def _do():
            try:
                await self.omni.trigger_reply(text)
            except Exception as exc:  # noqa: BLE001
                logger.warning("触发 OmniLLM 回复失败: %s", exc)

        self._trigger_task = asyncio.create_task(_do())

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
            if kind == "obs":
                # 每帧的人脸观测 → UI 叠加显示（人脸框/置信度/唇动/
                # 身份/唤醒）。**不进 downstream** —— 那是控制流，
                # 每帧 25Hz 投递会把下游淹没。控制信号走 wake/lip/identity。
                # 带上 G1 输入帧的**实际尺寸** —— 前端必须用它把框坐标
                # 映射到视频显示区。写死分辨率会导致框错位（实测踩过）。
                fs = getattr(self.face_worker.provider, "_frame_size", None) \
                    if self.face_worker else None
                self._last_face = {
                    "valid": bool(ev.valid),
                    "box": [round(v, 1) for v in ev.box] if ev.box else None,
                    "score": round(ev.score, 3),
                    "speaking": bool(ev.speaking),
                    "lip": ev.lip_state,
                    "interacting": bool(ev.interacting),
                    "person_id": int(ev.person_id),
                    "src_w": fs[0] if fs else None,
                    "src_h": fs[1] if fs else None,
                }
                self._push_face_display()
            elif kind == "wake":
                self.post_downstream(FaceWake(
                    t=ev.t, phase=ev.phase, track_id=ev.track_id,
                    dwell_ms=ev.dwell_ms, mean_confidence=ev.mean_confidence,
                ))
                self._last_wake = {
                    "phase": ev.phase,
                    "dwell_ms": int(ev.dwell_ms),
                    "score": round(ev.mean_confidence, 3),
                }
                self._push_face_display()
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
                # 记住识别结果，后续每帧的 face.state 都带上（否则前端
                # 只在识别那一瞬间能看到名字，之后又变回"未知"）
                self._last_identity = {
                    "name": ev.name, "uid": ev.uid, "person_id": ev.person_id,
                    "similarity": round(ev.similarity, 3),
                    "enrolled": bool(ev.is_enrolled),
                }
                self._push_face_display()

    def _push_face_display(self) -> None:
        """把最新的观测/唤醒/身份合成一条 face.state 发给 UI。

        节流到 ~10Hz（人脸是 25fps，UI 不需要那么细）—— 每帧发会白占
        带宽且前端画不过来。
        """
        now = time.monotonic()
        if now - getattr(self, "_last_face_push", 0.0) < 0.1:
            return
        self._last_face_push = now
        from .protocol import FaceDisplay
        self._send_display(FaceDisplay(
            tracks=[self._last_face] if getattr(self, "_last_face", None) else [],
            identity=getattr(self, "_last_identity", None),
            wake=getattr(self, "_last_wake", None),
        ))

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
        # 指标：按事件类型统一计数（比在各回调里散着打点更不易漏）
        k = getattr(ev, "kind", None)
        if k == "asr.partial":
            self.metrics.inc("asr_partials")
        elif k == "asr.final":
            self.metrics.inc("asr_finals")
        elif k == "asr.turnsense":
            self.metrics.inc("asr_turnsense")
        elif k == "omni.delta":
            self.metrics.inc("omni_deltas")
        elif k == "omni.done":
            self.metrics.inc("omni_dones")
        elif k == "omni.turnsense":
            self.metrics.inc("omni_turnsense")
        elif k == "face.wake":
            self.metrics.inc("face_wakes")
        elif k == "face.lip":
            self.metrics.inc("face_lips")
        elif k == "face.identity":
            self.metrics.inc("face_identities")
        elif k == "playback":
            self.metrics.inc("playback_receipts")
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
                    asr_timeout: float = 15.0,
                    omni_turn_timeout: float = 15.0,
                    tts_idle_timeout: float = 30.0) -> None:
        """优雅收尾：把残余数据推完、等最终结果。

        与 ``close()`` 分离的原因：收尾**可以慢**（等 ASR 跑完最终识别、
        等 OmniLLM 吐完残余），但 ``close()`` **必须快** —— 它会阻塞
        WS 处理协程的清理，被 cancel 会丢数据。
        """
        if self.closed:
            return
        logger.info("会话 %s 开始收尾（%s）", self.session_id, reason)
        # 顺序有讲究：
        #   AEC（推完尾部窗口，让 ASR 有完整输入）
        #   → OmniLLM（说完整当前这轮，触发的 Speak 才有机会合成）
        #   → ASR（吃下尾部后才会吐最终结果）
        if self.aec is not None:
            try:
                await self.aec.drain(timeout=8.0)
            except Exception as exc:  # noqa: BLE001
                logger.warning("AEC drain 异常: %s", exc)

        # ⚠️ 等 OmniLLM 把**当前这一轮**说完。
        # 不等的话，close() 会抢先关掉 omni，response.done 永远到不了 ——
        # 于是 downstream 收不到 OmniResponseDone、不会触发 Speak，
        # 表现为"偶发地一句 TTS 都没有"。
        if self.omni is not None:
            try:
                await self.omni.flush_audio()
            except Exception as exc:  # noqa: BLE001
                logger.debug("omni.flush_audio 异常: %s", exc)
            await self._wait_omni_turn(timeout=omni_turn_timeout)

        # 等 ASR 最终结果
        if self.asr is not None:
            try:
                await self.asr.drain(timeout=asr_timeout)
            except Exception as exc:  # noqa: BLE001
                logger.warning("ASR drain 异常: %s", exc)

        # ⚠️ 必须**最后**等执行器：ASR 的最终结果也会触发 Speak
        # （downstream_mode=asr 时），而 TTS 合成+分帧发送需要时间。
        # 不等的话 close() 会抢先把会话关掉 —— 音频合成出来了但没送到
        # 浏览器，测试端表现为 tts_end=0。
        # 注意这一步**不能**包在 omni 分支里（ASR 模式同样需要）。
        await self._wait_executor_idle(timeout=tts_idle_timeout)

    async def _wait_omni_turn(self, timeout: float = 15.0) -> bool:
        """等 OmniLLM 当前这一轮说完整（收到 response.done）。

        ⚠️ **不能**去窥探 ``_down_q`` —— ``run_downstream`` 是同一个队列的
        消费者，两个消费者会互抢事件。这里改用计数器（由 OmniClient 的
        事件回调递增），只观察不消费。
        """
        baseline = self.stats.get("omni_done", 0)
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            if self.stats.get("omni_done", 0) > baseline:
                logger.info("收尾：OmniLLM 本轮已结束（%.1fs）",
                            time.monotonic() - t0)
                return True
            if self.closed:
                return False
            await asyncio.sleep(0.05)
        logger.info("收尾：等待 OmniLLM 本轮结束超时（%.1fs）", timeout)
        return False

    async def _wait_executor_idle(self, timeout: float = 20.0) -> bool:
        """等执行器把在飞的 TTS 合成跑完。"""
        if self.executor is None:
            return True
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            if getattr(self.executor, "_speak_inflight", 0) == 0:
                return True
            await asyncio.sleep(0.1)
        logger.info("收尾：等待 TTS 合成完成超时（%.1fs）", timeout)
        return False

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
        # 关键指标一行摘要
        try:
            lines.append(f"  指标: {self.metrics.summary_line()}")
        except Exception:  # noqa: BLE001
            pass
        for k, v in self.stats.items():
            lines.append(f"  {k}: {v}")
        if self.aec is not None:
            r = self.aec.sample_ratio()
            if r is not None:
                lines.append(f"  aec 样本比: {r:.4f}")
            lat = self.aec.first_result_latency_ms()
            if lat is not None:
                lines.append(f"  aec 首窗延迟: {lat:.0f}ms")
        if self.ref_track is not None:
            lo, hi = self.ref_track.buf.written_span()
            t = self.clock.now()
            probe = (self.ref_track.read(max(0, t - 1600), 1600)
                     if t > 1600 else self.ref_track.read(0, 1600))
            peak = float(np.abs(probe).max())
            lines.append(
                f"  参考轨: 写入区间=[{lo},{hi}] now={t} "
                f"D={self.ref_track.delay_samples}采样"
                f"({self.ref_track.delay_samples / SR * 1000:.0f}ms)"
                f" 近期峰值={peak:.4f}"
            )
            if peak < 1e-6 and lo is not None and t > lo:
                lines.append("    ⚠️ 参考轨近期为静音 —— AEC 的 farend 是零，"
                             "回声不会被消除")
        return "\n".join(lines)
