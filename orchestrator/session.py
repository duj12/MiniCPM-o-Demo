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

#: 事件循环看门狗：探测间隔（秒）与告警阈值（秒）。
#: 阈值取 0.5s —— 正常抖动在几十 ms 量级，真卡一下就远超这个数；
#: 定太高会漏掉「卡 1~2s 又恢复」这种（它已经足以让 ASR 丢字、Omni 断连）。
_LOOP_LAG_INTERVAL = 0.5
_LOOP_LAG_WARN = 0.5


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
        self.face_worker = None
        self.downstream = None
        self.executor = None
        self.send_to_client = None   # Callable[[Any], Awaitable[None]]
        # 客户端能力位（session.start 的 caps）。有 ``playback_anchor`` 才
        # 会等浏览器的 armed 承诺 —— 旧客户端/存量测试没有这一位，因此
        # 完全走原来的预测路径，零回归。
        self.client_caps: set = set()

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
        #: 诊断用：会话建立时刻。用来在「一直没收到音频块」时给出等待时长
        #: （见 ``_periodic_diag``）—— 那是「web 端没有 ASR 结果」的头号根因。
        self._t_audio_wait_start = time.monotonic()

        # 音频转储（仅在 ORCH_DUMP_AUDIO 设置时开启；默认零开销）
        import os as _os
        self._dump_path = _os.environ.get("ORCH_DUMP_AUDIO") or None
        self._dump_mic: list = []
        self._dump_ref: list = []
        self._dump_raw: list = []      # 未做 D 补偿的原始参考轨（测 D 用）
        self._dump_aec: list = []
        # 视频帧转储：**原样存收到的 JPEG 字节**，不做任何转码 —— 发给
        # Omni/face 的就是这些字节，存 mp4 反而有损、且丢掉逐帧时间对应。
        # 每项 (t_ms, jpeg_bytes)，会话结束时写 .mjpeg + .tsv。
        self._dump_face: list = []
        self._dump_omni: list = []
        # ASR 触发的在途任务（turn_trigger="asr"）
        self._trigger_task: Optional[asyncio.Task] = None

        # 回声消除模式：browser | service | off（可被 session.start 覆盖）
        self.aec_mode = str(config.get("aec_mode") or "browser")
        # 声学延迟：优先用该设备的历史记录，没有则用配置默认值
        self._delay_store = None
        self._client_key = "default"
        # ⚠️ 只在**键缺失**时兜底，不用 `or`：`or` 会把合法的 0 变成 250
        # （0 是 falsy），让 config 里显式设的 0 失效。
        _dd = config.get("aec_default_delay_ms")
        self.delay_default_ms = float(250.0 if _dd is None else _dd)
        self.delay_source = "default"
        self._last_stats_push = 0.0
        # 参考轨落位锚点：response_id → 浏览器**承诺**的起播会话采样位置。
        # 浏览器在 tts.start 之后立刻回 ``playback{phase:'armed'}``，
        # 执行器等这个值来 place()。等不到就退回预测（见 _resolve_play_at）。
        self._armed: Dict[str, int] = {}
        self._armed_evt: Dict[str, asyncio.Event] = {}
        self.anchor_source = "none"      # ack | predicted | none
        self.anchor_delta_ms = 0.0
        # 播放提前量（前端起播预留的时间）—— 校准时要从中扣除
        _pd = config.get("playback_delay_ms")
        self.playback_delay_ms = int(200 if _pd is None else _pd)
        # 进行中的校准会话
        self._cal = None
        # ASR 流式文本的累积缓冲。
        # 2pass-online 给的是**增量片段**（"今天"/"吃饭"/"了吗"），必须自己
        # 拼接；2pass-offline 到达时用它覆盖并清空。
        self._asr_online_text = ""
        #: 上次**推给 IC** 的 ASR 状态快照 —— 「变化才推」的去重键。
        #: ⚠️ 没有它，归零那一刻就没人通知 IC：ASR 消息只在说话时才有，
        #:    静音时状态被 tick 清了，却没有任何推送 → IC 的 `说`/`抢`
        #:    永远停在最后一帧（比如 HIGH），SOP 06 判定跟着错。
        self._last_asr_pushed: Optional[dict] = None
        #: 上次**推给 UI** 的 ASR 文字。UI 要保留最后一句字幕（不随内部
        #: `transcript` 归零而消失），直到下一句开始才被替换。
        self._last_asr_text = ""

        # ---- 实时 ERLE：**按每次播报独立统计** ----
        # ⚠️ 早先的实现有 bug：mic 能量只在播放窗口累加，而 AEC 输出能量
        # 无条件累加 —— 两者的门控条件不同，比值根本不是"回声消除了多少"，
        # 而是"播放期能量 / 全程能量"。空闲时间一长分母无限增长，ERLE 会
        # 一路漂到 -40dB 并卡住，且新配置的影响被历史累积稀释到看不见。
        #
        # 现在：一次播报 = 一个测量窗口。窗口开始时清零，结束时结算。
        self._erle_active = False      # 当前是否处于测量窗口
        self._erle_windows = []        # 已完成窗口的 ERLE(dB)
        self._erle_cur_mic = 0.0
        self._erle_cur_aec = 0.0
        self._erle_idle_since = 0.0    # 播放结束后的静默起点（用于收窗）

    # ------------------------------------------------------------------ #
    #  浏览器侧输入
    # ------------------------------------------------------------------ #

    async def on_audio(self, x: np.ndarray, t_ms: int = 0,
                       ctx_time: float = 0.0, epoch: int = 0) -> None:
        """收到一段麦克风音频（float32 (1,T)，16kHz）。

        mic ingest 是**唯一**推进时钟的地方。时钟因此跟踪真实音频流，
        所有其他流都在它上面表达。

        ``ctx_time`` 是本块首采样在浏览器 ``AudioContext`` 上的时刻 ——
        登记成锚点后，服务端就能把浏览器承诺的起播时刻精确换算到会话
        采样轴上（见 ``SampleClock.record_anchor``）。
        """
        if self.closed:
            return
        if x.shape[1] != MIC_CHUNK:
            # 容忍非标准块：重采样/补齐，但绝不崩
            x = self._normalize_chunk(x)
        frame = self.clock.frame_of(x)
        if ctx_time > 0:
            self.clock.record_anchor(ctx_time, frame.t0, epoch)
        self.stats["audio_chunks_in"] += 1
        self.stats["audio_samples_in"] += frame.n_samples
        self.metrics.inc("audio_chunks")
        self.metrics.inc("audio_samples", frame.n_samples)
        # 实时 ERLE 的输入侧：只在**测量窗口**内累加（见 _erle_window）
        if self._erle_active:
            self._erle_cur_mic += float(np.mean(frame.data ** 2)) * frame.n_samples

        # 原始 mic 旁路：barge-in 检测用，零额外延迟
        self._raw_recent.append(float(np.sqrt(np.mean(x ** 2))))

        # 校准中：优先喂校准器（它要的是原始 mic，且需要连续采集）
        self._feed_calibration(frame)
        # ERLE 测量窗口维护（开窗/收窗）
        self._update_erle_window(frame)
        # 周期性链路诊断（每 5s）+ 状态推送（每 2s，供 UI 显示延迟）
        self._periodic_diag(frame)
        self.push_stats()

        # 送去清洗：有 AEC 走 AEC，否则直接扇出。
        # AEC 是**可选**的清洗环节，不是链路的一环 —— 缺它不影响正确性。
        #
        # ⚠️ **转储要在这里做，不能塞进 AEC 分支**。早先 `_dump_audio` 只在
        #    `if self.aec is not None` 里调 —— 于是用**浏览器原生 AEC**
        #    （前端不连云端 AEC，`self.aec is None`）时根本不转储，
        #    跑完整场一个文件都没有。转储是排查手段，不该依赖 AEC 模式。
        ref = self._ref_for(frame) if self.aec is not None else None
        self._dump_audio(frame.data, ref, frame)

        if self.aec is not None:
            self._trace_ref(ref, frame)
            await self.aec.push(frame.data, ref)
        else:
            await self._fanout_direct(frame.data)

    def _dump_audio(self, mic: np.ndarray, ref: np.ndarray,
                    frame: AudioFrame) -> None:
        """把 mic / farend 落盘（仅在开启转储时）。

        「回声消不掉」这个问题的**唯一确诊手段**：拿到 mic 与 farend 的
        原始波形，就能直接算出 mic 里到底有没有回声、以及它与 farend 差
        多少采样。在此之前我们一直在用"互相关峰比""ERLE""mic_peak"这些
        间接指标推断，已经推错过好几次（甚至得出过与 ASR 现象矛盾的
        结论）—— 必须看波形。

        开启：环境变量 ``ORCH_DUMP_AUDIO=/path/prefix``（会话结束时写成
        ``<prefix>-<sid>-mic.wav`` / ``-ref.wav`` / ``-raw.wav`` / ``-aec.wav``）。
        默认关闭，零开销 —— 实时路径上不该有额外 I/O。

        ⚠️ ``-ref.wav`` 是**已按 D 补偿过**的（它就是喂给 AEC 的那一份），
        拿它去测 D 只能得到残差。测 D 必须用 ``-raw.wav``（未补偿的原始轨）
        —— 见 ``tests/measure_delay.py``。
        """
        if self._dump_path is None:
            return
        self._dump_mic.append(mic.reshape(-1).copy())
        n = frame.n_samples
        # `ref is None` = 没连云端 AEC（浏览器原生模式）—— 此时没有"喂给
        # AEC 的 farend"这回事，但要**如实落盘成静音**，而不是跳过：
        # 跳了会让 ref/raw 的长度和 mic 对不上，离线对齐就废了。
        self._dump_ref.append(
            ref.reshape(-1).copy() if ref is not None
            else np.zeros(n, dtype=np.float32))
        if self.ref_track is not None:
            self._dump_raw.append(
                self.ref_track.read_raw(frame.t0, frame.n_samples).copy())
        else:
            self._dump_raw.append(np.zeros(n, dtype=np.float32))

    def _dump_aec_out(self, seg: np.ndarray) -> None:
        if self._dump_path is None:
            return
        self._dump_aec.append(np.asarray(seg).reshape(-1).copy())

    def flush_audio_dump(self) -> None:
        """会话结束时把转储写成 wav（三路：mic / farend / aec 输出）。"""
        if self._dump_path is None or not self._dump_mic:
            return
        import wave
        from pathlib import Path
        base = Path(self._dump_path)
        base.parent.mkdir(parents=True, exist_ok=True)
        for name, chunks in (("mic", self._dump_mic), ("ref", self._dump_ref),
                             ("raw", self._dump_raw), ("aec", self._dump_aec)):
            if not chunks:
                continue
            x = np.concatenate(chunks)
            # mic/ref 是 [-1,1] float32；aec 输出同为 float32
            peak = float(np.abs(x).max()) if x.size else 0.0
            path = f"{base}-{self.session_id}-{name}.wav"
            with wave.open(path, "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(SR)
                w.writeframes(
                    np.clip(x * 32767.0, -32768, 32767).astype(np.int16).tobytes())
            logger.info("[%s] 音频转储 %s：%.1fs 峰值=%.5f",
                        self.session_id, path, x.size / SR, peak)

    def flush_video_dump(self) -> None:
        """会话结束时把视频帧写成 ``.mjpeg`` + ``.tsv``。

        **为什么不是 mp4**：发给 Omni/face 的就是**收到的原始 JPEG 字节**，
        中间没有任何转码。存 mp4 要重编码 —— 有损、而且丢掉"哪一帧对应哪个
        时刻"的精确关系，出问题时没法逐帧复现。MJPEG 就是 JPEG 首尾相接，
        `ffmpeg -i x.mjpeg out.mp4` 随时能转来看，但**验证要用原文件**。

        格式对齐 G1 库自己 debug 的落盘方式（它也是 mjpeg + tsv），
        以及 ``assets/video/test.mp4`` 那类素材。

        ``.tsv`` 列：``frame_index`` / ``t_ms``（会话采样轴，毫秒）/ ``bytes``。
        """
        if self._dump_path is None:
            return
        from pathlib import Path
        for name, frames in (("face", self._dump_face),
                             ("omni", self._dump_omni)):
            if not frames:
                continue
            mjpeg = Path(f"{self._dump_path}-{self.session_id}-{name}.mjpeg")
            tsv = Path(f"{self._dump_path}-{self.session_id}-{name}.tsv")
            mjpeg.parent.mkdir(parents=True, exist_ok=True)
            with open(mjpeg, "wb") as f:
                for _, jpeg in frames:
                    f.write(jpeg)
            with open(tsv, "w", encoding="utf-8") as f:
                f.write("frame_index\tt_ms\tbytes\n")
                for i, (t_ms, jpeg) in enumerate(frames):
                    f.write(f"{i}\t{t_ms}\t{len(jpeg)}\n")
            total = sum(len(j) for _, j in frames)
            logger.info("[%s] 视频转储 %s：%d 帧 / %.1fMB（+ %s）",
                        self.session_id, mjpeg, len(frames),
                        total / 1e6, tsv.name)

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
        """用该设备的**离线实测**延迟作为初值。

        D 是每台设备一个固定常量（见 ``config.aec_default_delay_ms`` 的
        说明），不再运行时自适应，所以这里就是"取历史记录，没有则用配置
        默认值"，没有收敛过程。

        返回采用的延迟（ms）。``source`` 记在 ``self.delay_source``：
        ``stored``（有历史记录）/ ``default``（配置默认值 —— **未实测，
        AEC 大概率不生效**，UI 会显著提示）。
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
            "[%s] 声学延迟 %.0fms（来源=%s，client=%s）"
            "%s",
            self.session_id, ms, source, self._client_key,
            "" if source == "stored" else
            " ⚠️ 该设备尚未离线实测 D —— 算法 AEC 大概率不生效，"
            "请跑 tests/measure_delay.py",
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
        """开始一次声学延迟校准（播放双段啁啾 + 采集回声）。

        与自适应的区别：**主动**发起，不依赖 TTS 播放窗口，用户点一下
        按钮即可；用的是宽带啁啾，互相关峰值比语音尖锐得多。

        ⚠️ 浏览器页面用的是**独立通道** ``/v1/calibrate``（见
        calibrate_endpoint.py），那条路径让前端直接复用 PcmPlayer，
        播放行为与 TTS 完全一致。本方法是会话内的备用路径，同样走
        ``tts.*`` 消息通道，并**由服务端自己算起播锚点**（它知道发送
        时刻 + 约定的播放提前量）。
        """
        from .calibrate import CalibrationSession
        if getattr(self, "_cal", None) is not None and not self._cal.done:
            logger.info("[%s] 校准已在进行中", self.session_id)
            return False
        self._cal = CalibrationSession(sr=SR)
        signal, _starts = self._cal.start()
        # 采集起点 = 会话当前采样位置（之后每帧都喂给它）
        self._cal_t0 = self.clock.now()
        # 走**与 TTS 相同的播放通道** —— 否则测的是另一条链路的延迟
        rid = f"cal_{int(time.monotonic())}"
        from .protocol import TtsAudio, TtsEnd, TtsStart
        await self.send_to_client(TtsStart(
            response_id=rid, text="（正在校准回声延迟…）", sample_rate=24000))
        # 分块送，与 TTS 一致
        chunk = 24000 // 2
        seq = 0
        for i in range(0, len(signal), chunk):
            await self.send_to_client(TtsAudio.from_int16(
                signal[i:i + chunk], rid, seq))
            seq += 1
        await self.send_to_client(TtsEnd(response_id=rid))
        # 校准信号也进参考轨（这样 AEC 期间不会把啁啾当回声残留）
        if self.ref_track is not None:
            at = self.clock.now() + int(
                self.playback_delay_ms * SR / 1000.0)
            self.ref_track.place(rid, 0, signal, at)
            # 浏览器按同一约定起播 → 起播位置相对采集起点就是这段偏移
            self._cal.set_play_anchor(at - self._cal_t0)
        logger.info("[%s] 校准信号已发出（%.1fs）", self.session_id,
                    len(signal) / 24000)
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
        from .protocol import ErrorMsg
        cal = getattr(self, "_cal", None)
        if cal is None:
            return
        res = cal.finish()
        if res is None or not res.get("ok"):
            reason = (res or {}).get("reason") or (res or {}).get("error") \
                or "校准失败，请重试"
            logger.warning("[%s] 校准失败：%s", self.session_id, reason)
            self._send_display(ErrorMsg(code="calibrate_failed",
                                        message=reason))
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
        """手动设置声学延迟 D（**离线测完之后写回**用）。

        正常情况下 D 来自 ``ORCH_AEC_DEFAULT_DELAY_MS`` 或 DelayStore，
        不需要走这里。保留它是为了：离线脚本还没把值写进配置时，可以先
        在会话里临时试一个数并观察 ERLE —— 抑制最强的那个就是对的。
        """
        if self.ref_track is None:
            return
        self.ref_track.delay_samples = int(max(0.0, ms) * SR / 1000.0)
        self.delay_source = "manual"
        self.stats["delay_measured"] = 1
        # 清空历史窗口 —— 之后**下一次播报**的 ERLE 就是新配置的独立测量值。
        # 不清的话中位数会被旧配置的结果拖住，看不出变化。
        self._erle_windows.clear()
        self._erle_active = False
        self._erle_cur_mic = 0.0
        self._erle_cur_aec = 0.0
        logger.info("[%s] 手动设置声学延迟 = %.0f ms（ERLE 历史已清空，"
                    "请触发一次新播报来看效果）", self.session_id, ms)
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
        erle = self.current_erle_db()
        self._send_display(SessionStats(
            aec_mode=self.aec_mode,
            delay_ms=round(d_ms, 1),
            delay_source=self.delay_source,
            delay_measured=bool(self.stats.get("delay_measured")),
            delay_samples=self.ref_track.delay_samples if self.ref_track else 0,
            aec_active=(self.aec_mode == "service" and self.aec is not None),
            ref_nonzero_ratio=round(nz / total, 3) if total else 0.0,
            erle_db=round(erle, 1) if erle is not None else None,
            anchor_source=self.anchor_source,
            anchor_delta_ms=round(self.anchor_delta_ms, 2),
            anchor_residual_ms=round(self.clock.anchor_residual_ms(), 2),
        ))

    def _periodic_diag(self, frame: AudioFrame) -> None:
        """每 5 秒打一条链路诊断 —— 排查"回声没消掉"时这是第一现场。

        包含四件事：
          · 参考轨送出情况（ref_push_nonzero / 总数）→ 参考有没有到 AEC
          · 声学延迟 D → 远超 20ms 说明设备没测过或测错了
          · **落位锚点来源** → ack（浏览器承诺，准）还是 predicted（服务端
            猜，误差逐句变化，是回声消不掉的根因）
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
            "锚点=%s(偏差 %.1fms, 残差 %.1fms, %d 个) "
            "ERLE=%s 写入区间=[%s,%s] now=%d omni触发=%d",
            self.session_id,
            self.stats.get("audio_chunks_in", 0), total, nz,
            (100.0 * nz / total) if total else 0.0,
            d, (d / SR * 1000) if d >= 0 else -1,
            self.anchor_source, self.anchor_delta_ms,
            self.clock.anchor_residual_ms(), self.clock.anchor_count(),
            f"{erle:.1f}dB" if erle is not None else "n/a",
            lo, hi, self.clock.now(),
            self.stats.get("omni_triggers", 0),
        )

    def _update_erle_window(self, frame: AudioFrame) -> None:
        """维护「一次播报 = 一个测量窗口」的 ERLE 统计。

        窗口语义（这很关键，早先实现没有窗口概念导致指标失真）：
          · 参考轨开始有内容 → **开窗**，清零累加器
          · 播放结束且静默 >0.5s → **收窗**，结算本次 ERLE 入列表
          · 下次播报重新开窗

        这样每次改配置后的**下一次播报**给出的就是该配置的独立测量值，
        不受历史累积影响。
        """
        if self.ref_track is None or self.aec is None:
            return
        playing = self.ref_track.is_active(frame.t0, lookahead=int(0.3 * SR))
        now = time.monotonic()
        if playing:
            if not self._erle_active:
                self._erle_active = True
                self._erle_cur_mic = 0.0
                self._erle_cur_aec = 0.0
                logger.debug("[%s] ERLE 测量窗口开启", self.session_id)
            self._erle_idle_since = 0.0
        elif self._erle_active:
            # 播放结束：等 0.5s 让 AEC 的尾部输出也收进来，再结算
            if self._erle_idle_since == 0.0:
                self._erle_idle_since = now
            elif now - self._erle_idle_since > 0.5:
                self._close_erle_window()

    def _close_erle_window(self) -> None:
        self._erle_active = False
        self._erle_idle_since = 0.0
        if self._erle_cur_mic <= 0 or self._erle_cur_aec <= 0:
            return
        erle = 10.0 * np.log10(self._erle_cur_mic / self._erle_cur_aec)
        self._erle_windows.append(erle)
        if len(self._erle_windows) > 20:
            self._erle_windows.pop(0)
        logger.info(
            "[%s] 本轮播报 ERLE = %.1f dB（D=%d 采样/%.0fms，累计 %d 轮）",
            self.session_id, erle, self.ref_track.delay_samples if self.ref_track else -1,
            (self.ref_track.delay_samples / SR * 1000) if self.ref_track else -1,
            len(self._erle_windows),
        )
        self.push_stats(force=True)

    def current_erle_db(self) -> Optional[float]:
        """最近若干次播报的 ERLE 中位数（dB）。

        ``10·log10(mic能量 / AEC输出能量)``。含近端语音，故绝对值偏低；
        看**改配置前后的变化**才有意义 —— 抑制变强说明方向对了。
        """
        if self._erle_active and self._erle_cur_mic > 0 and self._erle_cur_aec > 0:
            # 窗口进行中：给个实时值（播报未结束时的粗略参考）
            return 10.0 * np.log10(self._erle_cur_mic / self._erle_cur_aec)
        if not self._erle_windows:
            return None
        return float(np.median(self._erle_windows[-5:]))

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
        if self._dump_path is not None:
            # 原样存 —— 与喂给 face_worker 的是同一份字节
            self._dump_face.append((int(self.clock.now() * 1000 // SR), jpeg))
        if self.face_worker is not None:
            # put_nowait：队列满则丢最旧帧（新帧对唇动状态更有价值）
            self.face_worker.offer(jpeg, self.clock.now())

    async def on_video_omni(self, jpeg: bytes, t_ms: int = 0) -> None:
        """收到 OmniLLM 用视频帧（1fps）。"""
        if self.closed:
            return
        self.stats["video_omni_frames"] += 1
        if self._dump_path is not None:
            self._dump_omni.append((int(self.clock.now() * 1000 // SR), jpeg))
        if self.omni is not None:
            self.omni.offer_frame(jpeg)

    async def on_playback_receipt(self, response_id: str,
                                  phase: str, ctx_time: float,
                                  seq: int = 0,
                                  sample_offset: int = 0,
                                  *, start_ctx: float = 0.0,
                                  stop_ctx: float = 0.0,
                                  epoch: int = 0) -> None:
        """播放回执。

        ⚠️ 浏览器的 ``ctx_time``（AudioContext 秒）与我们的会话采样时钟
        是**两个时钟域**，绝不能直接混用。换算必须走
        ``clock.ctx_to_sample()``（它靠 mic 块自带的锚点拟合，是精确的）。

        早期实现直接把 ``ctx_time`` 当采样位置用，导致写入位置变成大负数、
        参考轨读出来全是 0（AEC 的 farend 恒为静音）。不要退回那种做法。
        """
        if self.closed or self.ref_track is None:
            return

        # ---- 实际出声边沿（playing / stopped）----
        # ⚠️ **只有这两个 phase 能定义 playback_active**。它们由浏览器按
        #    `player.remaining()`（读 AudioContext 音频时钟）算出、边沿触发，
        #    和 tts.start/tts.end 无关。
        #
        #    早先用 `armed`（承诺起播，还带 200ms 提前量）当"开始出声"、
        #    用 `ended`（音频送完）当"停止"，两头都不准：开头早报 200ms+，
        #    结尾早报几百 ms（耳朵里尾巴还在响）。PRD 要的是表达层事实，
        #    那就只能由表达层按真实播放进度写。
        #
        #    多句连播时浏览器一路报 playing（后一句紧接着排上），中间不会
        #    掉到 stopped —— 正是"前一句播完立即播下一句"该有的表现。
        if phase in ("playing", "stopped"):
            self._notify_playback_active(phase == "playing")
            return

        if phase == "armed":
            # 浏览器**承诺**的起播时刻 —— 执行器正等着它去 place() 参考轨。
            # ⚠️ 这里**不再**置 playback_active（它只表示"排程好了"，不是
            #    "在出声"，见上）。
            self._note_armed(response_id, start_ctx, epoch)
            self._post_downstream_playback(response_id, phase, ctx_time, seq)
            return

        if (phase == "started" and start_ctx > 0.0
                and response_id in self._armed):
            # 校验：实际排程时刻 vs 承诺。**只告警，不修正** ——
            # 此时参考轨已按承诺落位、AEC 已在读它，重新对齐意味着
            # 一段"位置错的参考"要被回滚，而 realign 需要保留每句的
            # 重采样结果并理清它与 truncate/resize 的交互，代价远大于收益。
            # 几毫秒的偏差损失几个 dB 抑制，但"截断重写"一旦时机错位就是
            # 整段回声漏进 ASR（正是我们要修的故障）。
            actual = self.clock.ctx_to_sample(start_ctx, epoch)
            if actual is not None:
                delta = actual - self._armed[response_id]
                self.anchor_delta_ms = delta / SR * 1000.0
                if abs(delta) > int(0.005 * SR):
                    logger.warning(
                        "[%s] 起播承诺未兑现：response=%s 承诺与实测差 "
                        "%.1fms —— 参考轨会有同等错位（容忍窗仅 ±5ms）",
                        self.session_id, response_id, delta / SR * 1000.0,
                    )
            self._post_downstream_playback(response_id, phase, ctx_time, seq)
            return
        # ⚠️ **回执一律不截断参考轨**（`started` / `ended` / `cancelled` 都不）。
        #
        # 这是踩过两次的地方，两次都是"在回执里按 clock.now() 截断"惹的祸：
        #
        # ① `ended` 也截（早先的 bug）：`tts.end` 到达时浏览器**才刚开始播**
        #    （TTS 是整段一次性送完的，还有 200ms 提前量），于是"当前时刻"
        #    远在整段音频之前 → **整段参考被清掉**。真机实测：TTS 报 5.85s、
        #    ref 写入区间也正好 5.85s，但非零只有 0.6s（`非零 6/152 = 4%`）。
        #
        # ② `cancelled` 也截（本轮）：服务端在打断时已经**精确**截断过了
        #    （见 `ActionExecutor._interrupt_current`，它用
        #    `max(now, started)` 区分"还没起播"与"已播到中途"）。等浏览器的
        #    回执绕一圈回来时，**新句往往已经开始落位** —— 此时再按
        #    `clock.now()` 截一刀，会**把新句的参考切掉一截**
        #    （实测：新句 3.0s → 1.98s），新回复的回声又对不上了。
        #
        # ⚠️ 唯一的例外：`cancelled` 带回 `sample_offset`（前端报的
        # **实际播出**采样数）时，用**观测值**校正参考轨。
        #
        # 为什么这个例外是安全的（而"按 clock.now() 截断"不安全）：
        # 它**锚在本 response 自己的落位起点上**，与"会话现在跑到哪了"
        # 无关 —— 所以新句有没有开始落位都不影响它，不会误伤。
        #
        # 服务端的 `_interrupt_current` 已按预测做过一次截断；这里用真实
        # 观测值再校一次，把"预测与实际"的偏差抹平（这正是用户指出的：
        # 只要参考轨严格跟随**实际播出**，打断后它自然就是对的）。
        if phase == "cancelled":
            # 两个**独立**的实测观测：前端报的实际播出采样数（sample_offset）
            # 与"实际停下的 ctx 时刻 - 落位起点"（stop_ctx）。取**保守**的
            # 那个（更小 = 保留更少）。`resize` 只能缩短，所以取小是安全的；
            # 取大则会留下没播出去的音频当参考，AEC 去追一个不存在的回声。
            keep = int(sample_offset) if sample_offset > 0 else 0
            if stop_ctx > 0.0 and self._armed.get(response_id) is not None:
                span_start = self.ref_track.started_at(response_id)
                stop_at = self.clock.ctx_to_sample(stop_ctx, epoch)
                if span_start is not None and stop_at is not None:
                    by_ctx = max(0, stop_at - span_start)
                    keep = min(keep, by_ctx) if keep > 0 else by_ctx
            if keep > 0:
                n = self.ref_track.resize(response_id, keep)
                if n:
                    logger.info(
                        "按前端**实测**播出量校正 %s：保留 %.2fs，"
                        "清掉其后 %d 采样（%.2fs）—— 预测!=实际 时以此为准",
                        response_id, keep / SR, n, n / SR,
                    )
        self._post_downstream_playback(response_id, phase, ctx_time, seq)
        # ⚠️ 这里**不再**置 playback_active：
        #    `ended`   = 音频**送**完了（流式下一次推完几十秒，送到时往往才刚
        #                起播）—— 拿它当"停止"会在开播两秒后就报停，而耳朵里
        #                还在响。
        #    `cancelled` = 被打断，但浏览器随后会按真实进度报 `stopped`。
        #    两种情况都由浏览器的 playing/stopped 收口（见本函数开头）。
        #
        #    `ended`/`cancelled` 本身**不能删** —— 参考轨的截断与
        #    `sample_offset` 校正还靠它们。

    # ------------------------------------------------------------------ #
    #  播放锚点（armed 承诺）
    # ------------------------------------------------------------------ #

    class _ArmedEvent:
        """一个可被 set 多次的等待点（armed 可能比执行器的等待先到）。"""
        __slots__ = ("evt", "value")

        def __init__(self) -> None:
            self.evt = asyncio.Event()
            self.value: Optional[int] = None

        def put(self, value: Optional[int]) -> None:
            self.value = value
            self.evt.set()

    def _note_armed(self, response_id: str, start_ctx: float,
                    epoch: int = 0) -> None:
        """记录浏览器承诺的起播时刻，唤醒等待中的执行器。

        ⚠️ 换算不出来（锚点还不够、context 换了）也要唤醒 —— 否则执行器
        会一直干等到超时，白白给每句话加 500ms 延迟。此时 value 为 None，
        执行器退回预测路径。
        """
        at: Optional[int] = None
        if start_ctx > 0.0:
            at = self.clock.ctx_to_sample(start_ctx, epoch)
        if at is not None:
            self._armed[response_id] = at
            self.anchor_source = "ack"
        wait = self._armed_evt.get(response_id)
        if wait is not None:
            wait.put(at)

    def arm_waiter(self, response_id: str) -> "_ArmedEvent":
        """为某个 response 注册等待点（执行器在发 tts.start 之后调用）。"""
        w = self._ArmedEvent()
        if response_id in self._armed:
            # 回执先到了（本地回环/极快网络）—— 直接给值，别让它等超时
            w.put(self._armed[response_id])
        self._armed_evt[response_id] = w
        return w

    def release_waiter(self, response_id: str) -> None:
        self._armed_evt.pop(response_id, None)

    async def wait_armed(self, response_id: str,
                         timeout: float) -> Optional[int]:
        """等浏览器承诺的起播时刻；超时或换算不出返回 None（→ 预测回退）。"""
        w = self._armed_evt.get(response_id)
        if w is None:
            return None
        try:
            await asyncio.wait_for(w.evt.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            return None
        return w.value

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
            self._dump_aec_out(seg)
            self.stats["aec_segments_out"] += 1
            self.stats["aec_samples_out"] += int(np.asarray(seg).size)
            self.metrics.inc("aec_segments")
            # 实时 ERLE 的输出侧：只看**测量窗口内**的输出。
            # （早先无条件累加 → 分母被空闲时段的输出灌大，比值失真）
            if self._erle_active:
                a = np.asarray(seg)
                self._erle_cur_aec += float(np.mean(a ** 2)) * a.size
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
        from .asr.client import (
            extract_turnsense, parse_confidence, parse_final,
        )
        from .downstream.interface import (
            AsrFinal, AsrPartial, AsrTurnSense,
        )
        from .protocol import AsrDisplay

        def _post_turnsense(ts: dict) -> None:
            self.post_downstream(AsrTurnSense(
                t=self.clock.now(),
                label=ts["label"] or "invalid",
                probabilities=ts["probabilities"],
                segment_start_ms=ts["segment_start_ms"],
                segment_end_ms=ts["segment_end_ms"],
                speech_duration_s=ts["speech_duration_s"],
            ))

        def on_message(msg: dict) -> None:
            mode = str(msg.get("mode") or "")
            # state 由 AsrClient.recv_loop 在本回调之前归纳好，这里直接取。
            st = self.asr.state.to_dict() if self.asr else None
            # ⚠️ 「UI 上没有 ASR 文字」有两种完全不同的原因：服务端**没收到**
            #    结果，和收到了但**没下发 UI**。这条日志把两者分开 —— 它记录
            #    服务端回来的每一条原始消息的关键字段（默认关闭，零开销；
            #    开 ORCH_LOG_LEVEL=DEBUG 即可）。
            #    ⚠️ 往这里加日志必须放在**各 return 之前**，否则会漏掉其中
            #    一类（turnsense 那条就走 early return）。
            # ⚠️ 拉置信度必须走 `parse_online_confidence`（先标量、后对象），
            #    **不能只看 `msg["confidence"]`** —— 两个服务端字段形态不同：
            #    asr-2pass(C++) 流式走标量 `online_confidence`，
            #    Fun-ASR(Python) 走 `confidence` 对象。
            #    早先这里只打对象，于是 C++ 的帧在日志里全显示 conf=None，
            #    看起来像"这批帧没带置信度"，实际有值 —— 把人带偏过。
            from .asr.client import parse_online_confidence as _poc
            logger.debug("[%s] ASR<- mode=%r text=%r is_final=%s "
                         "conf=%s（原始 online_conf=%s obj=%s）ts=%s",
                         self.session_id, mode, (msg.get("text") or "")[:60],
                         msg.get("is_final"), _poc(msg),
                         msg.get("online_confidence"),
                         (msg.get("confidence") or {}).get("avg")
                         if isinstance(msg.get("confidence"), dict) else None,
                         (msg.get("turnsense") or {}).get("label")
                         if isinstance(msg.get("turnsense"), dict) else None)

            if mode == "turnsense":
                ts = extract_turnsense(msg)
                if ts is not None:
                    _post_turnsense(ts)
                return

            if mode == "2pass-online":
                delta = msg.get("text", "")
                # ⚠️ 2pass-online 是**增量片段**（"今天" / "吃饭" / "了吗"），
                # 不是累积文本 —— 官方客户端也是自己 += 拼起来显示的
                # （funasr_wss_client.py: `text_print_2pass_online += text`）。
                # 必须自己累积，否则 UI 上只剩最后一个词。
                #
                # ⚠️ 空文本帧**也要下发**：低置信度被服务端过滤掉时文本就是空的，
                # 那恰恰是「用户在出声但没识别出来」的信号，state 里能看出来。
                # 早先这里有 `if delta:` 守卫，把这类帧整条丢了。
                self._asr_online_text += delta
                self.post_downstream(AsrPartial(
                    t=self.clock.now(), text=self._asr_online_text,
                    confidence=parse_confidence(msg),
                    segment_id=self.asr.partials,
                    state=st,
                ))
                self._send_display(AsrDisplay(
                    phase="partial", text=self._asr_online_text,
                    t_ms=int(self.clock.seconds() * 1000),
                    state=st,
                ))
                return

            # 2pass-offline（最终结果）。⚠️ turnsense 也可能**嵌在这条消息里**
            # （流结束的收尾帧走这条路径，没有独立的 mode=turnsense 消息）——
            # 早先只认独立消息，漏掉了它。
            embedded_ts = extract_turnsense(msg)
            if embedded_ts is not None:
                _post_turnsense(embedded_ts)

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
                state=st,
            ))
            # 最终结果到达：用它覆盖流式累积文本，并**清空缓冲**为下一段
            # 做准备（服务端也是这么做的：text_print_2pass_online = ""）
            self._asr_online_text = ""
            if fin["text"]:
                self._send_display(AsrDisplay(
                    phase="final", text=fin["text"],
                    t_ms=int(self.clock.seconds() * 1000),
                    state=st,
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
                ok = await self.omni.trigger_reply(text)
                if not ok:
                    # ⚠️ **必须报出来**。早先这里忽略返回值，于是连接断掉后
                    #    ASR 照常识别、触发却静默失败 —— 用户看到的现象是
                    #    "能识别但永远不回复"，很难定位（真机实测）。
                    self.stats["omni_trigger_failed"] = \
                        self.stats.get("omni_trigger_failed", 0) + 1
                    logger.warning(
                        "[%s] ASR 触发未能送出（OmniLLM 连接可能已断）—— "
                        "本轮不会有回复", self.session_id)
            except Exception as exc:  # noqa: BLE001
                logger.warning("触发 OmniLLM 回复失败: %s", exc)

        self._trigger_task = asyncio.create_task(_do())

    def _send_display(self, msg, *, critical: bool = False) -> bool:
        """投递 UI 消息。返回**是否入队成功**。

        ASR 的回调是同步的，不能在里面 await；入队后由 ``run_display``
        异步取出发送。

        ``critical=True`` 的消息（**IC 动作**）**不允许丢** —— 队列满时
        腾掉最旧的**可丢**消息给它让位。

        ⚠️ 为什么 IC 动作必须特殊对待：离线判题（D16/D18）**只看客户端
        收到的 `ic` 消息**。`GREET` 这类动作**一瞬就过去**（下一拍就变
        `HOLD`），一旦这条显示消息被丢，判题侧永远看不到 —— 而播报其实
        正常发生了（走的是另一条路：`_dispatch` 返回 `Speak`）。现象是
        「喇叭响了、dump 里没有 GREET」，极难定位（实测踩过）。

        早先所有消息同等对待、满了就丢、且**丢弃只记一个从没被暴露过的
        计数器** —— 所以丢了也无人知晓。
        """
        try:
            self._display_q.put_nowait(msg)
            return True
        except asyncio.QueueFull:
            if not critical:
                self.stats["display_dropped"] = \
                    self.stats.get("display_dropped", 0) + 1
                return False
            # 关键消息：腾掉最旧的一条**非关键**消息让位。
            # 直接丢队首风险太大（可能又丢了一条关键消息），所以从队首
            # 找到第一条非关键的丢掉。
            return self._force_enqueue_critical(msg)

    def _force_enqueue_critical(self, msg) -> bool:
        """队列满时为关键消息腾位：丢掉最早的**非关键**消息。

        关键消息带 ``_critical`` 标记；非关键的（人脸/ASR 字幕/SessionStats）
        丢一条不影响正确性 —— 它们下一拍还会再来（或被下游容忍缺失）。
        """
        q = self._display_q
        kept: list = []
        dropped_one = False
        while True:
            try:
                old = q.get_nowait()
            except asyncio.QueueEmpty:
                break
            if not dropped_one and not getattr(old, "_critical", False):
                dropped_one = True          # 丢掉它，给新消息腾位
                self.stats["display_evicted"] = \
                    self.stats.get("display_evicted", 0) + 1
                continue
            kept.append(old)
        # 把保留的放回去，再放新的关键消息
        for m in kept:
            try:
                q.put_nowait(m)
            except asyncio.QueueFull:
                break
        try:
            q.put_nowait(msg)
            return True
        except asyncio.QueueFull:
            # 全队列都是关键消息 —— 这种情况不该发生（关键消息很少），
            # 真发生了就记下来，别静默丢。
            self.stats["display_dropped_critical"] = \
                self.stats.get("display_dropped_critical", 0) + 1
            logger.error("[%s] 显示队列全是关键消息，无法腾位 —— "
                         "这条 IC 动作会丢（判题侧可能漏记）", self.session_id)
            return False

    async def run_face_signals(self) -> None:
        """把人脸线程的回调转成 downstream 事件 + UI 消息。

        ``FaceWorker`` 在专用线程里跑，回调是同步的 —— 这里用队列解耦，
        绝不阻塞人脸线程。
        """
        if self.face_worker is None:
            return
        # 上次投给下游的 face state_seq（去重用，见下面的 obs 分支）
        self._last_face_state_seq = -1
        from .downstream.interface import (
            FaceIdentity, FaceLipState, FaceState, FaceWake,
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
                    # ⚠️ person_id 只是**参考值** —— 上游从名字正则解析
                    #    `person_N`，而线上人脸库存的是真名，解析全失败 →
                    #    所有人都是同一个兜底值。**认人请用 uid**。
                    "person_id": int(ev.person_id),
                    "uid": (ev.state or {}).get("identity_id"),
                    "identity_state": ev.identity_state,
                    "track_id": ev.track_id,
                    "src_w": fs[0] if fs else None,
                    "src_h": fs[1] if fs else None,
                }
                # G1 每帧 state 快照（≈208ms 一次）。挂在观测上随 face.state
                # 一起发 —— 前端不必为它单开一条通道。
                if ev.state is not None:
                    self._last_face_state = ev.state
                    # provider 只在刷新帧给 `state`，所以「有就给」即是去重，
                    # 25Hz 的观测不会把下游淹掉。InteractionCore 需要这些档位
                    # 来判 SOP（face_present / bbox / track / dwell / identity）。
                    #
                    # ⚠️ **不要按 `state_seq` 去重**：身份结果回来时 seq 可能
                    #    不 +1，按 seq 比会丢掉带身份的那一帧。
                    seq = int(ev.state_seq)
                    self._last_face_state_seq = seq
                    self.post_downstream(FaceState(
                        t=self.clock.now(), state=dict(ev.state),
                        state_seq=seq,
                    ))
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
                    "identity_state": ev.identity_state,
                }
                self._push_face_display()

    def _notify_playback_active(self, active: bool) -> None:
        """告诉 InteractionCore「TTS 在不在出声」。

        ⚠️ PRD §3.2 明确：表达层事实（播完 / 已开口 / 是否在播）**由表达层写**
        —— 我们就是表达层，且只有我们知道真实播放状态。IC 的 SOP 07（抢话
        停播）/ 23 / 39 全靠它。
        """
        ic = getattr(self, "interaction", None)
        if ic is None:
            return
        try:
            ic.ic.apply("apply_playback_active", playback_active=bool(active))
        except Exception as exc:  # noqa: BLE001
            logger.debug("回写 playback_active 失败: %s", exc)

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
            # 与会话采样轴对齐（0 = 首块音频）—— 与 AsrDisplay.t_ms 同一口径
            t_ms=int(self.clock.seconds() * 1000),
            tracks=[self._last_face] if getattr(self, "_last_face", None) else [],
            identity=getattr(self, "_last_identity", None),
            wake=getattr(self, "_last_wake", None),
            state=getattr(self, "_last_face_state", None),
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
            self._check_audio_watchdog()
            # ASR「本轮结论」的过期检查 —— **必须周期性地推**。
            #
            # ⚠️ 「说完一句就静音」时**没有任何新 ASR 消息**，而清空逻辑
            #    （`expire_if_idle`）只在收到消息或读快照时才会被调用 ——
            #    两者在静音下永不相遇，于是 `完`/`信`/`transcript` 会**永远
            #    挂着**（实测：静音后 `完` 恒为 HIGH）。tick 是唯一与
            #    ASR 消息无关的时钟，所以挂这里。
            #
            #    这与 `_check_audio_watchdog` 是**同一类坑**：那个也不能挂在
            #    `_periodic_diag`（由 on_audio 调用，收不到音频时自己就不跑）。
            if self.asr is not None:
                self.asr.expire_if_idle()
                self._push_asr_state_if_changed()
            self.post_downstream(Tick(t=self.clock.now()))

    def _push_asr_state_if_changed(self) -> None:
        """ASR 状态**变了就推**（IC 忠实内部；UI 文字单独保留）。

        ⚠️ **为什么必须在 tick 里推**：`AsrDisplay` / `apply_vad` 原先只在
        `on_message`（收到 ASR 消息）里发。而「说完一句就静音」时**没有新
        消息** —— 状态被 tick 清成 NONE 了，却没人告诉 IC / UI，它们永远
        停在最后一条消息的快照上（实测现象：一轮结束后页面 `说`/`抢` 仍是
        HIGH、`完`/`信` 也一直是 HIGH）。
        这与「清空逻辑绑在读快照上」是同一类毛病，只是发生在**推送**环节。

        去重比对整份快照（5 个字段），变了才推 —— 静音时只在归零那一刻推
        一次，之后每拍比对的成本可忽略。
        """
        if self.asr is None:
            return
        cur = self.asr.state.to_dict()
        if cur == self._last_asr_pushed:
            return
        self._last_asr_pushed = cur
        # ⚠️ 排查用：这条日志能直接回答「静音后到底推没推」。
        #    （默认 DEBUG；前面几次排查都因为看不到这一步而只能推断）
        logger.debug("[%s] ASR 状态变化 → 推下游: 说=%s 抢=%s 信=%s 完=%s text=%r",
                     self.session_id, cur.get("user_speaking_confidence"),
                     cur.get("barge_in_confidence"), cur.get("asr_confidence"),
                     cur.get("turn_complete_confidence"),
                     (cur.get("transcript") or "")[:20])

        from .downstream.interface import AsrStateUpdate
        from .protocol import AsrDisplay

        # ---- ① IC：**与内部状态完全一致**（transcript 归零后就是空串）----
        self.post_downstream(AsrStateUpdate(t=self.clock.now(), state=cur))

        # ---- ② UI：state 用当前的（已归零），text 保留最后一句 ----
        # 文字不随内部 `transcript` 归零而消失 —— 用户要一直看得到刚才说了
        # 什么，直到下一句开始才被替换（那时 `text` 又非空，自然覆盖）。
        if cur.get("transcript"):
            self._last_asr_text = cur["transcript"]
        self._send_display(AsrDisplay(
            phase="partial", text=self._last_asr_text,
            t_ms=int(self.clock.seconds() * 1000), state=cur,
        ))

    async def _loop_lag_probe(self) -> None:
        """事件循环卡顿看门狗：每 0.5s 唤醒一次，测**实际**被推迟了多久。

        为什么需要：本服务所有活儿都在**同一个事件循环**里 —— mic 收流、
        扇出给 ASR/Omni（都是 await 网络发送）、TTS、UI 推送、人脸信号转投。
        任何一处跑了个**同步阻塞**调用（GIL 下的重活、第三方库的阻塞 IO、
        大数组在 C 里跑很久），整个循环就会停摆：
        ASR 不再收音频、人脸不再出框、ping 不再回应（→ gateway 20s 后
        以 `1011 keepalive ping timeout` 断开 Omni）—— 现象就是
        「几十秒后整个卡住，ASR 和视频都没反应」，而**日志里什么都看不到**
        （因为打日志本身也要靠循环）。真机踩过。

        光看"最后一次心跳在什么时候"分不出「循环卡住」和「没有事件」，
        所以这里测的是 ``sleep`` 的**实际延迟**：它是循环健康度的直接测量，
        与有没有业务事件无关。超阈值时顺带把最可能阻塞的那几项计数打出来。
        """
        while not self.closed:
            t0 = time.monotonic()
            await asyncio.sleep(_LOOP_LAG_INTERVAL)
            lag = time.monotonic() - t0 - _LOOP_LAG_INTERVAL
            if lag < _LOOP_LAG_WARN:
                self._loop_lag_max = max(getattr(self, "_loop_lag_max", 0.0), lag)
                continue
            self._loop_lag_max = max(getattr(self, "_loop_lag_max", 0.0), lag)
            logger.warning(
                "[%s] ⚠️ 事件循环卡顿 %.0fms（阈值 %.0fms，本会话最大 %.0fms）"
                "—— ASR/人脸/Omni 都会同时停摆。当时计数：音频块=%d 下游积压=%d"
                " 人脸帧=%d 人脸丢帧=%d ASR发=%s",
                self.session_id, lag * 1000, _LOOP_LAG_WARN * 1000,
                self._loop_lag_max * 1000,
                self.stats.get("audio_chunks_in", 0),
                self._down_q.qsize() if getattr(self, "_down_q", None) else -1,
                self.stats.get("video_face_frames", 0),
                self.stats.get("face_dropped", 0),
                getattr(getattr(self, "asr", None), "chunks_sent", "?"))

    def _check_audio_watchdog(self) -> None:
        """会话建立后一直没收到音频 → 告警（每 10s 一次，不刷屏）。

        ⚠️ **必须挂在 tick 上，不能挂在 ``_periodic_diag`` 里** —— 后者由
        ``on_audio`` 调用，收不到音频时它自己就不会跑，这个告警永远不会触发
        （正是它要报的那个场景）。踩过。

        首块音频是整条链路的**起点**：它没来，服务端一切正常、日志干净，
        而页面上既没有 ASR、也没有任何报错。「web 端没有 ASR 结果」绝大多数
        就是这一种 —— 采集没起来 / AudioContext 停在 suspended / worklet
        没启动。这条日志把「没收到音频」与「收到了但没识别」一刀切开。
        """
        if self.stats.get("audio_chunks_in", 0) > 0:
            return
        waited = time.monotonic() - self._t_audio_wait_start
        if waited < 10.0:
            return
        now = time.monotonic()
        if now - getattr(self, "_last_audio_warn_at", 0.0) < 10.0:
            return
        self._last_audio_warn_at = now
        logger.warning(
            "[%s] 已过 %.0fs 仍未收到**任何音频块** —— 浏览器没在发音频。"
            "查页面「诊断」面板的 AudioContext 状态（suspended 则点一下页面）"
            "与「音频块」计数；Python 客户端正常而只有 web 端如此，问题在浏览器侧。",
            self.session_id, waited)

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
        # 音视频转储落盘（默认关闭，无副作用）
        try:
            self.flush_audio_dump()
        except Exception as exc:  # noqa: BLE001
            logger.warning("音频转储写入失败: %s", exc)
        try:
            self.flush_video_dump()
        except Exception as exc:  # noqa: BLE001
            logger.warning("视频转储写入失败: %s", exc)
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
