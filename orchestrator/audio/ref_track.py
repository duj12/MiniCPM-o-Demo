"""TTS 参考轨（AEC 的 farend 来源）。

回答一个问题：**此刻从扬声器出来的是什么，在麦克风时钟上？**

构造方式：
  - **内容**来自我们持有的 TTS PCM（24kHz int16，来自 TTS 服务）
  - **位置**来自浏览器的播放回执（``playback.started`` 带 ``ctx_time``）
  - **延迟**由 ``AcousticDelayTracker`` 测得（扬声器→麦克风的物理延迟）

为什么不能只用"到达时刻"：TTS 音频到达 orchestrator 的时刻 ≠ 它在浏览器
播放的时刻。中间隔着发送、抖动缓冲、以及播放器的 200ms 提前量。而 AEC
的参考必须与**实际播出**的时刻对齐，否则消不掉回声。

三个容易漏掉、漏了就静默失效的点（见各方法注释）：
  1. 重采样器必须有状态（否则块边界失配）
  2. 浏览器播放走另一条 24k→ctx_rate 滤波路径，残余差异构成 ERLE 地板
  3. 回执时刻相对于 ``clock.now()`` 在**未来**（播放提前量），不要 clamp
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from ..clock import SR, TrackBuffer
from .resample import PassthroughResampler, StatefulResampler

# TTS 服务的输出采样率（实测确认：24kHz int16）
TTS_SR = 24000


@dataclass
class PlaybackChunk:
    """一段被调度播放的音频在轨上的落位记录。"""

    response_id: str
    seq: int
    t0: int          # 会话采样轴上的起点（已扣除声学延迟）
    t1: int
    ctx_time: float  # 浏览器给出的播放时刻（AudioContext 秒）


class RefTrack:
    """扬声器输出信号在麦克风时钟上的表示，16kHz float32。

    未写入区域读出为 0（语义正确：没在播就是静音）。
    """

    def __init__(self, sr: int = SR, ring_seconds: float = 30.0,
                 tts_sr: int = TTS_SR) -> None:
        self.sr = sr
        self.buf = TrackBuffer(int(sr * ring_seconds))
        # 必须有状态：TTS 分块到达，逐块独立重采样会在边界产生相位不连续，
        # 而 AEC 靠逐样本匹配参考 —— 边界失配会让 ERLE 周期性塌陷。
        self._rs = (StatefulResampler(tts_sr, sr) if tts_sr != sr
                    else PassthroughResampler())

        # 声学路径延迟（采样数）。由 AcousticDelayTracker 更新。
        self.delay_samples = 0

        # 播放锚点：把浏览器的 ctx_time 映射到会话采样轴
        self._anchor_ctx: Optional[float] = None
        self._anchor_sample: Optional[int] = None

        # 每个 response 已落位的区间，供 truncate 使用
        self._placed: Dict[str, List[PlaybackChunk]] = {}

    # ------------------------------------------------------------------ #

    def set_anchor(self, ctx_time: float, sample_pos: int) -> None:
        """设定 ctx_time → 采样轴的锚点。

        在收到首个 ``playback.started`` 时调用。浏览器每次会话重建
        AudioContext 后需要重设。
        """
        self._anchor_ctx = ctx_time
        self._anchor_sample = sample_pos

    def ctx_to_sample(self, ctx_time: float) -> int:
        """把浏览器的 AudioContext 时刻换算到会话采样轴。

        ⚠️ 返回值可能**大于** ``clock.now()`` —— 这是正确的。播放器有
        200ms 提前量，回执到达时那个时刻还没到。**不要 clamp 到 now**，
        否则参考轨会与实播错位一个提前量。
        """
        if self._anchor_ctx is None or self._anchor_sample is None:
            # 没有锚点：退化到"立即"，调用方应先 set_anchor
            return self._anchor_sample or 0
        delta_s = ctx_time - self._anchor_ctx
        return self._anchor_sample + int(round(delta_s * self.sr))

    # ------------------------------------------------------------------ #

    def place(self, response_id: str, seq: int, pcm24: np.ndarray,
              ctx_time: float) -> PlaybackChunk:
        """把一段 TTS PCM 按 ``ctx_time`` 落位到轨上。

        ``pcm24`` 是 24kHz int16（TTS 服务的原始输出）。返回落位记录。
        """
        if pcm24.ndim != 1:
            raise ValueError("pcm24 必须是 1-D")
        # 重采样为 16k float32（保留跨块状态）
        x16 = self._rs.process(pcm24)
        t_start = self.ctx_to_sample(ctx_time)
        # 扣除声学延迟：ref[t - D] 的回声出现在 mic[t]
        write_at = t_start - self.delay_samples
        self.buf.write(write_at, x16)
        chunk = PlaybackChunk(
            response_id=response_id, seq=seq,
            t0=write_at, t1=write_at + x16.shape[0], ctx_time=ctx_time,
        )
        self._placed.setdefault(response_id, []).append(chunk)
        return chunk

    def truncate(self, response_id: str, from_ctx_time: Optional[float] = None) -> int:
        """把某个 response **尚未播出**的参考清零。

        取消播放（barge-in / tts.cancel）时必须调用。否则 AEC 会拿到一段
        没有对应回声的参考信号，它会**主动误适配**去追这个不存在的回声 ——
        比不给参考更糟。

        ``from_ctx_time`` 为取消发生的浏览器时刻（None = 整个 response 全清）。
        返回被清零的采样数。
        """
        chunks = self._placed.pop(response_id, [])
        if not chunks:
            return 0
        if from_ctx_time is None:
            lo = min(c.t0 for c in chunks)
            hi = max(c.t1 for c in chunks)
        else:
            cut = self.ctx_to_sample(from_ctx_time) - self.delay_samples
            lo = cut
            hi = max(c.t1 for c in chunks)
        if hi <= lo:
            return 0
        self.buf.zero_range(lo, hi)
        return hi - lo

    def read(self, t0: int, n: int) -> np.ndarray:
        """读 [t0, t0+n) 的参考信号（16kHz float32，未写入处为 0）。"""
        return self.buf.read(t0, n)

    def is_active(self, t: int, lookahead: int = 0) -> bool:
        """t 处（或未来 lookahead 采样内）是否有非零参考。

        用于判断"用户是否可能听到 TTS"——决定是否把真实 ref 送给 AEC。
        """
        span = self.buf.written_span()
        if span[0] is None:
            return False
        lo, hi = span
        return not (t + lookahead < lo or t > hi)

    def reset(self) -> None:
        self._rs.reset()
        self._placed.clear()
        self._anchor_ctx = None
        self._anchor_sample = None


class AcousticDelayTracker:
    """估计扬声器→麦克风的声学路径延迟 D。

    ⚠️ 这是**必须由我方处理**的一环：云端 AEC（`speech_frontend` 的
    `StreamInference`）把 nearend 与 farend 按**相同偏移**推入，即假定两者
    已样本对齐。仓库里的 `GCCPHATDelayEstimator` 从未被流式路径 import，
    且它依赖的 `xmov_aec/config/aec_config.py` 在仓库里不存在（死代码）。

    实现用 GCC-PHAT：对 mic 与 ref 做互功率谱相位变换，峰值位置即延迟。
    只在**播放窗口内**估计（空闲时 ref 为静音，估出来是垃圾）。
    """

    def __init__(self, sr: int = SR, max_delay_ms: float = 300.0,
                 median_window: int = 15) -> None:
        self.sr = sr
        self.max_delay = int(sr * max_delay_ms / 1000.0)
        self._history: List[int] = []
        self._median_window = median_window
        self.delay = 0
        self.estimates = 0

    def estimate(self, mic: np.ndarray, ref: np.ndarray) -> Optional[int]:
        """用一对等长信号估计延迟。返回延迟采样数，不足条件返回 None。

        mic/ref 均为 1-D float32。正值 = ref 领先 mic（即 mic 是 ref 的延迟副本）。
        """
        if mic.shape != ref.shape or mic.shape[0] < 1024:
            return None
        if float(np.sqrt(np.mean(ref ** 2))) < 1e-4:
            return None  # ref 近似静音，估不出
        if float(np.sqrt(np.mean(mic ** 2))) < 1e-5:
            return None

        n = mic.shape[0]
        n_fft = 1
        while n_fft < 2 * n:
            n_fft <<= 1
        X = np.fft.rfft(mic, n_fft)
        Y = np.fft.rfft(ref, n_fft)
        # GCC-PHAT：只保留相位
        R = X * np.conj(Y)
        mag = np.abs(R)
        mag[mag < 1e-10] = 1e-10
        cc = np.fft.irfft(R / mag, n_fft)
        # 只看正延迟方向（mic 滞后于 ref），搜索窗限 max_delay
        window = min(self.max_delay, n_fft // 2 - 1)
        peak = int(np.argmax(np.abs(cc[:window + 1])))
        # 置信度：峰值 / 次峰。太低说明估计不可靠。
        cc_abs = np.abs(cc[:window + 1]).copy()
        cc_abs[max(0, peak - 2):peak + 3] = 0
        if cc_abs.max() > 0 and np.abs(cc[peak]) < 1.5 * cc_abs.max():
            return None

        self._history.append(peak)
        if len(self._history) > self._median_window:
            self._history.pop(0)
        self.estimates += 1
        self.delay = int(np.median(self._history))
        return peak

    def reset(self) -> None:
        self._history.clear()
        self.delay = 0
        self.estimates = 0
