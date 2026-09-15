"""TTS 参考轨（AEC 的 farend 来源）。

回答一个问题：**此刻从扬声器出来的是什么，在麦克风时钟上？**

## 时间基准（曾经踩过的坑）

参考轨的**所有位置一律用会话采样时钟**（``SampleClock`` 的计数），
**绝不混用浏览器的 AudioContext 时间**。

早期实现把 ``clock.seconds()``（会话时钟秒）当成浏览器的 ``ctx_time``
传给 ``place()``，而 ``ctx_to_sample()`` 会去减 ``_anchor_ctx``（浏览器
时钟）—— 两个时钟域相减得到约 -1.3e7 的写入位置，``read()`` 因区间
不相交而**恒返回 0**。结果：AEC 的 farend 一直是静音，回声完全不消，
用户自己的 TTS 播报被 ASR 识别成说话。

现在的做法：播放起点由**服务端自己算** —— 送完音频时的会话位置 +
播放提前量（``playback_delay_ms``）。这比用浏览器回执换算更稳，因为
回执要跨两个时钟域且有网络往返；而播放提前量是我们在配置里约定的常量。
浏览器回执仍用于**取消/结束时的截断**（那只需"从现在起"的语义，不依赖
绝对对齐）。

## 声学延迟补偿

扬声器到麦克风有物理延迟 D。设音频为 ``s``，播放起点在会话位置
``T_play``，则 ``mic[t] = s(t - T_play - D)``；而原始轨
``raw[p] = s(p - T_play)``，于是 ``mic[t] = raw[t - D]``。

要让 farend 与 mic 里的回声分量对齐，需 ``ref[t] = mic 中的回声[t]``
``= raw[t - D]``：**把原始轨整体后移 D**（``read(t) = raw.read(t - D)``）。

注意方向 —— 是把参考**推后** D，不是提前。早期实现写成提前，即使
位置算对也会错位 2D。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from ..clock import SR, TrackBuffer
from .resample import PassthroughResampler, StatefulResampler

logger = logging.getLogger(__name__)

# TTS 服务的输出采样率（实测确认：24kHz int16）
TTS_SR = 24000


@dataclass
class PlaybackChunk:
    """一段被调度播放的音频在轨上的落位记录（会话采样单位）。"""

    response_id: str
    seq: int
    t0: int          # 播放起点（口语义上的"开始出声"）
    t1: int
    at_sample: int   # 与 t0 相同，保留字段名便于排查


class RefTrack:
    """扬声器输出信号在麦克风时钟上的表示，16kHz float32。

    内部保存**未补偿**的原始轨（raw），``read()`` 时按 ``delay_samples``
    平移给出补偿后的参考。这样延迟估计可以拿 raw 直接和 mic 相关，
    不会因为补偿而自我抵消。

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
        # 已落位的区间（按 response 分组），供 truncate 用
        self._placed: Dict[str, List[PlaybackChunk]] = {}

    # ------------------------------------------------------------------ #

    def place(self, response_id: str, seq: int, pcm24: np.ndarray,
              at_sample: int) -> PlaybackChunk:
        """把一段 TTS PCM 落位到轨上。

        ``at_sample``：**会话采样时钟**上"开始出声"的位置。
        由调用方按 ``clock.now() + playback_delay_samples`` 计算，
        **不要**传浏览器的 ``ctx_time``。
        """
        if pcm24.ndim != 1:
            raise ValueError("pcm24 必须是 1-D")
        x16 = self._rs.process(pcm24)
        self.buf.write(at_sample, x16)
        chunk = PlaybackChunk(
            response_id=response_id, seq=seq,
            t0=at_sample, t1=at_sample + x16.shape[0], at_sample=at_sample,
        )
        self._placed.setdefault(response_id, []).append(chunk)
        return chunk

    def truncate(self, response_id: str,
                 from_sample: Optional[int] = None) -> int:
        """把某个 response **尚未播出**的参考清零。

        取消播放（barge-in / tts.cancel）时必须调用。否则 AEC 会拿到一段
        没有对应回声的参考信号，它会**主动误适配**去追这个不存在的回声 ——
        比不给参考更糟。

        ``from_sample`` 为取消发生的**会话采样位置**（None = 整个 response）。
        返回被清零的采样数。
        """
        chunks = self._placed.pop(response_id, [])
        if not chunks:
            return 0
        if from_sample is None:
            lo = min(c.t0 for c in chunks)
            hi = max(c.t1 for c in chunks)
        else:
            lo = from_sample
            hi = max(c.t1 for c in chunks)
        if hi <= lo:
            return 0
        self.buf.zero_range(lo, hi)
        return hi - lo

    # ------------------------------------------------------------------ #

    def read_raw(self, t0: int, n: int) -> np.ndarray:
        """读**未做声学延迟补偿**的原始参考。

        延迟估计要用它（与 mic 做互相关求 D）。用补偿后的信号估计会
        自我抵消 —— 补偿多少就测不出多少。
        """
        return self.buf.read(t0, n)

    def read(self, t0: int, n: int) -> np.ndarray:
        """读**已补偿**的参考（给 AEC 用）。

        ``ref[t] = raw[t - D]``：把原始轨整体后移 D，使其与 mic 中的
        回声分量对齐。
        """
        if self.delay_samples == 0:
            return self.buf.read(t0, n)
        return self.buf.read(t0 - self.delay_samples, n)

    def is_active(self, t: int, lookahead: int = 0) -> bool:
        """t 处（或未来 lookahead 采样内）是否有非零参考。"""
        lo, hi = self.buf.written_span()
        if lo is None:
            return False
        return not (t + lookahead < lo or t > hi)

    def reset(self) -> None:
        self._rs.reset()
        self._placed.clear()
        self.delay_samples = 0


class AcousticDelayTracker:
    """估计扬声器→麦克风的声学路径延迟 D（GCC-PHAT）。

    ⚠️ 必须由我方处理：AEC 服务把 nearend/farend 按**相同偏移**推入，
    即假定两者已样本对齐。仓库里的 ``GCCPHATDelayEstimator`` 从未被流式
    路径 import，且它依赖的 ``xmov_aec/config/aec_config.py`` 在仓库里
    **不存在**（死代码），故此处重新实现。

    约定：``mic[t] ≈ ref[t - d]`` 时返回 d，即 **ref 领先 mic** d 个采样
    （mic 是 ref 的延迟副本）。
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
        """用一对等长信号估计延迟。返回本次估计值，不满足条件返回 None。

        mic/ref 均为 1-D float32。正值 = ref 领先 mic。
        """
        if mic.shape != ref.shape or mic.shape[0] < 2048:
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
        cc_abs = np.abs(cc[:window + 1])
        peak = int(np.argmax(cc_abs))
        # 置信度：峰值须显著高于次峰。太低说明估计不可靠。
        tmp = cc_abs.copy()
        tmp[max(0, peak - 2):peak + 3] = 0
        if tmp.max() > 0 and cc_abs[peak] < 1.5 * tmp.max():
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
