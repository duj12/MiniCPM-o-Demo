"""声学延迟的**主动校准**。

## 为什么需要主动校准

自适应估计只在 TTS 播放窗口内有素材，且第一句播报时用的还是不准的 D
（消得不好）。主动校准让用户在开始对话前花 ~2 秒测一次，之后所有播报
都是对齐的。

## 校准信号为什么用啁啾而非语音

GCC-PHAT 靠互相关的峰值位置定延迟。语音是**准周期**的，自相关有多个
相近的峰，容易选错；宽带线性调频（chirp）的自相关是尖锐单峰，鲁棒得多。
同时避开人声频段的中低频段，听感上是一声短促的"啾"。

## 流程

  1. 服务端生成一段 1.5s 的啁啾
  2. 通过**与 TTS 相同的通道**送给浏览器播放（复用 tts.audio 消息）
     —— 必须是同一条播放路径，否则测的是另一条链路的延迟
  3. 播放期间采集麦克风
  4. GCC-PHAT 互相关 → 峰值位置 = 往返延迟
  5. 扣除已知的播放提前量，得到声学+网络延迟 D
  6. 更新参考轨并持久化
"""
from __future__ import annotations

import logging
import time
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

CAL_SR = 24000          # 与 TTS 输出一致（走同一条播放路径）
CAL_DURATION_S = 1.5
# 人声主要能量在 300~3400Hz，把啁啾放在 500~4000Hz 仍可听但更易分辨，
# 且低于扬声器/麦克风的截止频率
CAL_F_START = 500.0
CAL_F_END = 4000.0


def make_chirp(sr: int = CAL_SR, duration_s: float = CAL_DURATION_S,
               f0: float = CAL_F_START, f1: float = CAL_F_END) -> np.ndarray:
    """生成线性调频校准信号（int16）。两端加淡入淡出避免爆音。"""
    n = int(sr * duration_s)
    t = np.arange(n, dtype=np.float64) / sr
    # 线性调频：相位是频率的积分
    phase = 2 * np.pi * (f0 * t + (f1 - f0) / (2 * duration_s) * t ** 2)
    x = np.sin(phase)
    # 5ms 淡入淡出
    ramp = int(0.005 * sr)
    if ramp > 0 and 2 * ramp < n:
        x[:ramp] *= np.linspace(0, 1, ramp)
        x[-ramp:] *= np.linspace(1, 0, ramp)
    return (x * 0.35 * 32767).astype(np.int16)


class CalibrationSession:
    """一次校准的生命周期。

    用法::

        cal = CalibrationSession(sr=16000)
        chirp24 = cal.start()              # 拿到要播放的信号（24kHz int16）
        ... 送浏览器播放 ...
        cal.feed(mic_chunk_16k)            # 播放期间持续喂麦克风
        d_samples = cal.finish()           # 返回延迟（采样，16k 基准）
    """

    def __init__(self, sr: int = 16000, search_ms: float = 1500.0) -> None:
        self.sr = sr
        self.search = int(sr * search_ms / 1000.0)
        self.chirp24 = make_chirp()
        # 降采样到 16k 用于互相关（与麦克风同域）
        self.chirp16 = self._resample(self.chirp24.astype(np.float32) / 32768.0,
                                      CAL_SR, sr)
        self._mic: list = []
        self._mic_len = 0
        self._started_at: Optional[float] = None
        self.done = False

    @staticmethod
    def _resample(x: np.ndarray, src: int, dst: int) -> np.ndarray:
        if src == dst:
            return x
        n_out = int(len(x) * dst / src)
        pos = np.linspace(0, len(x) - 1, n_out)
        i0 = np.floor(pos).astype(int)
        i1 = np.minimum(i0 + 1, len(x) - 1)
        fr = (pos - i0).astype(np.float32)
        return (x[i0] * (1 - fr) + x[i1] * fr).astype(np.float32)

    # ------------------------------------------------------------------ #

    def start(self) -> np.ndarray:
        """开始校准，返回要播放的 24kHz int16 信号。"""
        self._mic = []
        self._mic_len = 0
        self.done = False
        self._started_at = time.monotonic()
        logger.info("校准开始：播放 %.2fs 啁啾（%.0f→%.0fHz）",
                    len(self.chirp24) / CAL_SR, CAL_F_START, CAL_F_END)
        return self.chirp24

    def feed(self, mic_16k: np.ndarray) -> None:
        """喂入麦克风采样（16kHz，AEC **之前**的原始信号才有回声）。"""
        if self.done:
            return
        x = np.asarray(mic_16k, dtype=np.float32).reshape(-1)
        self._mic.append(x)
        self._mic_len += x.size

    def expected_total_s(self) -> float:
        """预计需要采集多久（播放时长 + 最大搜索延迟 + 余量）。"""
        return len(self.chirp24) / CAL_SR + self.search / self.sr + 0.5

    def ready(self) -> bool:
        """采集是否足够。"""
        return self._mic_len >= int(self.expected_total_s() * self.sr)

    # ------------------------------------------------------------------ #

    def finish(self, playback_delay_ms: float = 200.0) -> Optional[dict]:
        """计算延迟。返回 ``{delay_ms, peak_ratio, ok}`` 或 None。

        得到的绝对延迟里含**播放提前量**（服务端在送音频时就预留的），
        参考轨落位时已经把它算进去了，故这里要扣除。
        """
        if self.done:
            return None
        self.done = True
        if not self._mic:
            logger.warning("校准失败：没收到麦克风数据")
            return None

        mic = np.concatenate(self._mic)
        ref = self.chirp16
        if mic.size < ref.size + 1024:
            logger.warning("校准失败：麦克风数据不足（%d < %d）",
                           mic.size, ref.size)
            return None

        # 准备播放起点：服务端送音频的时刻 ≈ 采集开始后 playback_delay
        lead = int(playback_delay_ms * self.sr / 1000.0)

        n_fft = 1
        total = mic.size + ref.size
        while n_fft < total:
            n_fft <<= 1
        X = np.fft.rfft(mic, n_fft)
        Y = np.fft.rfft(ref, n_fft)
        R = X * np.conj(Y)
        mag = np.abs(R)
        mag[mag < 1e-10] = 1e-10
        cc = np.fft.irfft(R / mag, n_fft)

        # 搜索范围：从 0 到 search（mic 滞后 ref）
        win = min(self.search, len(cc) - 1)
        seg = np.abs(cc[:win + 1])
        peak = int(np.argmax(seg))
        # 置信度：峰值 / 次峰
        tmp = seg.copy()
        tmp[max(0, peak - 32):peak + 33] = 0
        ratio = float(seg[peak] / tmp.max()) if tmp.max() > 0 else 0.0

        # 扣除播放提前量得到"真实"声学+网络延迟
        d_samples = peak - lead
        d_ms = d_samples / self.sr * 1000.0
        ok = ratio > 1.8 and 0 <= d_ms <= 1500
        logger.info(
            "校准完成：相关峰 %d 采样，提前量 %d，净延迟 %.0fms"
            "（峰值比 %.2f，%s）",
            peak, lead, d_ms, ratio, "可信" if ok else "**置信度低**",
        )
        return {"delay_ms": d_ms, "delay_samples": d_samples,
                "peak": peak, "peak_ratio": round(ratio, 2), "ok": ok}
