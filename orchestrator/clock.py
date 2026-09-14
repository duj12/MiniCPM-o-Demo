"""会话级音频时间基准。

**一个会话一条时间轴**：16kHz 采样计数，从 0 开始单调递增。
mic ingest 是**唯一写入者**；所有其他流（AEC 输出、ASR 输入、OmniLLM 输入、
TTS 参考）都在这个时间轴上**表达**，而不是各自计时。

为什么必须统一：AEC 的参考信号要与麦克风**样本级对齐**。如果各行其是，
云端 AEC 拿到的 ref 与 mic 之间存在未知漂移，回声就消不掉。

设计约束：
  - 时钟**永不回退、永不卡住**。mic 断流时用静音补齐并推进，
    否则后续所有按绝对样本索引写入的 ref 都会错位。
  - 纯整数运算，避免浮点累积误差。
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np  # noqa: F401  (类型注解与调用方都用到)

SR = 16000  # 全链路统一采样率


@dataclass(frozen=True)
class AudioFrame:
    """带时间戳的一段音频。``data`` 为 (C, T) float32，16kHz。"""

    t0: int  # 会话时钟上的首个采样索引
    data: np.ndarray

    @property
    def t1(self) -> int:
        return self.t0 + self.data.shape[1]

    @property
    def n_samples(self) -> int:
        return self.data.shape[1]

    def __post_init__(self) -> None:
        if self.data.ndim != 2:
            raise ValueError(f"data 必须是 (C, T)，得到 shape={self.data.shape}")
        if self.data.dtype != np.float32:
            raise ValueError(f"data 必须是 float32，得到 {self.data.dtype}")


class SampleClock:
    """会话级单调 16kHz 采样计数。

    只有 mic ingest 调用 ``advance()``。其他组件只读 ``now()``。

    同时记录墙钟锚点，用于把浏览器回执里的 ``ctx_time``（AudioContext 秒）
    换算到会话采样轴。
    """

    __slots__ = ("sr", "_t", "_t_wall0")

    def __init__(self, sr: int = SR) -> None:
        self.sr = sr
        self._t = 0
        # ⚠️ 墙钟锚点**不在构造时设** —— 会话构造与开始收流之间可能隔很久
        # （连接外部服务、握手），那段时间不该算进漂移。改为在**第一块
        # 音频**到达时锚定（见 start()）。
        self._t_wall0: Optional[float] = None

    def start(self) -> None:
        """锚定墙钟起点（收到第一块音频时调用）。"""
        if self._t_wall0 is None:
            self._t_wall0 = time.monotonic()

    def advance(self, n: int) -> int:
        """推进 n 个采样，返回推进**后**的位置（即 t1）。"""
        if n < 0:
            raise ValueError("采样数不能为负")
        self._t += n
        return self._t

    def now(self) -> int:
        """当前采样位置（已消费的采样数，即下一个采样的索引）。"""
        return self._t

    def seconds(self) -> float:
        return self._t / self.sr

    def wall_elapsed(self) -> float:
        if self._t_wall0 is None:
            return 0.0
        return time.monotonic() - self._t_wall0

    def drift_samples(self) -> int:
        """采样轴与墙钟的偏差（采样数）。正值 = 时钟落后于实时。

        长会话里这个值应保持在一个小常数附近；持续增长说明有数据丢失
        （调用方没补静音）或积压。未开始收流时返回 0。
        """
        if self._t_wall0 is None:
            return 0
        return int(self.wall_elapsed() * self.sr) - self._t

    def frame_of(self, data: np.ndarray) -> AudioFrame:
        """把一段数据登记到时间轴上，返回带 t0/t1 的帧。"""
        if self._t_wall0 is None:
            self.start()
        t0 = self._t
        self.advance(data.shape[1])
        return AudioFrame(t0=t0, data=data)

    def silence_frame(self, gap_samples: int) -> AudioFrame:
        """补一段静音并推进时钟（mic 断流时用）。"""
        if gap_samples <= 0:
            raise ValueError("gap 必须为正")
        data = np.zeros((1, gap_samples), dtype=np.float32)
        return self.frame_of(data)


class TrackBuffer:
    """按**绝对采样索引**读写的环形缓冲，未写入区域读出为 0。

    参考轨（RefTrack）用它暂存"扬声器正在播什么"。之所以要环形而不是
    简单列表：读取端按 ``read(t0, n)`` 随机访问，且读的位置可能超前于
    写入（TTS 有播放延迟，回执到达时数据已在轨上）。
    """

    __slots__ = ("capacity", "_buf", "_written_lo", "_written_hi")

    def __init__(self, capacity_samples: int) -> None:
        if capacity_samples <= 0:
            raise ValueError("capacity 必须为正")
        self.capacity = capacity_samples
        self._buf = np.zeros(capacity_samples, dtype=np.float32)
        # 已写入的索引区间（半开）；None 表示还没写过
        self._written_lo: Optional[int] = None
        self._written_hi: Optional[int] = None

    def _slice(self, start: int, n: int) -> tuple[np.ndarray, int]:
        """返回 (目标视图, 第一段长度)。调用方负责处理回绕。"""
        pos = start % self.capacity
        first = min(n, self.capacity - pos)
        return self._buf[pos:pos + first], first

    def write(self, t0: int, data: np.ndarray) -> None:
        """在绝对索引 t0 处写入。data 为 1-D float32。"""
        if data.ndim != 1:
            raise ValueError("TrackBuffer.write 只接受 1-D 数据")
        n = data.shape[0]
        if n == 0:
            return
        if n > self.capacity:
            # 超长写入：只保留最后 capacity 个采样
            data = data[-self.capacity:]
            t0 += n - self.capacity
            n = self.capacity
        view, first = self._slice(t0, n)
        view[:] = data[:first]
        if n > first:
            self._buf[: n - first] = data[first:]

        if self._written_lo is None:
            self._written_lo, self._written_hi = t0, t0 + n
        else:
            self._written_lo = min(self._written_lo, t0)
            self._written_hi = max(self._written_hi, t0 + n)

    def read(self, t0: int, n: int) -> np.ndarray:
        """读绝对索引区间 [t0, t0+n)。未写入或已被覆盖的位置返回 0。"""
        out = np.zeros(n, dtype=np.float32)
        if n == 0:
            return out
        if self._written_lo is None:
            return out
        # 与已写区间求交
        lo = max(t0, self._written_lo)
        hi = min(t0 + n, self._written_hi)
        if lo >= hi:
            return out
        span = hi - lo
        pos = lo % self.capacity
        first = min(span, self.capacity - pos)
        out[lo - t0: lo - t0 + first] = self._buf[pos:pos + first]
        if span > first:
            out[lo - t0 + first: hi - t0] = self._buf[: span - first]
        return out

    def zero_range(self, t0: int, t1: int) -> None:
        """把 [t0, t1) 清零（取消播放时截断 ref 尾部用）。

        不清 ``_written_*`` 边界：这些位置**是**被写过的（内容是静音），
        语义上仍然有效。
        """
        if t1 <= t0:
            return
        n = min(t1 - t0, self.capacity)
        pos = t0 % self.capacity
        first = min(n, self.capacity - pos)
        self._buf[pos:pos + first] = 0.0
        if n > first:
            self._buf[: n - first] = 0.0

    def written_span(self) -> tuple[Optional[int], Optional[int]]:
        return self._written_lo, self._written_hi
