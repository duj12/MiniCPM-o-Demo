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
from collections import deque
from dataclasses import dataclass
from typing import Deque, NamedTuple, Optional, Tuple

import numpy as np  # noqa: F401  (类型注解与调用方都用到)

SR = 16000  # 全链路统一采样率

# 锚点窗口：最近 N 个 mic 块参与仿射拟合。100ms/块 → 32 块 ≈ 3.2s。
# 窗口越长越稳（噪声平均掉），越短越能跟上设备时钟的斜率变化。3.2s 是折中：
# 实测两次播放间隔约 15s，10ppm 的斜率误差在这段时间里累积 ≈2.4 采样
# （0.15ms），拟合窗口只要覆盖到「斜率变化的量级」就够，不需要更长。
ANCHOR_WINDOW = 32
# 拟合斜率的合理范围：与标称采样率的相对偏差。设备 ctx 采样率与真实播放
# 时钟的偏差通常是几十 ppm；2e-3（2000ppm）之外一定是选错了点或 context 换了。
ANCHOR_SLOPE_TOL = 2e-3
# 拟合残差上限（ms）。超过说明锚点根本不在一条直线上 —— 不能用。
ANCHOR_RESIDUAL_MS = 20.0


class Anchor(NamedTuple):
    """一个 mic 块的「浏览器时钟 ↔ 会话采样」对应点。

    ``ctx``    —— 该块**首采样**在浏览器 AudioContext 上的时刻（秒）
    ``sample`` —— 该块首采样在会话时钟上的采样索引
    ``wall``   —— 服务端收到它的墙钟（仅诊断用）
    """

    ctx: float
    sample: int
    wall: float


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

    同时维护**浏览器 AudioContext 时钟 → 会话采样轴**的仿射映射
    （见 ``record_anchor`` / ``ctx_to_sample``），这是参考轨精确落位的依据。
    """

    __slots__ = ("sr", "_t", "_t_wall0", "_anchors", "_epoch", "_fit")

    def __init__(self, sr: int = SR) -> None:
        self.sr = sr
        self._t = 0
        # ⚠️ 墙钟锚点**不在构造时设** —— 会话构造与开始收流之间可能隔很久
        # （连接外部服务、握手），那段时间不该算进漂移。改为在**第一块
        # 音频**到达时锚定（见 start()）。
        self._t_wall0: Optional[float] = None
        # ctx ↔ 会话采样的锚点（见 record_anchor）。_fit 是懒计算的缓存。
        self._anchors: Deque[Anchor] = deque(maxlen=ANCHOR_WINDOW)
        self._epoch: Optional[int] = None
        self._fit: Optional[Tuple[float, float]] = None

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

    # ------------------------------------------------------------------ #
    #  浏览器时钟 → 会话采样 的仿射映射
    # ------------------------------------------------------------------ #

    def record_anchor(self, ctx_time: float, sample: int,
                      epoch: int = 0) -> None:
        """登记一个对应点：会话采样 ``sample`` 对应浏览器 ctx 时刻 ``ctx_time``。

        为什么需要它：参考轨必须落在**浏览器实际播出**的时刻上，而播出时刻
        只有浏览器自己知道（`AudioContext.currentTime`）。服务端曾经用
        "送音频时的 clock.now() + 提前量"来**预测**，但 clock.now() 只由
        浏览器送来的 100ms mic 块推进 —— 它比真实时刻慢 0~100ms 且**逐块
        抖动**（浏览器主线程同时在跑 25fps 抓帧）。预测误差因此逐句变化，
        固定常量 D 吸收不了 → 回声消不掉。

        这里改成**测量**：每个 mic 块在发送时报出「本块首采样的 ctx 时刻」。
        两个时钟域数的是**同一路 16kHz 流**，所以 (ctx, sample) 是一对
        **精确**的对应点 —— 映射本身没有量化误差，100ms 的块间隔只影响
        "斜率变化后多快重新收敛"，不影响精度。这正是它和"用锚点估算当前
        偏移"的本质区别：后者会被量化到 ±100ms，在 AEC 需要的 ±5ms 面前
        是死路。

        ``epoch`` 是浏览器 AudioContext 的代号（页面建 context 时生成）。
        异 epoch 的锚点一律丢弃 —— 页面刷新/context 重建后 ``currentTime``
        会归零，混用会让拟合彻底跑偏。

        ⚠️ ``ctx_time <= 0`` 被当作"本次没上报"的哨兵值丢弃。副作用是
        context 刚建好、``currentTime`` 恰好为 0 的那**一个**块会被丢掉 ——
        窗口有 32 块（3.2s），少一个无影响，不值得为它引入 None/可选类型。
        """
        if not (ctx_time > 0.0):
            return
        if self._epoch is None:
            self._epoch = epoch
        elif epoch != self._epoch:
            # 换了 context：旧锚点的时间轴已经无效，全部作废
            self.clear_anchors()
            self._epoch = epoch
        self._anchors.append(Anchor(ctx=float(ctx_time), sample=int(sample),
                                    wall=time.monotonic()))
        self._fit = None

    def clear_anchors(self) -> None:
        self._anchors.clear()
        self._fit = None

    def _ensure_fit(self) -> Optional[Tuple[float, float]]:
        """拟合 ``sample ≈ a·ctx + b``，返回 (a, b)；不可信时返回 None。

        用**最小二乘**而不是"假定斜率 = sr、只取偏移的中位数"：设备时钟有
        几十 ppm 的漂移，固定斜率会让 b 随会话推进单边偏移（10ppm × 15s
        ≈ 2.4 采样，已经吃掉 ±5ms 容忍窗的一半）。拟合出的斜率还兼职做
        "context 换了 / 采样率变了" 的检测。

        斜率或残差不过门时退回固定斜率 + 偏移中位数；仍不过门则返回 None
        （调用方退回预测路径，而不是拿一个错误的映射去落位）。
        """
        if self._fit is not None:
            return self._fit
        n = len(self._anchors)
        if n < 3:
            return None
        cs = [a.ctx for a in self._anchors]
        ss = [float(a.sample) for a in self._anchors]
        mc = sum(cs) / n
        ms = sum(ss) / n
        var = sum((c - mc) ** 2 for c in cs)
        fit: Optional[Tuple[float, float]] = None
        if var > 1e-12:
            cov = sum((c - mc) * (s - ms) for c, s in zip(cs, ss))
            a = cov / var
            if abs(a - self.sr) / self.sr <= ANCHOR_SLOPE_TOL:
                fit = (a, ms - a * mc)
        if fit is None or self._residual_ms(fit) > ANCHOR_RESIDUAL_MS:
            # 退回固定斜率 + 偏移中位数（对离群点稳健）
            offs = sorted(s - self.sr * c for c, s in zip(cs, ss))
            med = offs[len(offs) // 2]
            alt = (float(self.sr), med)
            if self._residual_ms(alt) > ANCHOR_RESIDUAL_MS:
                return None
            fit = alt
        self._fit = fit
        return fit

    def _residual_ms(self, fit: Tuple[float, float]) -> float:
        a, b = fit
        n = len(self._anchors)
        if n == 0:
            return float("inf")
        acc = 0.0
        for an in self._anchors:
            d = (a * an.ctx + b) - an.sample
            acc += d * d
        return (acc / n) ** 0.5 / self.sr * 1000.0

    def ctx_to_sample(self, ctx_time: float,
                      epoch: int = 0) -> Optional[int]:
        """把浏览器 ``ctx_time`` 换算成会话采样位置；不可信时返回 None。"""
        if epoch and self._epoch is not None and epoch != self._epoch:
            return None
        fit = self._ensure_fit()
        if fit is None:
            return None
        a, b = fit
        return int(round(a * ctx_time + b))

    def ctx_now_estimate(self) -> Optional[float]:
        """估算浏览器**此刻**的 AudioContext 时刻。

        用最近一个锚点加上之后的墙钟增量。精度受墙钟与音频钟的相对漂移
        限制（几十 ppm，秒级只有毫秒量级），比 `now()` 那种"已 ingest 的
        采样数"准得多 —— 后者系统性落后真实播放位置 100~300ms。

        用途：打断时估计"已经播到哪了"。**必须配合一个余量使用**（见
        ``ActionExecutor._interrupt_current``）：估算值再准也不该拿去做
        精确截断，因为参考轨一旦清少了就补不回来。
        """
        if not self._anchors:
            return None
        last = self._anchors[-1]
        return last.ctx + (time.monotonic() - last.wall)

    def anchor_residual_ms(self) -> float:
        """当前映射的拟合残差（ms）；还没有可用映射时返回 -1。"""
        fit = self._ensure_fit()
        if fit is None:
            return -1.0
        return self._residual_ms(fit)

    def anchor_count(self) -> int:
        return len(self._anchors)


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
