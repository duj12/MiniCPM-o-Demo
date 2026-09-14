"""有状态的流式重采样。

**为什么不能用逐块独立的重采样**：TTS 输出 24kHz，参考轨需要 16kHz，
比值 3:2。若对每个 chunk 独立重采样，chunk 边界处会引入相位不连续，
导致参考信号恰好在这些点与实播音频失配 —— 而 AEC 正是靠**逐样本**的
参考匹配来消回声的，边界失配会让 ERLE 周期性塌陷。

本模块在调用之间保留**全局输出采样序号**，因此
``concat(resample(c) for c in chunks)`` 与 ``resample(concat(chunks))``
逐样本一致（首尾边缘除外）。
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np


def _gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return a


class StatefulResampler:
    """多相线性插值重采样，跨调用保持状态。

    只支持有理数比率（``in_rate``/``out_rate`` 化为最简分数）。
    对 24000→16000 即 3:2：每 3 个输入采样产 2 个输出采样。

    实现刻意保持简单（线性插值）：
      - 浏览器端播放走的是**另一条** 24k→ctx_rate 的滤波路径，
        我们的 ref 与实播之间本就存在不可避免的滤波差异；
      - 该差异构成 ERLE 的地板，不是要追的 bug。
      用高阶滤波器只会增加复杂度而不消除这个结构性差异。

    算法：维护**全局输出序号 k** 与已消费的输入采样数 ``_in_consumed``。
    第 k 个输出采样落在输入轴的位置 ``p = k * down / up``。为使
    ``concat`` 与整段一致，位置必须相对**输入流起点**计算，而不是
    相对当前块。这是上一版出错的原因。

    为处理跨块的插值（位置落在两块之间），保留上一块最后一个采样作为
    ``_prev``，把当前块的坐标系左移 1。
    """

    __slots__ = ("in_rate", "out_rate", "_up", "_down",
                 "_k", "_in_consumed", "_prev", "_started")

    def __init__(self, in_rate: int, out_rate: int) -> None:
        if in_rate <= 0 or out_rate <= 0:
            raise ValueError("采样率必须为正")
        g = _gcd(in_rate, out_rate)
        self.in_rate = in_rate
        self.out_rate = out_rate
        self._up = out_rate // g      # 每个输入采样的输出数
        self._down = in_rate // g     # 每 down 个输入采样输出 up 个
        self.reset()

    def reset(self) -> None:
        self._k = 0               # 下一个待产生的全局输出序号
        self._in_consumed = 0     # 已完全消费的输入采样数（进入过 process 的）
        self._prev: Optional[float] = None
        self._started = False

    def process(self, x: np.ndarray) -> np.ndarray:
        """把一块输入重采样。``x`` 为 1-D float32，返回 1-D float32。

        跨调用连续：把多次调用结果顺序拼接，等价于对拼接后的输入做一次
        整段重采样。
        """
        if x.ndim != 1:
            raise ValueError("StatefulResampler 只接受 1-D 输入")
        n = x.shape[0]
        if n == 0:
            return np.zeros(0, dtype=np.float32)
        x = x.astype(np.float32, copy=False)

        # 构造"可在全局坐标系寻址"的数组：索引 0 对应全局输入位置
        # self._in_consumed - (1 if 有历史 else 0)。
        has_prev = self._started
        if has_prev:
            padded = np.empty(n + 1, dtype=np.float32)
            padded[0] = self._prev  # type: ignore[assignment]
            padded[1:] = x
            origin = self._in_consumed - 1   # padded[0] 的全局输入索引
        else:
            padded = x
            origin = 0
        # padded 覆盖全局输入索引 [origin, origin + len(padded))。
        # 插值 k 需要 ``idx`` 与 ``idx+1`` **都在** padded 内，否则必须
        # 推迟到下一块（不能用钳位兜底 —— 那会在每个块边界产生可见误差）。
        # 已消费的全局输入索引上界（不含）：self._in_consumed + n
        avail_end = self._in_consumed + n   # 下一块起点，padded 中不可用

        out: List[float] = []
        k = self._k
        while True:
            num = k * self._down
            idx = num // self._up
            if idx + 1 >= avail_end:
                break                    # 需要下一块的数据，停在这里
            frac = (num % self._up) / self._up
            local = idx - origin
            a = padded[local]
            b = padded[local + 1]
            out.append(float(a) * (1.0 - frac) + float(b) * frac)
            k += 1

        self._k = k
        self._in_consumed += n
        self._prev = float(x[-1])
        self._started = True
        return np.asarray(out, dtype=np.float32)


class PassthroughResampler:
    """同采样率时的占位实现（避免无谓拷贝）。"""

    __slots__ = ()

    def reset(self) -> None:
        return

    def process(self, x: np.ndarray) -> np.ndarray:
        return x
