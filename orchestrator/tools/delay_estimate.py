"""声学路径延迟估计（GCC-PHAT）—— **离线工具**。

估计扬声器→麦克风的物理延迟 D，用来给 ``RefTrack.delay_samples`` 定一个
固定常量（每台设备测一次，见 ``orchestrator/tests/measure_delay.py``）。

## 为什么从实时路径上撤下来

这个类原本住在 ``audio/ref_track.py``，由 ``session._maybe_update_delay``
在每次播放窗口内调用。实机（iPhone）证实它收敛不了，而且失败模式很糟：
回声弱时 GCC-PHAT 找不到真峰，偶尔噪声凑出一个峰值比刚好过门限的假峰就被
**立刻采纳**，D 从此被钉死在错值上、回声再也消不掉（文件内 458-469 行留有
那次事故的日志实录）。现在 D 是固定常量，实时路径不再需要它。

但**不要删掉它**：它是本仓库唯一有测试覆盖的 GCC-PHAT 实现，
``tests/test_ref_track.py`` / ``test_aec_live_fidelity.py`` / ``test_aec_effect.py``
/ ``test_aec_loop.py`` 都在用，离线测 D 的流程也靠它。

⚠️ 必须由我方处理延迟：AEC 服务把 nearend/farend 按**相同偏移**推入，
即假定两者已样本对齐。仓库里的 ``GCCPHATDelayEstimator`` 从未被流式路径
import，且它依赖的 ``xmov_aec/config/aec_config.py`` 在仓库里**不存在**
（死代码），故此处重新实现。
"""
from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np

from ..clock import SR

logger = logging.getLogger(__name__)

class AcousticDelayTracker:
    """估计扬声器→麦克风的声学路径延迟 D（GCC-PHAT）。

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
        # 最近一次 estimate() 的失败原因 / 置信度 —— 供调用方分级记日志。
        # ⚠️ 没有这两个字段时，"估不出来"是完全静默的：下面的形状校验
        # 曾经因为契约不符而**每次直接 return None**，整条自适应延迟路径
        # 因此从未生效，却没有任何日志能看出来（潜伏了很久）。
        self.last_reason: Optional[str] = None
        self.last_ratio: float = 0.0
        # 需要多少个**互相一致**的估计才允许改变 delay。单个估计可能是
        # 噪声峰（回声弱时尤其常见），不能直接采纳 —— 见 estimate() 尾部。
        # 3 是平衡：真值稳定时 3 个窗（约 3 秒）内就收敛，而孤立的噪声峰
        # 凑不出 3 个**互相接近**的值。
        self.MIN_CORROBORATION = 3
        # 这些估计的最大散度（采样）。真实延迟在几秒内基本不变，散度大
        # 说明还在追噪声。160 采样 = 10ms @16k，与 AEC 的容忍窗同量级
        # （实测 D 偏 6ms 抑制就从 12.6dB 掉到 2.5dB）。
        self.MAX_SPREAD_SAMPLES = 160

    def estimate(self, mic: np.ndarray, ref_earlier: np.ndarray,
                 offset: int = 0, min_ratio: float = 1.5) -> Optional[int]:
        """估计 mic 中回声的延迟 D_path（采样）。

        **信号约定**（很关键，错了会恒报 0）：

          · ``mic[j]`` 对应绝对时间 ``T0 + j``
          · ``ref_earlier[i]`` 对应绝对时间 ``T0 - offset + i``
            —— 参考窗必须**覆盖更早的时间**，因为 mic 里的回声来自
            ``D_path`` 之前的播放

        返回的 ``D_path`` 满足
        ``mic[j] ≈ ref_earlier[j + offset - D_path]``。
        ``offset == D_path`` 时两者恰好对齐。

        **两种窗口几何**（由 ``offset`` 选择，都可以，但含义不同）：

          · ``offset > 0``（**生产链路用这个**）：参考窗比 mic 多出
            ``offset`` 个采样，显式覆盖 mic 窗之前的时间。此时
            ``D_path ∈ [0, offset]``。互相关搜 ``k ∈ [-offset, 0]``。
            ⚠️ 早先这里写的是 ``mic.shape != ref_earlier.shape``（要求
            等长），而调用方按这个几何传的是 ``N`` 与 ``N + offset`` ——
            于是**每次调用都在形状校验处返回 None**，自适应延迟估计整条
            路径从未生效过。

          · ``offset == 0``：等长窗口（离线分析：手上只有一段录音）。
            此时依赖"回声落在窗口内部"的部分重叠，``D_path ∈ [0, search]``。
            互相关搜 ``k ∈ [0, search]``。
        """
        self.last_reason = None
        self.last_ratio = 0.0
        offset = int(offset)
        n_mic = int(mic.shape[0])

        if ref_earlier.shape[0] != n_mic + offset:
            self.last_reason = (
                f"形状不符：mic={n_mic} ref={ref_earlier.shape[0]}，"
                f"期望 ref=mic+offset={n_mic + offset}"
            )
            return None
        if n_mic < 2048:
            self.last_reason = f"mic 窗过短（{n_mic} < 2048）"
            return None
        if float(np.sqrt(np.mean(ref_earlier ** 2))) < 1e-4:
            self.last_reason = "参考窗近似静音 —— 不在播放窗口内"
            return None
        if float(np.sqrt(np.mean(mic ** 2))) < 1e-5:
            self.last_reason = "麦克风近似静音"
            return None

        # ⚠️ 线性互相关不能循环卷绕：需要 n_fft ≥ len(mic) + len(ref) - 1。
        # 早先按 2*len(mic) 算，在 ref 变长（= mic + offset）后会卷绕 ——
        # 真峰被折到错误的 lag 上，表现为测出的延迟离谱但置信度不低。
        n_fft = 1
        while n_fft < n_mic + ref_earlier.shape[0]:
            n_fft <<= 1

        X = np.fft.rfft(mic, n_fft)
        Y = np.fft.rfft(ref_earlier, n_fft)
        # GCC-PHAT：只保留相位，对幅度差异不敏感
        R = X * np.conj(Y)
        mag = np.abs(R)
        mag[mag < 1e-10] = 1e-10
        cc = np.fft.irfft(R / mag, n_fft)

        # cc[k] = Σ_j mic[j]·ref[j-k]，即峰值在 k 处表示
        # mic[j] ≈ ref[j-k]。代入上面的约定得 **D = k + offset**。
        if offset > 0:
            search = min(offset, self.max_delay, n_fft // 2 - 1)
            if search < 1:
                self.last_reason = f"搜索窗为空（offset={offset}, n_fft={n_fft}）"
                return None
            # D ∈ [0, offset] → k ∈ [-search, 0]：只搜负 lag（含 k=0，
            # 它对应端点 D == offset）。正 lag 是 mic 领先 ref，在这个
            # 几何下非物理，搜它只会引入假峰。
            cand = np.concatenate([
                np.abs(cc[n_fft - search:]),        # k = -search .. -1
                np.abs(cc[:1]),                     # k = 0
            ])
            d_max = offset
        else:
            search = min(self.max_delay, n_fft // 2 - 1)
            if search < 1:
                self.last_reason = f"搜索窗为空（n_fft={n_fft}）"
                return None
            cand = np.abs(cc[: search + 1])        # k = 0 .. search
            d_max = search

        peak_i = int(np.argmax(cand))
        k = peak_i - (search if offset > 0 else 0)

        tmp = cand.copy()
        lo = max(0, peak_i - 2)
        tmp[lo:peak_i + 3] = 0
        ratio = float(cand[peak_i] / tmp.max()) if tmp.max() > 0 else 0.0
        self.last_ratio = ratio
        # 置信度：峰值须显著高于次峰
        if tmp.max() > 0 and cand[peak_i] < min_ratio * tmp.max():
            self.last_reason = f"相关峰不明显（峰比 {ratio:.2f} < {min_ratio}）"
            return None

        d_path = k + offset
        if d_path < 0 or d_path > d_max:
            self.last_reason = f"估计值 {d_path} 超出物理范围 [0, {d_max}]"
            return None
        if d_path > self.max_delay:
            logger.warning(
                "声学延迟估计 %d 采样（%.0fms）超出 max_delay=%d —— "
                "可能是选错峰，仍会采纳但请留意",
                d_path, d_path / self.sr * 1000, self.max_delay,
            )
        # ⚠️ **单个估计不直接采纳** —— 必须先攒够一致性证据。
        #
        # 真机踩过的坑：回声弱（mic RMS ≈0.001）时 GCC-PHAT 找不到真峰，
        # 峰比长期在 1.0~1.3 徘徊；偶尔噪声凑出一个峰比刚好过 1.5 的假峰，
        # 就被**立刻采纳**。日志实录：
        #     「声学延迟自适应: 4000 → 1 采样（0ms），本次估计 1
        #        （累计 1 次样本）」   ← 前面已失败 76 次
        # 从此 D 被钉在 0ms，回声再也消不掉 —— 表现为"长回复好好地说着，
        # 突然就无法打断、开始识别自己说的话了"。
        #
        # 现在：用**滑动中位数**判断，且要求至少 `MIN_CORROBORATION` 个
        # 相互接近的估计才允许改变 `delay`。离群值进不了中位数。
        self._history.append(d_path)
        if len(self._history) > self._median_window:
            self._history.pop(0)
        self.estimates += 1

        if len(self._history) >= self.MIN_CORROBORATION:
            med = int(np.median(self._history))
            # 一致性检查：最近的估计要聚在 med 附近，否则是在追噪声
            recent = self._history[-self.MIN_CORROBORATION:]
            spread = max(recent) - min(recent)
            if spread <= self.MAX_SPREAD_SAMPLES:
                if self.delay != med:
                    logger.info(
                        "声学延迟收敛: %d → %d 采样（%.0fms），"
                        "依据最近 %d 次估计（散度 %d 采样）",
                        self.delay, med, med / self.sr * 1000,
                        len(recent), spread,
                    )
                self.delay = med
            else:
                self.last_reason = (
                    f"估计散度过大（{spread} 采样 > {self.MAX_SPREAD_SAMPLES}）"
                    f"—— 还在追噪声，暂不采纳"
                )
        return d_path

    def reset(self) -> None:
        self._history.clear()
        self.delay = 0
        self.estimates = 0
        self.last_reason = None
        self.last_ratio = 0.0
