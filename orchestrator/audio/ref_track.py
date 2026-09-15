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

    @staticmethod
    def _normalize_pcm(x: np.ndarray) -> np.ndarray:
        """把 TTS PCM 统一到 [-1, 1] float32。

        int16（TTS 服务的原生格式）除以 32768；float32 若幅度超过 1.5
        也按 int16 量纲处理（容错：有些调用方传 float 但没归一化）。
        已经是 [-1,1] 的 float 原样返回。
        """
        if x.dtype == np.int16:
            return x.astype(np.float32) / 32768.0
        x = x.astype(np.float32, copy=False)
        peak = float(np.abs(x).max()) if x.size else 0.0
        if peak > 1.5:
            # 明显是 int16 量纲的浮点（如 pcm.astype(np.float32)）
            return x / 32768.0
        return x

    def place(self, response_id: str, seq: int, pcm24: np.ndarray,
              at_sample: int) -> PlaybackChunk:
        """把一段 TTS PCM 落位到轨上。

        ``at_sample``：**会话采样时钟**上"开始出声"的位置。
        由调用方按 ``clock.now() + playback_delay_samples`` 计算，
        **不要**传浏览器的 ``ctx_time``。

        ⚠️ **量纲统一到 [-1, 1]**：TTS 服务返回的是 int16 PCM
        （实测幅度 ±18000，见 tests/README.md），而麦克风（nearend）是
        [-1,1] 的 float32。若把 int16 量纲原样存在轨上，送进 AEC 的
        farend 就会比 nearend **大约 32768 倍** —— 模型看到的参考大了 5 个
        数量级，回声估计会被压到 0，表现为**回声完全不消**。
        （报告里"farend 峰值 rms=4848"这条"正常"的记录，正是这个量纲
        问题的现场：4848 是 int16 量纲，而 mic 只有 0.01~0.1 量级。）
        """
        if pcm24.ndim != 1:
            raise ValueError("pcm24 必须是 1-D")
        pcm24 = self._normalize_pcm(pcm24)
        x16 = self._rs.process(pcm24)
        self.buf.write(at_sample, x16)
        chunk = PlaybackChunk(
            response_id=response_id, seq=seq,
            t0=at_sample, t1=at_sample + x16.shape[0], at_sample=at_sample,
        )
        self._placed.setdefault(response_id, []).append(chunk)
        return chunk

    def started_at(self, response_id: str) -> Optional[int]:
        """该 response **最早**一个落位块的起点（会话采样）。

        用于打断时确定"已经播到哪了"：`tts.end` 到达时浏览器往往才刚起播，
        此时从"当前时刻"截断会把还没播、但即将播的那段也清掉 —— 而浏览器
        其实会把它播出来（已 `node.start()` 排程的 buffer 无法取消）。
        取 min(当前时刻, 起播点) 才是正确的截断位置。

        ⚠️ 注意这里返回的是**预测**的起播点（落位时按约定的播放提前量算
        的），不是浏览器的 `ctx_time` 回执 —— 两者跨时钟域，不能混用。
        """
        chunks = self._placed.get(response_id)
        if not chunks:
            return None
        return min(c.t0 for c in chunks)

    def truncate(self, response_id: str,
                 from_sample: Optional[int] = None) -> int:
        """把某个 response **尚未播出**的参考清零。

        取消播放（barge-in / tts.cancel）时必须调用。否则 AEC 会拿到一段
        没有对应回声的参考信号，它会**主动误适配**去追这个不存在的回声 ——
        比不给参考更糟。

        ``from_sample`` 为取消的**截断点**（会话采样位置；None = 整个
        response）。返回被清零的采样数。

        ⚠️ **只清本 response 自己落位的区间**。`zero_range` 是无差别的，
        若直接清 ``[from_sample, 本response末尾)``，会把**期间插入的别的
        response**（如上一句还没播完、用户已插话、新一轮回复已落位）一起
        清掉 —— 那些音频是**真的会播出来**的，参考轨一旦被清就是"有回声
        但没有 farend"，AEC 全部失效。

        所以这里按 chunk 逐个求交集，只清属于本 response 的那部分。
        """
        chunks = self._placed.pop(response_id, [])
        if not chunks:
            return 0
        n_zeroed = 0
        for c in chunks:
            lo = c.t0 if from_sample is None else max(c.t0, from_sample)
            hi = c.t1
            if hi <= lo:
                continue
            self.buf.zero_range(lo, hi)
            n_zeroed += hi - lo
        return n_zeroed

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
        # 最近一次 estimate() 的失败原因 / 置信度 —— 供调用方分级记日志。
        # ⚠️ 没有这两个字段时，"估不出来"是完全静默的：下面的形状校验
        # 曾经因为契约不符而**每次直接 return None**，整条自适应延迟路径
        # 因此从未生效，却没有任何日志能看出来（潜伏了很久）。
        self.last_reason: Optional[str] = None
        self.last_ratio: float = 0.0

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
        self._history.append(d_path)
        if len(self._history) > self._median_window:
            self._history.pop(0)
        self.estimates += 1
        self.delay = int(np.median(self._history))
        return d_path

    def reset(self) -> None:
        self._history.clear()
        self.delay = 0
        self.estimates = 0
        self.last_reason = None
        self.last_ratio = 0.0
