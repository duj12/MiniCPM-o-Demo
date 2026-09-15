"""声学延迟的**主动校准**。

## 为什么需要主动校准

自适应估计只在 TTS 播放窗口内有素材，且第一句播报时用的还是不准的 D
（消得不好）。主动校准让用户在开始对话前花 ~2 秒测一次，之后所有播报
都是对齐的。

## 为什么宽容度要求这么高

实测（``tests/test_aec_live_fidelity.py``，真实 AEC 服务）：

    D 补偿 = 84ms（真值） → 回声抑制 12.6 dB
    D 补偿 = 90ms         →  2.5 dB
    D 补偿 = 250ms        →  0.3 dB（等于完全不工作）

**容忍窗只有约 ±5ms。** 所以校准必须给出毫秒级准确的值 —— 差 6ms 就
从 12.6dB 掉到 2.5dB；差 166ms（旧默认值 250 vs 真值 84）就完全失效。

## 校准信号为什么用啁啾而非语音

GCC-PHAT 靠互相关的峰值位置定延迟。语音是**准周期**的，自相关有多个
相近的峰，容易选错（实测踩过：820ms 这种明显选错峰的结果被存下来）；
宽带线性调频（chirp）的自相关是尖锐单峰，鲁棒得多。

## 一次点击拿到自洽性校验

发**两段同样的啁啾**（中间一段静音），服务端做两次**独立**互相关。
两个结果必须一致 —— 否则说明其中一次选错了峰，直接判不可信。
这比"让用户点两次然后自己比对"体验好得多，也能挡住单次的偶发错误。

## 播放起点必须由前端上报（anchor）

早期实现用一个配置常量 ``playback_delay_ms`` 当播放起点。它**错两次**：
① 校准路径根本没有那个提前量（浏览器收到就播）；
② 前端只在起播**之后**才开始回传麦克风，于是 mic 索引 0 ≈ 起播时刻，
   真实往返只剩几十毫秒的纯声学延迟 —— 再减掉 200ms 就恒为负。

## 关于 anchor 的两条约定（弄反了就是"越加越大"）

  ① ``anchor`` = **开始出声**的时刻，相对**麦克风流起点**（16k 采样）。
     前端在 ``src.start(t)`` 时算 ``(t − capture_start) × 16000``。
  ② 互相关给出的是「该段啁啾在 **mic 流**中的起点」。于是

         D = (mic 中该段起点) − (该段在信号里的位置 + anchor)

     即"比预期晚到多少" —— **纯声学延迟**。

  ⚠️ 不能拿"mic 起点"直接减 anchor：那样得到的是"声学延迟 + 该段在
  信号内的偏移"，第二段会被多加 0.7s（静音间隔），越测越大。

  **拿不到 anchor 就拒绝给结论**，不再用一个猜的常量兜底。
"""
from __future__ import annotations

import logging
import time
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

CAL_SR = 24000          # 与 TTS 输出一致（走同一条播放路径）
CAL_MIC_SR = 16000      # 麦克风回传率（与 session 通道的 audio 消息一致）

# 两段啁啾，中间静音 —— 用于自洽性校验
CAL_CHIRP_S = 0.7
CAL_GAP_S = 0.3
# 人声主要能量在 300~3400Hz，把啁啾放在 500~4000Hz 仍可听但更易分辨，
# 且低于扬声器/麦克风的截止频率
CAL_F_START = 500.0
CAL_F_END = 4000.0

# 两段测得值的最大允许差（毫秒）。超过就判本次校准不可信 ——
# 真实延迟是物理量，两次测量应当一致；不一致只能是选错峰了。
CAL_SELF_CONSISTENCY_MS = 15.0


def make_chirp(sr: int = CAL_SR, duration_s: float = CAL_CHIRP_S,
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


def bandpass(x: np.ndarray, sr: int = CAL_SR, lo: float = CAL_F_START,
             hi: float = CAL_F_END) -> np.ndarray:
    """把信号限制到啁啾所在频段（默认 500–4000Hz）。

    ## 为什么必须做

    啁啾的能量 **99.8% 落在 500–4000Hz**，而**手机麦克风的低频隆隆声
    通常占主导** —— 手持抖动、空调、风、桌面震动都在 <300Hz，幅度往往
    远高于回声。

    不做带通时，这些低频会把**原始 rms 撑大**，于是：
      · 信噪比看起来是负的（实测真机 -2.6 ~ -5.0dB）
      · 互相关的归一化被低频主导，真峰被淹（峰比 1.0~2.7）
    而回声其实可能好端端地在那儿 —— 只是被低频盖住了。

    带通之后信噪比与峰比都只反映**啁啾所在频段**，这才是判据该看的量。

    用 FFT 实现：校准是离线整段处理，不需要流式滤波器。
    """
    if x.size == 0:
        return x
    n = x.size
    n_fft = 1
    while n_fft < n:
        n_fft <<= 1
    X = np.fft.rfft(x, n_fft)
    f = np.fft.rfftfreq(n_fft, 1.0 / sr)
    # 过渡带做一点软化，避免砖墙滤波在时域产生振铃（振铃会污染互相关）
    trans = max(50.0, lo * 0.15)
    mask = np.clip((f - (lo - trans)) / trans, 0.0, 1.0) * \
        np.clip(((hi + trans) - f) / trans, 0.0, 1.0)
    y = np.fft.irfft(X * mask, n_fft)[:n]
    return y.astype(np.float32)


def make_calibration_signal(sr: int = CAL_SR
                            ) -> Tuple[np.ndarray, List[int]]:
    """两段啁啾 + 中间静音。

    返回 ``(signal_int16, seg_starts)``：``seg_starts[i]`` 是第 i 段啁啾
    在 signal 内的起始采样位置 —— 服务端据此知道"第 i 段应该在什么时候
    被听到"，从而对每段做独立的互相关。
    """
    chirp = make_chirp(sr)
    gap = np.zeros(int(sr * CAL_GAP_S), dtype=np.int16)
    sig = np.concatenate([chirp, gap, chirp])
    return sig, [0, len(chirp) + len(gap)]


class CalibrationSession:
    """一次校准的生命周期。

    用法::

        cal = CalibrationSession(sr=16000)
        sig24, starts = cal.start()        # 拿到要播放的信号（24kHz int16）
        ... 送浏览器播放 ...
        cal.set_play_anchor(n)             # 前端上报真实起播位置（16k 采样）
        cal.feed(mic_chunk_16k)            # 持续喂麦克风
        res = cal.finish()                 # 返回延迟（采样，16k 基准）
    """

    def __init__(self, sr: int = CAL_MIC_SR, search_ms: float = 600.0) -> None:
        self.sr = sr
        # ⚠️ 搜索范围是**残余声学延迟**，不是全程往返 —— 有 anchor 之后
        # 我们知道每段预期出现在哪里，只需在它前后找一个小的窗口。
        # 纯声学路径（扬声器→空气→麦克风）典型 <50ms，600ms 已经很宽裕。
        # 早先按 1500ms 搜，既让 ready() 白等一秒多，也把相关窗撑得过长
        # 以致超出 mic 数据末尾、第二段被静默丢掉。
        self.search = int(sr * search_ms / 1000.0)
        self.signal24, self.seg_starts_24 = make_calibration_signal()
        # 各段起点换算到 16k（与麦克风同域）
        self.seg_starts = [int(s * sr / CAL_SR) for s in self.seg_starts_24]
        # ⚠️ 每段**啁啾本身**的长度（不是整个信号的一半 —— 信号 = 啁啾 +
        # 静音 + 啁啾，后半段用 `len(signal)/2` 会越界，导致第 2 段永远
        # 取不到参考、静默地只剩 1 段结果，自洽性校验形同虚设）。
        self.seg_len = int(CAL_CHIRP_S * sr)
        self.signal_len = int(len(self.signal24) * sr / CAL_SR)
        # 播放起点（16k 采样，相对采集起点）。**由前端上报**。
        self.play_anchor: Optional[int] = None
        self._mic: list = []
        self._mic_len = 0
        self._started_at: Optional[float] = None
        self.done = False

    # ------------------------------------------------------------------ #

    def start(self) -> Tuple[np.ndarray, List[int]]:
        """开始校准，返回要播放的 24kHz int16 信号与各段起点。"""
        self._mic = []
        self._mic_len = 0
        self.play_anchor = None
        self.done = False
        self._started_at = time.monotonic()
        logger.info("校准开始：播放两段 %.2fs 啁啾（%.0f→%.0fHz，间隔 %.2fs）",
                    CAL_CHIRP_S, CAL_F_START, CAL_F_END, CAL_GAP_S)
        return self.signal24, self.seg_starts_24

    def set_play_anchor(self, n_samples: int) -> None:
        """前端上报：**开始出声**的时刻在麦克风流里的位置（16k 采样）。

        由前端在 ``src.start(t)`` 时算 ``(t - capture_start_ctx) × 16000``
        —— 同一 AudioContext 内取差，与设备实际采样率无关。
        """
        self.play_anchor = int(n_samples)
        logger.info("校准：起播锚点 = %d 采样（%.0fms，相对采集起点）",
                    self.play_anchor, self.play_anchor / self.sr * 1000.0)

    def feed(self, mic_16k: np.ndarray) -> None:
        """喂入麦克风采样（16kHz，AEC **之前**的原始信号才有回声）。"""
        if self.done:
            return
        x = np.asarray(mic_16k, dtype=np.float32).reshape(-1)
        self._mic.append(x)
        self._mic_len += x.size

    # ------------------------------------------------------------------ #

    def expected_total_s(self) -> float:
        """预计需要采集多久（起播锚点 + 播放时长 + 最大搜索延迟 + 余量）。"""
        anchor_s = (self.play_anchor / self.sr) if self.play_anchor else 0.0
        return anchor_s + len(self.signal24) / CAL_SR + self.search / self.sr + 0.5

    def ready(self) -> bool:
        """采集是否足够。

        ⚠️ 必须**等锚点上报之后再判断** —— 否则会在播放还没开始时就
        "攒够"数据、提前结算。拿到锚点前一律返回 False。
        """
        if self.play_anchor is None:
            return False
        need = (self.play_anchor
                + int(len(self.signal24) / CAL_SR * self.sr)
                + self.search)
        return self._mic_len >= need

    def min_needed_s(self) -> float:
        """最低需要的采集时长（秒）—— 供日志与前端对齐。"""
        anchor_s = (self.play_anchor / self.sr) if self.play_anchor else 0.0
        return anchor_s + len(self.signal24) / CAL_SR + self.search / self.sr

    # ------------------------------------------------------------------ #

    @staticmethod
    def _correlate(mic: np.ndarray, ref: np.ndarray,
                   search: int) -> Optional[Tuple[int, float]]:
        """在 ``mic`` 里找 ``ref``（允许延时 lag ∈ [0, search]）。

        返回 ``(lag, 峰比)``；``mic[j] ≈ ref[j - lag]``。找不到明显峰返回
        None。用 GCC-PHAT（只保留相位，对幅度差异不敏感）。
        """
        if mic.size < ref.size + search:
            return None
        if float(np.sqrt(np.mean(ref ** 2))) < 1e-4:
            return None
        if float(np.sqrt(np.mean(mic ** 2))) < 1e-5:
            return None

        n_fft = 1
        while n_fft < mic.size + ref.size:
            n_fft <<= 1
        X = np.fft.rfft(mic, n_fft)
        Y = np.fft.rfft(ref, n_fft)
        R = X * np.conj(Y)
        mag = np.abs(R)
        mag[mag < 1e-10] = 1e-10
        cc = np.fft.irfft(R / mag, n_fft)
        # cc[k] = Σ mic[j]·ref[j-k] → 峰值位置即 lag
        seg = np.abs(cc[: search + 1])
        if seg.size == 0:
            return None
        peak = int(np.argmax(seg))
        tmp = seg.copy()
        # 排除主峰附近 ±8 采样（窄带信号的主瓣宽度）
        tmp[max(0, peak - 8):peak + 9] = 0
        ratio = float(seg[peak] / tmp.max()) if tmp.max() > 0 else 0.0
        return peak, ratio

    def finish(self) -> Optional[dict]:
        """计算延迟。返回 ``{delay_ms, delay_samples, ...}`` 或 None。

        ⚠️ 有两个硬前提，缺一不可 —— 都不再"用常量兜底"：
          · **必须收到 play_anchor**（前端上报的真实起播位置）
          · **必须收到 16kHz 的麦克风数据**（协议带 sample_rate 校验）
        """
        if self.done:
            return None
        self.done = True
        if not self._mic:
            logger.warning("校准失败：没收到麦克风数据")
            return None
        if self.play_anchor is None:
            logger.warning(
                "校准失败：前端未上报起播锚点（calibrate.anchor）—— "
                "无法确定播放起点，拒绝用配置常量兜底（那正是早先恒为负"
                "的原因）")
            return {"ok": False, "reason": "前端未上报起播锚点 —— "
                    "请刷新页面后重试（旧版页面不支持）"}

        mic = np.concatenate(self._mic)
        # ⚠️ **互相关用原始信号，不要带通**。
        # 曾试着把 mic 与 ref 都带通后再相关，结果**打坏了峰**：对 0.7s 的
        # 短段做零相位 FFT 带通，边缘伪影会压过真峰（实测峰比 7.88→1.10，
        # 且选到错误的 lag）。
        # 其实不需要 —— GCC-PHAT 只保留相位，对低频隆隆声天然不敏感。
        # 实测在强低频干扰下（本底 0.0033 vs 回声 0.4），未带通的互相关
        # 依然给出精确 lag、峰比 19.5。
        # 带通**只用于能量/信噪比诊断**（见下面的 sig_rms / noise_rms）。
        mic_bp = bandpass(mic, self.sr)
        anchor = self.play_anchor

        # 逐段独立互相关。
        # 窗口从**预期位置之前一点**开始扫，好让相关峰居中而不是贴边
        # （`_correlate` 的搜索范围是 [0, search]，峰贴 0 会切掉半宽）。
        results = []
        margin = int(0.05 * self.sr)          # 50ms 前置余量
        for i, start in enumerate(self.seg_starts):
            expect = anchor + start           # 该段预期在 mic 流中的位置
            # ⚠️ w0 会为负（anchor 很小时，如 lead=0 → expect=0）。
            # 负索引会悄悄取到数组尾部，必须钳到 0 —— 代价只是前置余量
            # 变小（下面的 pre 用实际值，不假设等于 margin）。
            w0 = max(0, expect - margin)
            pre = expect - w0                 # 实际可用的前置余量
            # 该段的参考（从播放信号里切出对应的一段；**不带通**，理由见上）
            ref = self._signal16[start:start + self.seg_len]
            if ref.size < self.seg_len:
                continue
            window = mic[w0: w0 + pre + self.seg_len + self.search]
            if window.size < pre + self.seg_len + 1024:
                continue
            # ⚠️ 搜索范围要**按实际拿到的数据裁剪**：末尾那段天然凑不满
            # 整个 search（mic 到这就结束了）。要求"整段 search 都在"会
            # 让最后一段被静默丢弃 —— 表现为只测到 1 段、自洽性校验空转。
            avail = min(pre + self.search, window.size - ref.size)
            if avail <= 0:
                continue
            got = self._correlate(window, ref, avail)
            if got is None:
                continue
            # 窗口内找的位置 → 换算回 mic 流，再减预期起点 = 纯声学延迟。
            # （pre ≥ 0：若延迟为负，mic_pos 会小于 expect，d 相应为负）
            mic_pos = w0 + got[0]
            d = mic_pos - expect
            results.append({"seg": i, "d": d, "ratio": got[1]})
            logger.info("校准：第 %d 段 mic 位置=%d（预期 %d）→ 延迟 %.1fms "
                        "峰比=%.2f", i, mic_pos, expect,
                        d / self.sr * 1000.0, got[1])

        if not results:
            logger.warning("校准失败：两段都没找到相关峰 —— 环境太吵、"
                           "音量太小，或扬声器声音没进麦克风")
            return {"ok": False, "reason": "找不到相关峰 —— 请确认用扬声器"
                    "外放（非耳机）、音量调大、环境安静后重试"}

        ds = [r["d"] for r in results]
        best = max(results, key=lambda r: r["ratio"])
        spread = (max(ds) - min(ds)) / self.sr * 1000.0

        mic_rms = float(np.sqrt(np.mean(mic ** 2)))

        # ---- 信号有效性：**带限信噪比** ----
        #
        # 判据演进（两次都被真机推翻，记下来免得退回）：
        #   ① 最初用绝对门槛 mic_rms ≥ 0.004 —— 设备相关，手机音量稍低
        #      就够不到；而且把起播前那段静音也平均了进去
        #   ② 改成"起播前本底 vs 啁啾段"的信噪比 —— 仍被**低频隆隆声**
        #      毁掉：手机手持噪声/空调/风都在 <300Hz，幅度常高于回声，
        #      把两侧 rms 一起撑大，实测真机信噪比 -2.6 ~ -5.0dB
        #   ③ 现在：**先带通到啁啾频段（500–4000Hz）再比**。该频段的
        #      本底接近 0，所以只要有回声响度就不是问题，且与设备无关。
        pre = mic_bp[:max(0, min(anchor, mic_bp.size))]
        noise_rms = (float(np.sqrt(np.mean(pre ** 2)))
                     if pre.size >= self.sr // 4 else 0.0)
        seg_rms = []
        for start in self.seg_starts:
            seg = mic_bp[anchor + start: anchor + start + self.seg_len]
            if seg.size >= self.seg_len // 2:
                seg_rms.append(float(np.sqrt(np.mean(seg ** 2))))
        sig_rms = float(np.median(seg_rms)) if seg_rms else 0.0

        # 带通后本底基本为 0，信噪比会非常大；这个门槛只用来挡
        # "完全没有回声"（sig_rms ≈ 0）的情况
        MIN_SNR_DB = 6.0
        # 绝对下限：带通后的信号若低于此值，基本就是滤波器数值噪声
        MIN_ABS_RMS = 1e-5
        snr_db = (20.0 * np.log10(sig_rms / noise_rms)
                  if noise_rms > 1e-9 and sig_rms > 0 else float("inf"))
        signal_ok = (sig_rms >= MIN_ABS_RMS
                     and (noise_rms <= 1e-9 or snr_db >= MIN_SNR_DB))

        # 自洽性：多段结果必须一致（不一致 = 至少一次选错峰）
        consistent = len(results) < 2 or spread <= CAL_SELF_CONSISTENCY_MS

        d_samples = int(np.median(ds))
        d_ms = d_samples / self.sr * 1000.0
        ok = (signal_ok and consistent and 0 <= d_ms <= 1500)

        reason = ""
        if not signal_ok:
            reason = (f"没收到可辨的回声（{CAL_F_START:.0f}–{CAL_F_END:.0f}Hz "
                      f"带内：啁啾段 rms={sig_rms:.5f}，本底 "
                      f"rms={noise_rms:.5f}，信噪比 {snr_db:.1f}dB；"
                      f"原始 rms={mic_rms:.5f}）—— 请确认：① 用扬声器外放"
                      "而非耳机 ② 音量调大 ③ 别挡住麦克风")
        elif not consistent:
            reason = (f"两段测量不一致（相差 {spread:.0f}ms）—— 至少一次"
                      "选错了峰，请保持安静后重试")
        elif not (0 <= d_ms <= 1500):
            reason = f"测得延迟 {d_ms:.0f}ms 不在合理范围，请重试"
        elif len(results) < 2:
            reason = "只测到一段的峰（另一段可能被噪声淹没），请重试"

        logger.info(
            "校准完成：锚点 %d，各段延迟=%s，中位 %.1fms（峰比 %.2f，"
            "段间差 %.1fms，带内 啁啾=%.5f 本底=%.5f SNR=%.1fdB，"
            "原始 rms=%.5f）%s",
            anchor, ds, d_ms, best["ratio"], spread, sig_rms, noise_rms,
            snr_db, mic_rms, "可信" if ok else "**不可信**",
        )
        return {
            "ok": ok,
            "delay_ms": d_ms,
            "delay_samples": d_samples,
            "peak_ratio": round(best["ratio"], 2),
            "segments": len(results),
            "spread_ms": round(spread, 1),
            "mic_rms": round(mic_rms, 6),
            "sig_rms": round(sig_rms, 6),
            "noise_rms": round(noise_rms, 6),
            "snr_db": round(snr_db, 1) if snr_db != float("inf") else None,
            "reason": reason,
        }

    # ------------------------------------------------------------------ #
    #  16k 参考信号（懒构造）
    # ------------------------------------------------------------------ #

    @property
    def _signal16(self) -> np.ndarray:
        if not hasattr(self, "_sig16"):
            x = self.signal24.astype(np.float32) / 32768.0
            n_out = int(len(x) * self.sr / CAL_SR)
            pos = np.linspace(0, len(x) - 1, n_out)
            i0 = np.floor(pos).astype(int)
            i1 = np.minimum(i0 + 1, len(x) - 1)
            fr = (pos - i0).astype(np.float32)
            self._sig16 = (x[i0] * (1 - fr) + x[i1] * fr).astype(np.float32)
        return self._sig16
