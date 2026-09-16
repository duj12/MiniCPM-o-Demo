"""TTS 参考轨（AEC 的 farend 来源）。

回答一个问题：**此刻从扬声器出来的是什么，在麦克风时钟上？**

## 时间基准（两代踩坑史，务必读完再改）

参考轨的**所有位置一律用会话采样时钟**（``SampleClock`` 的计数）。浏览器
的 ``AudioContext`` 时间是**另一个时钟域**，只能经由
``SampleClock.ctx_to_sample()`` 换算后使用，绝不直接当采样位置。

**第一代坑**：把 ``clock.seconds()``（会话时钟秒）当成浏览器的 ``ctx_time``
传给 ``place()``，两个时钟域相减得到约 -1.3e7 的写入位置，``read()`` 因区间
不相交而**恒返回 0**。结果：AEC 的 farend 一直是静音，回声完全不消。
→ 教训：跨时钟域必须显式换算，不能想当然。

**第二代坑（本次修复）**：为绕开跨域换算，改成由服务端**预测**播放起点 ——
「送音频时的 ``clock.now()`` + 播放提前量」。但 ``clock.now()`` 只由浏览器
送来的 100ms mic 块推进，**比真实时刻慢 0~100ms 且逐块抖动**（浏览器主线程
同时在跑 25fps 抓帧）；而浏览器是收到 ``tts.audio`` 后 ``ctx.currentTime +
提前量`` 起播。两边隔着一整个网络往返 + 浏览器处理耗时 + mic 在途积压，
且这个误差**逐句变化** —— 固定常量 D 吸收不了变化量，于是「第一句碰巧对上、
后面就散掉」。
→ 教训：**预测跨时钟域的时刻是不可行的**，要把时刻从浏览器那边问出来。

**现在的做法**：浏览器在 ``tts.start`` 之后**承诺**起播时刻
（``playback{phase:'armed', start_ctx}``，此时音频还没到，往返藏在 TTS 合成
的 508ms 首帧延迟里，零额外代价），服务端用 ``ctx_to_sample()`` 把它换算成
会话采样位置再 ``place()``。``ctx_to_sample`` 靠每个 mic 块自带的
``ctx_time`` 拟合，是一条**精确**映射（两个时钟域数的是同一路 16kHz 流）。

副作用（好的那种）：网络、浏览器主线程抖动、mic 在途积压**全部从 D 里剔除**，
D 退化成「扬声器→麦克风的物理延迟 + 设备音频 I/O 缓冲」—— 每台设备一个
**固定常量**，离线测一次即可（``tests/measure_delay.py``）。

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


def _subtract(lo: int, hi: int, spans: List[tuple]) -> List[tuple]:
    """从区间 ``[lo, hi)`` 中挖掉 ``spans`` 覆盖的部分，返回剩余子区间。

    用于"只清自己的区间、不碰别人的" —— 打断后的二次校正（``resize``）
    必须避免清掉期间落位的其它 response（新句的音频是真会播出来的，
    清掉就是"有回声、没 farend"）。
    """
    pieces = [(lo, hi)]
    for s0, s1 in spans:
        nxt: List[tuple] = []
        for a, b in pieces:
            if s1 <= a or s0 >= b:          # 不相交
                nxt.append((a, b))
                continue
            if a < s0:                      # 左边残留
                nxt.append((a, s0))
            if s1 < b:                      # 右边残留
                nxt.append((s1, b))
        pieces = nxt
        if not pieces:
            break
    return pieces


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

        # 声学路径延迟（采样数）—— **每台设备一个固定常量**，离线测一次
        # （`tests/measure_delay.py`，用 `ORCH_DUMP_AUDIO` 的 dump 算），
        # 经 ORCH_AEC_DEFAULT_DELAY_MS 或 DelayStore 注入。运行时不再自适应：
        # 实测该估计器在真机上收敛不了，且一旦被噪声假峰钉死就会**永久失效**。
        self.delay_samples = 0
        # 已落位的区间（按 response 分组），供 truncate 用
        self._placed: Dict[str, List[PlaybackChunk]] = {}
        # 落位跨度（不在 _placed 里也能查）—— 供 resize() 二次校正用
        self._span_start: Dict[str, int] = {}
        self._span_end: Dict[str, int] = {}

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

        ⚠️ 它必须来自 ``clock.ctx_to_sample(浏览器承诺的 start_ctx)``，
        而**不是** ``clock.now() + 提前量`` —— 后者的误差（网络往返 +
        浏览器主线程抖动 + mic 在途积压）逐句变化，常量 D 吸收不了，
        正是"回声第一句好、后面失效"的根因。只有当浏览器没上报锚点时
        （旧客户端 / context 被挂起）才退回预测，且要标 `anchor_source`。

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
        # 记住落位跨度 —— `resize()` 要在 `truncate()` 把区间从 _placed
        # 移除**之后**仍能定位并二次校正（打断 → 收到回执 → 按实测播出量
        # 重新对齐）。参考轨是 30s 环形缓冲，这段时间内数据不会过期。
        lo, hi = self._span_start.get(response_id), self._span_end.get(response_id)
        self._span_start[response_id] = at_sample if lo is None else min(lo, at_sample)
        end = at_sample + x16.shape[0]
        self._span_end[response_id] = end if hi is None else max(hi, end)
        return chunk

    def started_at(self, response_id: str) -> Optional[int]:
        """该 response **最早**一个落位块的起点（会话采样）。

        用于打断时确定"已经播到哪了"：`tts.end` 到达时浏览器往往才刚起播，
        此时从"当前时刻"截断会把还没播、但即将播的那段也清掉 —— 而浏览器
        其实会把它播出来（已 `node.start()` 排程的 buffer 无法取消）。
        取 min(当前时刻, 起播点) 才是正确的截断位置。

        落位时刻现在是**精确**的（来自浏览器的 armed 承诺），所以这个起点
        就是"确实开始出声"的位置；只有降级到预测路径时才带预测误差。
        """
        chunks = self._placed.get(response_id)
        if not chunks:
            return None
        return min(c.t0 for c in chunks)

    def resize(self, response_id: str, keep_samples: int) -> int:
        """把某 response 落位内容的**有效长度**截到 ``keep_samples``。

        与 ``truncate`` 的区别：不要求该 response 还在 ``_placed`` 里 ——
        用于打断一次之后，**再用前端实测的播出量校正**（见
        ``session.on_playback_receipt`` 的 `sample_offset` 分支）。

        动机（用户指出的关键点）：参考轨应当严格跟随**实际播出**。打断
        那一刻我们只有**预测**（`clock.now()`），真实播出量要等浏览器的
        回执；回执到达时参考轨已经被截过一次（区间也从 ``_placed`` 移除
        了），所以需要记下落位起点、按"起点 + 实测播出量"重新对齐。

        参考轨是 30s 环形缓冲、写入区间不会过期，所以这里可以安全地
        二次校正。返回被清零的采样数。
        """
        lo = self._span_start.get(response_id)
        hi = self._span_end.get(response_id)
        if lo is None or hi is None:
            return 0
        # ⚠️ 上界必须用**本 response 自己的** span_end，不能退回全局
        # `written_span()` —— 那会一路清到轨上最新写入的位置，把**期间
        # 落位的其它 response（新句）一起清掉**。实测踩过：新句 3.0s → 0。
        cut = lo + max(0, int(keep_samples))
        if hi <= cut:
            return 0
        # ⚠️ 逐 chunk 清，且**跳过被其它 response 占用的区间** ——
        # 打断后新句往往紧跟着落位，两者在时间轴上可能重叠（真机里新句
        # 通常排在旧句之后，但调度抖动/时钟推进慢时就会压上）。无差别清
        # `[cut, span_end)` 会把新句一起清掉（实测：新句 3.0s → 0）。
        cleared = 0
        spans = self._other_spans(response_id)
        for c in self._placed.get(response_id, []):
            for lo_c, hi_c in _subtract(c.t0, c.t1, spans):
                s0, s1 = max(lo_c, cut), hi_c
                if s1 <= s0:
                    continue
                self.buf.zero_range(s0, s1)
                cleared += s1 - s0
        # 该 response 已被 truncate 移除（_placed 里没有）时，退化到
        # "只清本 response 自己的记录区间"
        if not self._placed.get(response_id):
            for lo_c, hi_c in _subtract(lo, min(hi, self._span_end.get(response_id, hi)),
                                        spans):
                s0, s1 = max(lo_c, cut), hi_c
                if s1 <= s0:
                    continue
                self.buf.zero_range(s0, s1)
                cleared += s1 - s0
        self._span_end[response_id] = min(hi, cut)
        return cleared

    def _other_spans(self, exclude: str) -> List[tuple]:
        """其它 response 占用的区间（供避免误伤）。"""
        out = []
        for rid, chunks in self._placed.items():
            if rid == exclude:
                continue
            for c in chunks:
                out.append((c.t0, c.t1))
        return out

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
        self._span_start.clear()
        self._span_end.clear()
        self.delay_samples = 0
