"""确定性下游桩 —— 让「说话 → ASR → TTS → 播放 → AEC」端到端可验证。

**本阶段不接 Policy / Agent。** 这个桩把 ASR 的最终文本直接作为 ``Speak``
吐出来，于是整条链路可以跑通并被验证；Policy 真实实现接入时，只需替换
这个实例，Orchestrator 一行不改。

两种模式：

  ``mode="asr"``  —— ASR 最终结果 → Speak（默认；验证 ASR→TTS 闭环）
  ``mode="omni"`` —— OmniLLM 的 response.done → Speak（验证 Omni→TTS 闭环）
  ``mode="echo"`` —— 只回显事件到日志，不触发 TTS（验证事件投递）
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from .interface import (
    AsrFinal,
    DownstreamAction,
    DownstreamEvent,
    Emit,
    OmniDelta,
    OmniResponseDone,
    SessionContext,
    Speak,
)

logger = logging.getLogger(__name__)

#: 句末标点 —— 见到就立刻把攒的文本发出去。分句由**服务端**做，这里攒批
#: 只是为了让每次送进去的文本有足够上下文（见下面 _pick_flush 的说明）。
_SENT_END = "。！？；!?;\n"

#: 攒批上限与兜底时长。
#:
#: ⚠️ **为什么必须攒批**（这是实测踩出来的）：服务端的 STREAM_SPLITTER 会对
#: **每次收到的文本**独立分句 —— 一次只喂 1~2 个字时，它就把每个字当成一句，
#: 给每个字配上句末语调再补静音。实测同一段 30 字文本：
#:
#:     每次喂 1 字  → 28 个音频帧，总长 17.66s   ← 听感就是"一两个字往外蹦"
#:     每次喂 8 字  →  4 个音频帧，总长  7.06s   ← 正常语速
#:     每次喂 15 字 →  2 个音频帧，总长  6.76s
#:
#: 所以攒到句末标点（或够长）再发，音质与「整段合成」一致；同时第一个
#: 短句仍然很早就发出去，首声延迟不受影响。
#:
#: **上限取 50**：实测 OmniLLM 大约 **265 字/秒**（每 ~49ms 来 ~13 字的 delta），
#: 攒满 50 字只要 ~190ms，远快于兜底超时 —— 所以正常情况下**是长度上限在切、
#: 超时根本轮不到**（超时实测改成 0.3/0.5/1.0/1.5s 结果完全一样，都是 49 字/段）。
_FLUSH_CHARS = 50

#: 兜底超时：只在 **LLM 卡住**时起作用（正常生成时长度上限先触发）。
#: 实测 delta 最大间隔 190ms，取 0.5s 留 2.6 倍余量 —— 能容忍短暂停顿不切碎句子；
#: 真卡住时又比 1s 更早把已有文本发出去（那个场景下**越短越好**）。
_FLUSH_SECONDS = 0.50


class PassthroughDownstream:
    """把语音/模型输出直接转成 TTS 播报。无状态、无 LLM、确定性。

    ``streaming`` 打开且 ``mode="omni"`` 时走**增量**路径：模型每吐一段文本
    就发一个 :class:`Speak`（同一个 ``stream_id``），执行器据此边合成边播，
    首声不必等整条回复产完。关掉则退回「整条回复一次 Speak」的整段合成。
    """

    def __init__(self, mode: str = "asr", max_text_chars: int = 0,
                 prefix: str = "", streaming: bool = False,
                 flush_chars: int = _FLUSH_CHARS,
                 flush_seconds: float = _FLUSH_SECONDS) -> None:
        self.mode = mode
        # 0 = 不截断。默认不截断：OmniLLM 的回复常有几百字，截断会让
        # TTS 只念前半句（实测踩过）。需要限制时显式传值。
        self.max_text_chars = max_text_chars
        self.prefix = prefix
        # 只有 omni 模式有真正的流式文本来源（response.output.delta）；
        # asr 模式的 AsrFinal 本来就是一整句，走整段合成更简单。
        self.streaming = bool(streaming) and mode == "omni"
        self.flush_chars = flush_chars
        self.flush_seconds = flush_seconds
        # ---- 流式攒批状态 ----
        # ⚠️ `_stream_id` **必须每轮不同**。早先是个常量 "omni"，于是第二轮
        # 的 delta 命中了还没收尾的第一轮流（执行器的 `_stream_id != act.stream_id`
        # 判等失败），文本被追加进上一轮的流 → 第二轮永远不开新流、永远不播。
        # 实测「两轮语音只出 1 次 tts.start」。
        self._turn = 0
        self._stream_id: Optional[str] = None
        self._buf: List[str] = []
        self._buf_started = 0.0
        self.speaks = 0
        self.seen: Dict[str, int] = {}

    def describe(self) -> Dict[str, Any]:
        caps = ["asr.final", "omni.done"]
        if self.streaming:
            caps.append("omni.delta")
        return {
            "name": "PassthroughDownstream",
            "mode": self.mode,
            "capabilities": caps,
        }

    async def on_session_start(self, ctx: SessionContext) -> List[DownstreamAction]:
        logger.info("[downstream] 会话开始 %s（mode=%s）", ctx.session_id, self.mode)
        return []

    async def on_event(self, ev: DownstreamEvent) -> List[DownstreamAction]:
        k = getattr(ev, "kind", "?")
        self.seen[k] = self.seen.get(k, 0) + 1

        # ---- 流式：攒够一句再发（不能每字一发，见 _FLUSH_CHARS 的说明）----
        if self.streaming and isinstance(ev, OmniDelta):
            if ev.delta_kind != "text" or not ev.text:
                return []
            return self._accumulate(ev.text)

        text = None
        if self.mode == "asr" and isinstance(ev, AsrFinal):
            text = ev.text
        elif self.mode == "omni" and isinstance(ev, OmniResponseDone):
            if self.streaming:
                # 流式下文本已经分句发过了，这里**只发收尾信号**，
                # 不能再把全文重发一遍（会重复合成一遍）。
                return self._finish_stream()
            text = ev.text

        if text:
            text = (self.prefix + text).strip()
            if self.max_text_chars > 0:
                text = text[: self.max_text_chars]
            if text:
                self.speaks += 1
                logger.info("[downstream] 触发 TTS #%d: %r", self.speaks, text)
                return [Speak(text=text)]

        if self.mode == "echo" and k in ("asr.final", "face.wake", "face.identity"):
            logger.info("[downstream] %s: %s", k, ev)

        return []

    # ------------------------------------------------------------------ #
    #  流式攒批
    # ------------------------------------------------------------------ #

    def _accumulate(self, delta: str) -> List[DownstreamAction]:
        """攒 delta，够一句（或够长/够久）才吐一个 Speak。

        **攒批是必须的**：服务端 STREAM_SPLITTER 对每次收到的文本独立分句，
        一次只喂 1~2 个字会被当成逐字成句 → 每个字都带句末语调，听感是
        「一两个字往外蹦」。见 ``_FLUSH_CHARS`` 的实测数据。
        """
        if self._stream_id is None:
            # 新的一轮：分配一个**新的** stream_id（旧的已结束）
            self._turn += 1
            self._stream_id = f"omni-{self._turn}"
        if not self._buf:
            self._buf_started = time.monotonic()
        elif (time.monotonic() - self._buf_started) >= self.flush_seconds:
            # ⚠️ 兜底超时要在**追加之前**判。否则本次 delta 会被并进积压里
            # 一起发（本地已有的文本迟迟不吐，反而被迟到的 delta 拖着）。
            # 正确语义：先把积压发出去，本次 delta 属于下一批。
            acts = self._flush()
            if delta:
                self._buf.append(delta)
                self._buf_started = time.monotonic()
            return acts
        self._buf.append(delta)
        if not self._should_flush():
            return []
        return self._flush()

    def _should_flush(self) -> bool:
        text = "".join(self._buf)
        if not text:
            return False
        # ① 句末标点 —— 首选：切在句子边界上，语调最自然
        if text[-1] in _SENT_END:
            return True
        # ② 够长了 —— 长句中途也要发，否则首声会一直等
        return len(text) >= self.flush_chars

    def _flush(self) -> List[DownstreamAction]:
        text = "".join(self._buf)
        self._buf = []
        self._buf_started = 0.0
        if not text:
            return []
        self.speaks += 1
        return [Speak(text=text, stream_id=self._stream_id or "",
                      is_final=False)]

    def _finish_stream(self) -> List[DownstreamAction]:
        """本轮文本发完：把残留补发，再发收尾信号，然后作废 stream_id。"""
        acts = self._flush()          # 尾巴上没到标点/不够长的部分
        sid = self._stream_id
        if sid is None:
            # 本轮一个字都没收到（比如全被过滤）—— 没有流要收
            return acts
        self._stream_id = None        # 下一轮 delta 会拿到新的 id
        acts.append(Speak(text="", stream_id=sid, is_final=True))
        return acts

    async def on_session_end(self, reason: str) -> None:
        logger.info("[downstream] 会话结束（%s），累计 TTS %d 次，事件统计: %s",
                    reason, self.speaks, self.seen)
