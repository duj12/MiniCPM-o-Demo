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
from typing import Any, Dict, List

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

#: 流式回复的 stream_id。目前一轮对话同时只有一路 TTS 流，用固定值即可；
#: 将来要多路并发（比如插话）时改成按 response_id 生成。
OMNI_STREAM_ID = "omni"


class PassthroughDownstream:
    """把语音/模型输出直接转成 TTS 播报。无状态、无 LLM、确定性。

    ``streaming`` 打开且 ``mode="omni"`` 时走**增量**路径：模型每吐一段文本
    就发一个 :class:`Speak`（同一个 ``stream_id``），执行器据此边合成边播，
    首声不必等整条回复产完。关掉则退回「整条回复一次 Speak」的整段合成。
    """

    def __init__(self, mode: str = "asr", max_text_chars: int = 0,
                 prefix: str = "", streaming: bool = False) -> None:
        self.mode = mode
        # 0 = 不截断。默认不截断：OmniLLM 的回复常有几百字，截断会让
        # TTS 只念前半句（实测踩过）。需要限制时显式传值。
        self.max_text_chars = max_text_chars
        self.prefix = prefix
        # 只有 omni 模式有真正的流式文本来源（response.output.delta）；
        # asr 模式的 AsrFinal 本来就是一整句，走整段合成更简单。
        self.streaming = bool(streaming) and mode == "omni"
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

        # ---- 流式：模型每吐一段就发一个增量 Speak ----
        if self.streaming and isinstance(ev, OmniDelta):
            if ev.delta_kind != "text" or not ev.text:
                return []
            self.speaks += 1
            return [Speak(text=ev.text, stream_id=OMNI_STREAM_ID,
                          is_final=False)]

        text = None
        if self.mode == "asr" and isinstance(ev, AsrFinal):
            text = ev.text
        elif self.mode == "omni" and isinstance(ev, OmniResponseDone):
            if self.streaming:
                # 流式下文本已经逐段发过了，这里**只发收尾信号**，
                # 不能再把全文重发一遍（会重复合成一遍）。
                return [Speak(text="", stream_id=OMNI_STREAM_ID, is_final=True)]
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

    async def on_session_end(self, reason: str) -> None:
        logger.info("[downstream] 会话结束（%s），累计 TTS %d 次，事件统计: %s",
                    reason, self.speaks, self.seen)
