"""下游接口 —— Policy / Agent 的预留边界。

**本阶段不实现 Policy 与 Agent**，但接口必须现在定死，否则后续接入要动
Orchestrator。设计原则：

  - 下游是**纯决策**：不碰传输、不持 socket、不 sleep。它返回 action，
    由 Orchestrator 执行。
  - 事件按**到达顺序**投递，各自带采样时钟 ``t``。Orchestrator **不**为
    保证 ``t`` 单调而回压 —— 实时路径上那需要无界重排窗口。下游必须容忍
    迟到事件；需要时序推理时自己用 ``Tick`` 开重排窗口。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union

try:  # Python >= 3.8（生产环境 3.10）
    from typing import Literal, Protocol
except ImportError:  # Python < 3.8（本地开发机 3.7）
    from typing_extensions import Literal, Protocol  # type: ignore


# ====================================================================== #
#  入站事件（Orchestrator → 下游）
# ====================================================================== #

@dataclass(frozen=True)
class AsrPartial:
    """ASR 部分结果（2pass-online）。"""
    t: int
    text: str
    confidence: Optional[float] = None   # 实测 2pass-online 的 confidence 可能为 None
    segment_id: int = 0
    # 五个离散状态量（user_speaking / barge_in / transcript / asr_confidence /
    # turn_complete_confidence），由 AsrStateTracker 归纳，见 asr/client.py。
    state: Optional[Dict[str, str]] = None
    kind: Literal["asr.partial"] = "asr.partial"


@dataclass(frozen=True)
class AsrFinal:
    """ASR 最终结果（2pass-offline）。时间戳为**毫秒**，原点为流起点。"""
    t0: int                       # 采样轴上的起点（由 segment_start ms 换算）
    t1: int
    text: str
    confidence: Optional[float] = None
    tokens: List[str] = field(default_factory=list)
    token_times_ms: List[tuple] = field(default_factory=list)  # [[start_ms, end_ms, ch, prob], ...]
    speaker_id: Optional[str] = None
    is_final: bool = False        # True = 整条流结束
    # 同 AsrPartial.state —— 五个离散状态量的快照
    state: Optional[Dict[str, str]] = None
    kind: Literal["asr.final"] = "asr.final"


@dataclass(frozen=True)
class AsrTurnSense:
    """ASR 的语义完整性判决。

    ⚠️ 实测：**条件触发，不是每段都发**（真实语音可能一次都没有）。
    下游必须容忍缺失，不能依赖它。

    ``probabilities`` 是 **3 元素数组**（complete/incomplete/invalid），
    **不是 dict** —— 取自实测。
    """
    t: int
    label: Literal["complete", "incomplete", "invalid"]
    probabilities: List[float] = field(default_factory=list)
    segment_start_ms: int = 0
    segment_end_ms: int = 0
    speech_duration_s: float = 0.0
    kind: Literal["asr.turnsense"] = "asr.turnsense"


@dataclass(frozen=True)
class OmniTurnSense:
    """OmniLLM 的话轮判决。**这是权威边界**（见设计决策 4）。"""
    t: int
    label: str
    kind: Literal["omni.turnsense"] = "omni.turnsense"


@dataclass(frozen=True)
class OmniDelta:
    """OmniLLM 输出增量。

    ``delta_kind`` ∈ text|audio|listen —— 注意字段名不能叫 ``kind``，
    它已被用作事件类型判别符（下游按 ``ev.kind`` 分派）。
    """
    t: int
    delta_kind: Literal["text", "audio", "listen"]
    text: Optional[str] = None
    kind: Literal["omni.delta"] = "omni.delta"


@dataclass(frozen=True)
class OmniResponseDone:
    """OmniLLM 一轮回复完成（qwen3omni 语义：done = 完整回复结束）。"""
    t: int
    response_id: str
    text: str
    kind: Literal["omni.done"] = "omni.done"


@dataclass(frozen=True)
class FaceWake:
    """人脸唤醒（G1 库的 ``interacting``）。"""
    t: int
    phase: Literal["begin", "end"]
    track_id: int
    dwell_ms: int = 0
    mean_confidence: float = 0.0
    kind: Literal["face.wake"] = "face.wake"


@dataclass(frozen=True)
class FaceIdentity:
    """人脸身份识别结果。"""
    t: int
    track_id: int
    person_id: int = -1
    uid: Optional[str] = None
    name: Optional[str] = None
    similarity: float = 0.0
    is_enrolled: bool = False
    kind: Literal["face.identity"] = "face.identity"


@dataclass(frozen=True)
class FaceLipState:
    """唇动状态。以 **10Hz** 发出（每个 100ms 音频窗一条）。

    ⚠️ 唯一例外：``False→True`` 的说话跳变**立即**发出，不等聚合窗口 ——
    那是 barge-in 边沿，100ms 很关键。
    """
    t0: int
    t1: int
    track_id: int
    speaking: bool
    lip_state: str = "SILENT"       # SILENT | SPEAKING | GAP
    confidence: float = 0.0
    kind: Literal["face.lip"] = "face.lip"


@dataclass(frozen=True)
class PlaybackReceipt:
    """浏览器播放回执。驱动 AEC 参考轨的时钟。"""
    t: int
    response_id: str
    phase: Literal["started", "ended", "cancelled"]
    ctx_time: float = 0.0
    seq: int = 0
    kind: Literal["playback"] = "playback"


@dataclass(frozen=True)
class Tick:
    """心跳（默认 50ms）。下游在此实现超时逻辑。"""
    t: int
    kind: Literal["tick"] = "tick"


DownstreamEvent = Union[
    AsrPartial, AsrFinal, AsrTurnSense,
    OmniTurnSense, OmniDelta, OmniResponseDone,
    FaceWake, FaceIdentity, FaceLipState,
    PlaybackReceipt, Tick,
]


# ====================================================================== #
#  出站动作（下游 → Orchestrator）
# ====================================================================== #

@dataclass(frozen=True)
class Speak:
    """请求 TTS 合成并播放。

    **流式模式**：``stream_id`` 非空时，同 id 的多次 Speak 属于**同一轮回复
    的增量** —— 执行器把它们依次喂给同一个 TTS 流，边合成边播。
    ``is_final=True`` 表示该轮文本已发完（此时 ``text`` 通常为空）。

    非流式（默认 ``stream_id=""``、``is_final=True``）：整段合成 ``text``。
    """
    text: str
    tts_type: str = "mltts"
    speaker_id: Optional[str] = None
    speaker_vector_b64: Optional[str] = None
    priority: int = 0
    #: 非空 = 流式回复的标识；同一轮的所有增量共用它
    stream_id: str = ""
    #: 该轮文本是否已发完（流式收尾信号）
    is_final: bool = True
    kind: Literal["speak"] = "speak"


@dataclass(frozen=True)
class Cancel:
    """中断当前播报。

    执行时 Orchestrator **原子地**做三件事：
      ① 通知浏览器停止并清空播放器
      ② 从取消点 truncate() RefTrack（否则 AEC 拿着没播出的音频当参考
         会主动误适配，比不给参考更糟）
      ③ 通知前端停止 TTS 播放
    """
    reason: str
    response_id: Optional[str] = None
    kind: Literal["cancel"] = "cancel"


@dataclass(frozen=True)
class SendToOmni:
    """直接向 OmniLLM 注入输入（文本或音频）。"""
    audio_b64: str = ""
    text: str = ""
    force_listen: bool = False
    max_new_tokens: Optional[int] = None
    kind: Literal["send_omni"] = "send_omni"


@dataclass(frozen=True)
class Emit:
    """遥测 / UI 事件（不参与控制流）。"""
    channel: str
    payload: Dict[str, Any] = field(default_factory=dict)
    kind: Literal["emit"] = "emit"


DownstreamAction = Union[Speak, Cancel, SendToOmni, Emit]


# ====================================================================== #
#  下游协议
# ====================================================================== #

@dataclass
class SessionContext:
    """会话启动时传给下游的上下文。"""
    session_id: str
    sample_rate: int = 16000
    identity: Dict[str, Any] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)


class Downstream(Protocol):
    """下游决策接口。**纯决策，不碰传输。**

    实现者只需实现这四个方法；Orchestrator 负责执行返回的 action。
    """

    def describe(self) -> Dict[str, Any]:
        """能力声明（名称、支持的事件类型等），供日志与调试。"""
        ...

    async def on_session_start(self, ctx: SessionContext) -> List[DownstreamAction]:
        ...

    async def on_event(self, ev: DownstreamEvent) -> List[DownstreamAction]:
        ...

    async def on_session_end(self, reason: str) -> None:
        ...


class AgentPort(Protocol):
    """Agent 执行的抽象端口。

    照 ``qwen-audio-agent/server/src/backend/backend-port.mjs`` 的方法集
    建模：Policy 决定**何时**调 Agent，AgentPort 管**怎么**调。
    本阶段只定义，不实现（用 InMemoryAgent 做测试替身）。
    """

    def describe(self) -> Dict[str, Any]: ...
    async def start(self) -> None: ...
    async def health(self) -> Dict[str, Any]: ...
    async def submit(self, task: Dict[str, Any]) -> str: ...
    def subscribe(self, listener: Callable[[Dict[str, Any]], None]) -> Callable[[], None]: ...
    async def cancel(self, task_id: str) -> None: ...
    async def respond_input(self, task_id: str, request_id: str,
                            response: Dict[str, Any]) -> None: ...
    async def close(self) -> None: ...
