"""浏览器 ↔ Orchestrator 的 WS 协议。

**为什么不复用 MiniCPM 的 RealtimeSession 协议**：

  1. 它与其 gateway 生命周期不可分（``queue_done`` → ``session.init`` →
     ``session.created.active_model``），Orchestrator 没有这些，伪造会让
     前端永远带着死握手逻辑
  2. 它的 ``input.append`` 把音频和视频**捆在一条消息**里 —— 结构上无法
     表达「25fps 人脸 + 1fps OmniLLM + 100ms 音频」三条节奏
  3. 没有容纳播放回执的位置，而 AEC 参考时钟依赖它

**音频在线保持 float32**（与现状一致）：OmniLLM 的 ``b64()`` 反正要转
float32，AEC 服务也要求 float32，只有 ASR 要 int16 —— 服务端转一次优于
浏览器转了再转回来。带宽很小：100ms float32 16k 单声道 = 6.4KB →
base64 8.5KB × 10/s = **85KB/s**。

**用 JSON 不用二进制**：这个速率下可调试性更值钱。
"""
from __future__ import annotations

import base64
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

try:
    from typing import Literal
except ImportError:  # Python < 3.8（本地开发机）
    from typing_extensions import Literal  # type: ignore

SR = 16000
MIC_CHUNK = 1600          # 100ms
FACE_FRAME_INTERVAL_MS = 40    # 25fps
OMNI_FRAME_INTERVAL_MS = 1000  # 1fps


# ====================================================================== #
#  客户端 → 服务端
# ====================================================================== #

@dataclass
class ClientHello:
    """会话启动。"""
    type: Literal["session.start"] = "session.start"
    identity: Dict[str, Any] = field(default_factory=dict)
    seeded_delay_samples: int = 0     # 上次测得的声学延迟，用于冷启动
    system_prompt: str = ""


@dataclass
class AudioChunk:
    """麦克风音频。``audio_base64`` 是 float32 raw（小端）base64。

    ``ctx_time`` 是本块**首采样**在浏览器 ``AudioContext`` 上的时刻（秒），
    ``epoch`` 是该 AudioContext 的代号。两者一起让服务端能建立
    「浏览器时钟 → 会话采样」的精确映射（见 ``SampleClock.record_anchor``），
    参考轨据此落在**实际播出**的时刻上，而不是靠预测。
    """
    audio_base64: str
    t_ms: int = 0
    ctx_time: float = 0.0
    epoch: int = 0
    type: Literal["audio"] = "audio"

    @staticmethod
    def from_float32(x: np.ndarray, t_ms: int = 0,
                     ctx_time: float = 0.0, epoch: int = 0) -> "AudioChunk":
        buf = np.ascontiguousarray(x, dtype=np.float32).tobytes()
        return AudioChunk(audio_base64=base64.b64encode(buf).decode(),
                          t_ms=t_ms, ctx_time=ctx_time, epoch=epoch)


@dataclass
class VideoFace:
    """人脸用的视频帧（25fps，小图）。"""
    frame_base64: str
    t_ms: int = 0
    type: Literal["video_face"] = "video_face"


@dataclass
class VideoOmni:
    """OmniLLM 用的视频帧（1fps，大图）。"""
    frame_base64: str
    t_ms: int = 0
    type: Literal["video_omni"] = "video_omni"


@dataclass
class PlaybackReceiptMsg:
    """播放回执 —— 驱动 AEC 参考轨的时钟。

    四个 phase 的分工（``armed`` 是本次新增，也是最关键的一个）：

      · ``armed``     —— **承诺**。浏览器收到 ``tts.start`` 后立刻回一个
                        ``start_ctx``（它打算起播的时刻），此时音频还没到。
                        服务端据此**精确落位**参考轨，而不是靠
                        ``clock.now() + 提前量`` 去猜 —— 猜的误差含网络往返
                        与浏览器主线程抖动，**逐句变化**，固定常量 D 吸收
                        不了（这正是"第一句好、后面失效"的根因）。
      · ``started``   —— **校验**。实际排程时刻。服务端只比对偏差并告警，
                        **不据此修正参考轨**（见 ``on_playback_receipt``）。
      · ``ended``     —— 音频已全部交给播放器。**不代表播完**，绝不能截断。
      · ``cancelled`` —— 打断。带 ``sample_offset``（实测播出采样数）与
                        ``stop_ctx``，服务端据此把参考轨校到"真正播出过"。
    """
    response_id: str
    phase: Literal["armed", "started", "ended", "cancelled"]
    ctx_time: float = 0.0        # 兼容保留：服务端已不再读它
    seq: int = 0
    sample_offset: int = 0
    # 以下均为关键字参数带默认值 —— 位置传参的存量调用必须继续可用
    start_ctx: float = 0.0       # armed: 承诺起播时刻；started: 实际排程时刻
    stop_ctx: float = 0.0        # cancelled: 实际停下的 ctx 时刻
    epoch: int = 0
    type: Literal["playback"] = "playback"


@dataclass
class SessionStop:
    type: Literal["session.stop"] = "session.stop"


@dataclass
class CalibrateRequest:
    """请求一次声学延迟校准。

    服务端会用扬声器播一段**专用校准信号**（宽带啁啾，比语音更适合
    互相关），同时采集麦克风，用 GCC-PHAT 估计延迟。全程约 2 秒。
    """
    type: Literal["calibrate"] = "calibrate"


# ====================================================================== #
#  服务端 → 客户端
# ====================================================================== #

@dataclass
class SessionReady:
    session_id: str
    sample_rate: int = SR
    # 播放提前量的**唯一真源**。浏览器在 tts.start 之后 ``lead_ms`` 起播并
    # 把该时刻回执给服务端，两边必须用同一个数 —— 所以由服务端下发，
    # 前端不自己写死常量（写死过一次，两边漂了）。
    lead_ms: int = 200
    type: Literal["session.ready"] = "session.ready"


@dataclass
class TtsStart:
    response_id: str
    text: str = ""
    sample_rate: int = 24000
    lead_ms: int = 200          # 见 SessionReady.lead_ms
    type: Literal["tts.start"] = "tts.start"


@dataclass
class TtsAudio:
    """TTS 音频。``audio_base64`` 是 **int16 PCM 24kHz**（TTS 服务原生格式）。"""
    response_id: str
    seq: int
    audio_base64: str
    type: Literal["tts.audio"] = "tts.audio"

    @staticmethod
    def from_int16(pcm: np.ndarray, response_id: str, seq: int) -> "TtsAudio":
        buf = np.ascontiguousarray(pcm, dtype=np.int16).tobytes()
        return TtsAudio(response_id=response_id, seq=seq,
                        audio_base64=base64.b64encode(buf).decode())


@dataclass
class TtsDelta:
    """流式合成时的**文本增量**（仅供 UI 字幕）。

    TTS 是在 LLM 还在生成时就开跑的，所以 ``tts.start`` 出去时还没有文本。
    这条消息让前端**边生成边显示**，而不是等整轮说完才一次性冒出来
    （早先就是那样，看起来像"没有流式显示"）。

    ⚠️ 它**不驱动播放** —— 播放完全由 ``tts.audio`` 决定。字幕比音频先到
    是正常的（文本先产生，音频要等合成）。
    """
    response_id: str
    text: str = ""
    type: Literal["tts.delta"] = "tts.delta"


@dataclass
class TtsEnd:
    response_id: str
    #: 本轮完整文本。整段合成时字幕在这里一次性到；流式下文字已经由
    #: `tts.delta` 填过，这里可能为空（仅作收尾）。
    text: str = ""
    type: Literal["tts.end"] = "tts.end"


@dataclass
class TtsCancel:
    response_id: str
    reason: str = ""
    type: Literal["tts.cancel"] = "tts.cancel"


@dataclass
class AsrDisplay:
    """ASR 文本（仅 UI 显示，控制流走 downstream 接口）。

    ``state`` 是五个离散状态量的快照（用户是否在说 / 抢话把握 / 转写 /
    转写置信 / 本轮说完把握），见 ``asr/client.py`` 的 ``AsrState``。
    """
    phase: Literal["partial", "final"]
    text: str
    t_ms: int = 0
    state: Optional[dict] = None
    type: Literal["asr"] = "asr"


@dataclass
class FaceDisplay:
    """人脸状态（仅 UI 显示，不参与控制流）。

    ``tracks[0]`` 是主说话人：``{valid, box, score, speaking, lip,
    interacting, person_id}``。``identity`` 是最近一次识别结果
    （在库里才有 name/uid）。``wake`` 是最近一次唤醒事件。
    """
    tracks: List[Dict[str, Any]] = field(default_factory=list)
    identity: Optional[Dict[str, Any]] = None
    wake: Optional[Dict[str, Any]] = None
    #: G1 的每帧 state 快照（≈208ms 刷新一次，见 asr 侧的 state 是同一类东西）。
    #: 字段：face_present_confidence / lip_speaking_confidence / track_id /
    #: dwell_ms / bbox_area_ratio / identity_id / identity_confidence /
    #: display_name / slot / state_seq / frame_index
    state: Optional[Dict[str, Any]] = None
    type: Literal["face.state"] = "face.state"


@dataclass
class SessionStats:
    """服务端定期推送的链路状态（供 UI 显示与排障）。

    两组关键数字：

      · **声学延迟 D** —— 参考轨按它做预对齐。它是**每台设备一个固定常量**
        （离线测一次，见 ``tests/measure_delay.py``），不再运行时自适应。
      · **落位锚点** —— 参考轨的播出时刻是"浏览器回报"还是"服务端预测"。
        预测路径含网络与浏览器抖动、**逐句变化**，是回声消不掉的根因，
        所以这一项必须显眼。
    """
    aec_mode: str = "browser"          # browser | service | off
    delay_ms: float = 0.0              # 当前使用的声学延迟
    delay_source: str = "default"      # stored | default | manual | client
    delay_measured: bool = False       # 是否已实测到（而非用默认值）
    delay_samples: int = 0
    aec_active: bool = False           # 云端 AEC 是否在工作
    ref_nonzero_ratio: float = 0.0     # 送出的 farend 非零占比
    # 播放期间实测的回声抑制比（dB）。含近端故绝对值偏低，
    # 但**调 D 前后的相对变化**能直接判断配置对不对。
    erle_db: Optional[float] = None
    # 落位锚点来源：ack（浏览器承诺，准）| predicted（服务端猜，不准）| none
    anchor_source: str = "none"
    # 最近一次「实际起播 - 承诺起播」的偏差（ms）。>5ms 就说明承诺没兑现，
    # 值得查（但**不据此修正参考轨** —— 见 on_playback_receipt）。
    anchor_delta_ms: float = 0.0
    # ctx→会话采样 仿射拟合的残差（ms），-1 表示锚点还不够拟合。
    anchor_residual_ms: float = -1.0
    type: Literal["session.stats"] = "session.stats"


@dataclass
class ErrorMsg:
    code: str
    message: str
    type: Literal["error"] = "error"


# ====================================================================== #
#  编解码
# ====================================================================== #

def decode_audio_b64(s: str) -> np.ndarray:
    """把客户端发来的 float32 base64 解成 (1, T) float32。"""
    raw = base64.b64decode(s)
    x = np.frombuffer(raw, dtype=np.float32)
    return x.reshape(1, -1).copy()


def encode_audio_b64(x: np.ndarray) -> str:
    buf = np.ascontiguousarray(x, dtype=np.float32).tobytes()
    return base64.b64encode(buf).decode()


def to_json(msg: Any) -> str:
    """dataclass → JSON 字符串（跳过 None）。"""
    import json
    d = {k: v for k, v in msg.__dict__.items() if v is not None}
    return json.dumps(d, ensure_ascii=False)
