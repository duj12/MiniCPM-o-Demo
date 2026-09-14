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
    """麦克风音频。``audio_base64`` 是 float32 raw（小端）base64。"""
    audio_base64: str
    t_ms: int = 0
    type: Literal["audio"] = "audio"

    @staticmethod
    def from_float32(x: np.ndarray, t_ms: int = 0) -> "AudioChunk":
        buf = np.ascontiguousarray(x, dtype=np.float32).tobytes()
        return AudioChunk(audio_base64=base64.b64encode(buf).decode(), t_ms=t_ms)


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
    """播放回执 —— 驱动 AEC 参考轨的时钟。"""
    response_id: str
    phase: Literal["started", "ended", "cancelled"]
    ctx_time: float = 0.0
    seq: int = 0
    sample_offset: int = 0
    type: Literal["playback"] = "playback"


@dataclass
class SessionStop:
    type: Literal["session.stop"] = "session.stop"


# ====================================================================== #
#  服务端 → 客户端
# ====================================================================== #

@dataclass
class SessionReady:
    session_id: str
    sample_rate: int = SR
    type: Literal["session.ready"] = "session.ready"


@dataclass
class TtsStart:
    response_id: str
    text: str = ""
    sample_rate: int = 24000
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
class TtsEnd:
    response_id: str
    type: Literal["tts.end"] = "tts.end"


@dataclass
class TtsCancel:
    response_id: str
    reason: str = ""
    type: Literal["tts.cancel"] = "tts.cancel"


@dataclass
class AsrDisplay:
    """ASR 文本（仅 UI 显示，控制流走 downstream 接口）。"""
    phase: Literal["partial", "final"]
    text: str
    t_ms: int = 0
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
    type: Literal["face.state"] = "face.state"


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
