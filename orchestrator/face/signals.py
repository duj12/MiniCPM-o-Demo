"""人脸信号的数据契约。

三个输出（对齐 G1 库的实测字段）：

  · **唤醒** —— 基于人脸框置信度 + 停留时间。G1 库的 ``out.interacting``
    在 C++ 内跨帧维护，我方直接消费。
  · **身份** —— 唤醒后累计 12 帧裁剪人脸，交给 FaceService 识别，
    结果回填 ``g1_face_set_person_id``。
  · **唇动** —— ``out.speaking`` / ``out.lip_state``（SILENT/SPEAKING/GAP）。

唇动信号以 **10Hz** 发出（每个 100ms 音频窗一条，聚合其中 ~2-3 帧），
与音频网格对齐，下游不需要更多。**唯一例外**：``False→True`` 的说话
跳变立即发出（barge-in 边沿，100ms 很关键）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

try:  # Python >= 3.8（生产环境 3.10）
    from typing import Literal
except ImportError:  # Python < 3.8（本地开发机 3.7）
    from typing_extensions import Literal  # type: ignore


@dataclass
class FaceObservation:
    """单帧的人脸观测（由 G1 库每帧同步返回）。"""

    t: int                                  # 会话采样轴时刻
    valid: bool = False
    box: Optional[tuple] = None             # (left, top, right, bottom)
    score: float = 0.0
    speaking: bool = False
    lip_state: str = "SILENT"               # SILENT | SPEAKING | GAP
    interacting: bool = False               # 唤醒达标
    person_id: int = -1                     # 未识别为 -1
    is_repeat: bool = False                 # 库为凑 24fps 补槽


@dataclass
class WakeEvent:
    """唤醒状态变化。"""

    t: int
    phase: Literal["begin", "end"]
    track_id: int = 0
    dwell_ms: int = 0
    mean_confidence: float = 0.0
    peak_confidence: float = 0.0
    box: Optional[tuple] = None


@dataclass
class IdentityEvent:
    """身份识别结果。"""

    t: int
    track_id: int = 0
    person_id: int = -1
    uid: Optional[str] = None
    name: Optional[str] = None
    similarity: float = 0.0
    is_enrolled: bool = False


@dataclass
class LipEvent:
    """唇动状态（按 100ms 音频窗聚合）。"""

    t0: int
    t1: int
    track_id: int = 0
    speaking: bool = False
    lip_state: str = "SILENT"
    confidence: float = 0.0
    mouth_open_ratio: float = 0.0


@dataclass
class FrameStats:
    """人脸线程的运行统计。"""

    frames_in: int = 0
    frames_dropped: int = 0
    frames_processed: int = 0
    wakes: int = 0
    identifies: int = 0
    last_error: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


def aggregate_lip(observations: List[FaceObservation], t0: int, t1: int
                  ) -> Optional[LipEvent]:
    """把一个音频窗内的多帧观测聚合成一条唇动事件。

    规则：窗口内**任一帧**报告说话即判为说话（避免漏掉 barge-in 边沿）；
    ``lip_state`` 取出现次数最多的那个。
    """
    if not observations:
        return None
    speaking = any(o.speaking for o in observations)
    states: Dict[str, int] = {}
    for o in observations:
        states[o.lip_state] = states.get(o.lip_state, 0) + 1
    lip_state = max(states.items(), key=lambda kv: kv[1])[0] if states else "SILENT"
    conf = sum(o.score for o in observations) / len(observations)
    return LipEvent(
        t0=t0, t1=t1,
        track_id=observations[-1].person_id if observations[-1].person_id >= 0 else 0,
        speaking=speaking, lip_state=lip_state, confidence=conf,
    )
