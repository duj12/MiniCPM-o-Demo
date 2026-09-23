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
    """单帧的人脸观测（由 provider 每帧同步返回）。

    ⚠️ ``state`` **不是每帧都有**：G1 的 state 心跳绑在 landmark 上
    （墙钟 ≈5Hz / 208ms），离散状态变化（track 变、把握档变）和**身份结果
    落地**时才提前刷新。所以判据是 ``state is not None``，**不要用
    ``state_seq`` 去重** —— 官方明确身份结果回来时 seq 可能不 +1，
    按 seq 去重会恰好丢掉带身份的那一帧（见 ``face/g1face_provider.py``）。
    """

    t: int                                  # 会话采样轴时刻
    valid: bool = False
    box: Optional[tuple] = None             # (left, top, right, bottom)
    score: float = 0.0
    speaking: bool = False
    lip_state: str = "SILENT"               # SILENT | SPEAKING | GAP
    #: 唤醒达标。判据是 ``state["dwell_ms"] >= 阈值``（不是库的 interacting ——
    #: 上游已把那个字段标注为「保留但不对外」）。
    interacting: bool = False
    #: ⚠️ **只作参考值**：g1face 从名字正则解析 ``person_N``，而线上人脸库里
    #: 存的是真名 → 解析全失败 → 所有人都落到同一个兜底值。UI 认人请用 uid。
    person_id: int = -1
    #: 当前 track 连续在场时长（ms）。**每帧实时值**，不随 5Hz state 快照
    #: 推迟 —— 唤醒判据与 `begin` 事件都该用它，不要从 `state` dict 里取
    #: （那在非刷新帧上是 None，会得到 0）。
    dwell_ms: int = 0
    #: 视觉跟踪 id（g1face 里是字符串，如 ``"1000003"``）；无人为 None。
    track_id: Optional[str] = None
    #: 身份状态机名：recognized / enrolled / unknown / register_failed /
    #: identify_error；None = 还没有识别结果。用来区分「低置信（unknown）」
    #: 与「压根没算出来（register_failed / identify_error）」。
    identity_state: Optional[str] = None
    is_repeat: bool = False                 # 库为凑 24fps 补槽
    # G1 每帧 state 快照（g1face 的 FrameStateBuilder 产物）。字段：
    # face_present_confidence / lip_speaking_confidence / track_id / dwell_ms /
    # bbox_area_ratio / identity_id / identity_confidence / display_name /
    # identity_state / slot / state_seq / frame_index
    state: Optional[dict] = None
    #: 刷新计数（**仅诊断**：身份结果回来时可能不 +1，别拿它判有无新信息）。
    state_seq: int = -1


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
    #: 状态机名（recognized/enrolled/unknown/register_failed/identify_error）。
    #: ``is_enrolled`` 只区分「在不在库」，这个字段区分**为什么没认出来** ——
    #: 低置信（unknown，有候选但没确认）与算不出来（register_failed）要分开处理。
    identity_state: Optional[str] = None


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
    # ⚠️ 这里早先写的是 ``observations[-1].person_id`` —— 把身份当成了跟踪 id
    #    （复制粘贴错误）。IC 不读这个字段，所以一直没暴露出来。
    last_tid = observations[-1].track_id
    try:
        tid = int(last_tid) if last_tid is not None else 0
    except (TypeError, ValueError):
        tid = 0
    return LipEvent(
        t0=t0, t1=t1, track_id=tid,
        speaking=speaking, lip_state=lip_state, confidence=conf,
    )
