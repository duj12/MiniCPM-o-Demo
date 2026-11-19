"""E2 状态的合并、情绪晚融合、以及渲染成 IC/Agent 认的字符串。

E2 的形状见 ``orchestrator/docs/Social_State_Schema_v2_1_Lite.md`` §4.3；
本模块是**编排侧**的那一半 —— state head 只负责"每个槽位的 label + 概率"，
这里负责：

* ``merge_e2``        —— 两轮 E2 的合并（以及 ``null`` 的语义）
* ``fuse_emotion``    —— 视觉情绪 + ASR 情绪的晚融合（schema §3.3 规则表）
* ``to_vision_strings`` —— E2 → IC ``apply_vision`` 的 8 个字符串
* ``render_e2_json``  —— E2 → Agent 的 ``STATE_JSON`` 一行

## 三种空值（schema §5.1，改之前先读）

| 记号 | 含义 | 出现在哪 |
|---|---|---|
| ``null`` | 本回合**不可观察**（人不在画面） | 枚举槽位 / 数组槽位 |
| ``unknown`` | 枚举**证据不足** | 枚举槽位的 label |
| ``[]`` | 观察到了，**确实没有** | ``ACTION.events`` / ``ENV.objects`` |

所以 ``ACTION.events=null``（看不到人）与 ``[]``（人在、没动作）**语义相反**，
合并时不能一视同仁 —— 见 ``_person_missing``。
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

SCHEMA_VERSION = "2.1"
STATE_KEYS: Tuple[str, ...] = ("ATTENTION", "ACTION", "EMOTION", "FEATURE",
                               "BEHAVIOR", "RELATION", "ENV", "POSITION")

#: 情绪极性。``neutral`` / ``unknown`` 两侧都不算，``surprised`` 不计入冲突
#: （schema §3.3：惊讶是瞬时的，与"情绪反了"不是一回事）。
_POSITIVE = frozenset({"happy", "satisfied"})
_NEGATIVE = frozenset({"confused", "dissatisfied", "impatient", "angry", "sad"})
_NO_EVIDENCE = frozenset({"", "unknown", "none"})


# --------------------------------------------------------------------------- #
#  合并
# --------------------------------------------------------------------------- #

def _person_missing(e2: Dict[str, Any]) -> bool:
    """``null`` 三件套同时成立才算"人不在画面"。

    只判 ``ACTION.events is None`` 不够：模型偶发把 events 置空但其余槽位
    有值（那是 ``[]`` 的误写），这时按"人不见"整份清空会**丢一份好状态**。
    """
    if not isinstance(e2, dict):
        return False
    act = e2.get("ACTION") or {}
    attn = e2.get("ATTENTION") or {}
    orient = (attn.get("orientation") or {}).get("label") if isinstance(attn, dict) else None
    return act.get("events", "missing") is None and orient is None


def _merge_node(base: Any, patch: Any, skip_none: bool) -> Any:
    if isinstance(base, dict) and isinstance(patch, dict):
        out = dict(base)
        for k, v in patch.items():
            # ``None`` = 本轮未观测（``env.scene`` 在 turn 轮不问、
            # ``duration_s`` 的 score 没给）⇒ 保旧值。
            # ⚠️ "人不在画面"那种 null 走 ``_person_missing`` 分支，不到这里。
            if skip_none and v is None:
                continue
            out[k] = _merge_node(base.get(k), v, skip_none)
        return out
    if skip_none and patch is None:
        return base
    return patch


def merge_e2(base: Optional[Dict[str, Any]], patch: Dict[str, Any],
             stage: str = "turn") -> Dict[str, Any]:
    """把新一轮 E2 并进缓存。

    * ``stage="full"`` —— **整体替换**（起点全量是完整读数，旧键不该残留：
      换人后 ``FEATURE`` 必须换掉）；
    * ``stage="turn"`` —— 按键合并，**``None`` 一律保旧值**（本轮没问/没看见、
      不等于"现在没有了"）。唯一例外是"人不在画面"：那时 ``null`` 本身就是
      读数，整份替换（只保留 ``ENV.scene`` 这种静态槽位）。
    """
    patch = patch or {}
    if stage == "full" or not base:
        return dict(patch)
    if _person_missing(patch):
        merged = dict(patch)
        base_env = base.get("ENV") or {}
        if merged.get("ENV", {}).get("scene") is None and base_env.get("scene"):
            merged["ENV"] = dict(merged.get("ENV") or {})
            merged["ENV"]["scene"] = base_env["scene"]
        if not merged.get("ENV", {}).get("objects"):
            merged.setdefault("ENV", {})["objects"] = base_env.get("objects") or []
        return merged
    return _merge_node(base, patch, skip_none=True)


# --------------------------------------------------------------------------- #
#  情绪晚融合（schema §3.3）
# --------------------------------------------------------------------------- #

def _polarity(label: str) -> int:
    if label in _POSITIVE:
        return 1
    if label in _NEGATIVE:
        return -1
    return 0


def _conf(node: Any) -> Optional[float]:
    if isinstance(node, dict):
        c = node.get("confidence")
        if isinstance(c, (int, float)):
            return float(c)
    return None


def fuse_emotion(visual: Dict[str, Any],
                 asr_emotion: Optional[str],
                 asr_prob: Optional[float] = None,
                 asr_fresh: bool = True) -> Tuple[Dict[str, Any], Optional[bool]]:
    """视觉 + 语音 → (综合 ``label``, ``conflict``)。

    ``visual`` 是 E2 的 ``EMOTION.visual``（``{"label","confidence"}``）。
    返回的 ``conflict`` 三态：

    * ``False`` —— 有证据、不冲突；
    * ``True``  —— 视觉明确非中性、语音极性相反；
    * ``None``  —— **还没做融合**（没有语音侧证据）。schema 明确它**不是**
      ``False``：下游据此区分"验过了"与"没验"。
    """
    v_label = str((visual or {}).get("label") or "")
    v_conf = _conf(visual)
    a_label = str(asr_emotion or "")
    a_known = a_label not in _NO_EVIDENCE and asr_fresh

    if not a_known:
        # 语音侧没证据：结论就用视觉的，但不声称"验过一致性"。
        return (dict(visual or {}), None)

    v_known = v_label not in _NO_EVIDENCE
    if not v_known or v_label == "neutral":
        # 视觉中性/证据不足、语音明确 ⇒ 采用语音结论，且视觉值不被覆盖。
        label = a_label
        conf = float(asr_prob) if isinstance(asr_prob, (int, float)) else None
        return ({"label": label, "confidence": conf}, False)

    if v_label == a_label:
        return (dict(visual), False)

    # 一侧是 surprised 就不算"反了"（瞬时惊讶与情绪极性无关）。
    if "surprised" in (v_label, a_label):
        return (dict(visual), False)

    if _polarity(v_label) and _polarity(a_label) and _polarity(v_label) != _polarity(a_label):
        # 视觉明确非中性、语音反向 —— 采用**视觉**结论并标记冲突。
        # ASR 情绪永远不覆盖 visual（§3.3：两者独立、可交叉校验）。
        return (dict(visual), True)

    # 同极性不同类（如 satisfied vs happy）：不冲突，视觉优先。
    return (dict(visual), False)


# --------------------------------------------------------------------------- #
#  渲染
# --------------------------------------------------------------------------- #

def _label(node: Any) -> Optional[str]:
    if isinstance(node, dict):
        v = node.get("label")
        return str(v) if v else None
    return None


def _fmt(node: Any, with_conf: bool, null_mark: str = "null") -> str:
    """``{"label":x,"confidence":p}`` → ``x(0.82)``；``None`` → ``null``。"""
    if node is None:
        return null_mark
    if not isinstance(node, dict):
        return str(node)
    label = node.get("label")
    if label is None:
        return null_mark
    if with_conf:
        c = node.get("confidence")
        if isinstance(c, (int, float)):
            return f"{label}({c:.2f})"
    return str(label)


def to_vision_strings(e2: Dict[str, Any], with_conf: bool = True) -> Dict[str, str]:
    """E2 → 8 个字符串（IC ``apply_vision`` / Agent ``visual_background``）。

    **只返回有内容的键** —— 自由文本槽位（``FEATURE.appearance``）本来就不在
    E2 里，调用方用文本描述那份补。返回缺键而不是空串，调用方才能"保旧值"。
    """
    if not isinstance(e2, dict) or not e2:
        return {}
    missing = _person_missing(e2)
    out: Dict[str, str] = {}

    attn = e2.get("ATTENTION") or {}
    if attn:
        parts = []
        for k in ("orientation", "gaze"):
            node = attn.get(k)
            if node is not None:
                parts.append(f"{k}={_fmt(node, with_conf)}")
        if parts:
            out["ATTENTION"] = "; ".join(parts)

    act = e2.get("ACTION") or {}
    if "events" in act:
        events = act.get("events")
        if events is None:
            out["ACTION"] = "null（人物不在画面）"
        elif not events:
            out["ACTION"] = "[]（人物在场，未观察到动作）"
        else:
            out["ACTION"] = "; ".join(
                f"{e.get('type')}"
                + (f"×{e['count']}" if e.get("count") else "")
                + (f"({e['confidence']:.2f})"
                   if with_conf and isinstance(e.get("confidence"), (int, float))
                   else "")
                for e in events)

    emo = e2.get("EMOTION") or {}
    label_node = emo.get("label")
    if isinstance(label_node, dict) and label_node.get("label"):
        txt = f"label={_fmt(label_node, with_conf)}"
        visual = emo.get("visual")
        if isinstance(visual, dict) and visual.get("label") != label_node.get("label"):
            # 视觉与综合结论不一致 ⇒ 把原始视觉值也带上，不然冲突信息会丢
            txt += f"; visual={_fmt(visual, with_conf)}"
        if emo.get("conflict") is True:
            txt += "; conflict=true"
        out["EMOTION"] = txt

    feat = e2.get("FEATURE") or {}
    appearance = feat.get("appearance")
    accessories = feat.get("accessories") or []
    if appearance or accessories:
        parts = []
        if appearance:
            parts.append(str(appearance))
        if accessories:
            parts.append("配饰：" + "、".join(str(a) for a in accessories))
        out["FEATURE"] = "；".join(parts)

    beh = e2.get("BEHAVIOR") or {}
    if beh:
        parts = []
        for k in ("posture", "motion", "activity"):
            node = beh.get(k)
            if node is not None:
                parts.append(f"{k}={_fmt(node, with_conf)}")
        if beh.get("duration_s") is not None:
            parts.append(f"duration_s={beh['duration_s']}")
        if parts:
            out["BEHAVIOR"] = "; ".join(parts)

    rel = e2.get("RELATION") or {}
    if rel:
        parts = []
        if rel.get("person_count") is not None:
            parts.append(f"person_count={rel['person_count']}")
        for k in ("interaction", "target"):
            node = rel.get(k)
            if node is not None:
                parts.append(f"{k}={_fmt(node, with_conf)}")
        if parts:
            out["RELATION"] = "; ".join(parts)

    env = e2.get("ENV") or {}
    if env:
        parts = []
        for k in ("scene", "lighting"):
            node = env.get(k)
            if node is not None:
                parts.append(f"{k}={_fmt(node, with_conf)}")
        objects = env.get("objects") or []
        if objects:
            parts.append("objects=" + "、".join(str(o) for o in objects))
        if parts:
            out["ENV"] = "; ".join(parts)

    pos = e2.get("POSITION") or {}
    if pos:
        parts = []
        node = pos.get("region")
        if node is not None:
            parts.append(f"region={_fmt(node, with_conf)}")
        relpos = pos.get("relative_positions") or []
        if relpos:
            parts.append("relative_positions=" + "、".join(str(p) for p in relpos))
        if parts:
            out["POSITION"] = "; ".join(parts)

    if missing:
        # 人不在画面时，人物相关的槽位是 ``null``（本回合不可观察），
        # 不是 ``unknown``（证据不足）。给个中文尾巴，免得下游把它当成
        # "模型没答"而保留上一轮的旧值。
        for key in ("ATTENTION", "BEHAVIOR", "POSITION"):
            out.setdefault(key, "null（人物不在画面）")
    return out


def render_e2_json(e2: Dict[str, Any]) -> str:
    """E2 → 一行 ``STATE_JSON: {...}``（Agent 想要结构化数据时直接解析）。"""
    payload = {"schema_version": SCHEMA_VERSION, "state": e2}
    return "STATE_JSON: " + json.dumps(payload, ensure_ascii=False,
                                       separators=(",", ":"))


#: **E2 只回答一半**的两个键：``ENV`` 里 ``scene`` / ``lighting`` 是枚举、
#: ``objects`` 是自由文本；``POSITION`` 里 ``region`` 是枚举、
#: ``relative_positions`` 是自由文本。这两个键**不能整键覆盖** ——
#: 否则短文本模式（``ORCH_TEXT_SHORT=1``）下 state head 一答就把
#: 文本侧那份「桌上有杯子」「站在展台右侧」冲掉了。
_HALF_KEYS = frozenset({"ENV", "POSITION"})


def merge_vision_text(text_state: Dict[str, str],
                      e2: Dict[str, Any], with_conf: bool = True) -> Dict[str, str]:
    """文本描述的 8 键与 E2 的 8 键合成一份。

    E2 覆盖**它能回答的槽位**；``FEATURE`` 这类自由文本槽位 E2 不给（返回
    缺键），于是文本那份留着。这正是"Qwen3-Omni 只出短文本"的落地方式。

    ⚠️ ``ENV`` / ``POSITION`` 两个键是**两半拼起来的**（见 ``_HALF_KEYS``），
    所以这里做拼接而不是覆盖；拼接顺序是「文本在前、E2 在后」，免得
    E2 那半段把文本里的物品清单挤到看不见的位置。
    """
    merged = dict(text_state or {})
    for key, e2_part in to_vision_strings(e2, with_conf=with_conf).items():
        old = merged.get(key)
        if key in _HALF_KEYS and old and old != e2_part:
            merged[key] = f"{old}；{e2_part}"
        else:
            merged[key] = e2_part
    return merged


def flatten_confidence(e2: Dict[str, Any]) -> Dict[str, float]:
    """``{E2 路径: 概率}`` —— 给日志/前端的一眼可读版。"""
    out: Dict[str, float] = {}

    def walk(prefix: str, node: Any) -> None:
        if isinstance(node, dict):
            if ("label" in node or "type" in node) and "confidence" in node:
                c = node.get("confidence")
                if isinstance(c, (int, float)):
                    out[prefix] = float(c)
                return
            for k, v in node.items():
                walk(f"{prefix}.{k}" if prefix else str(k), v)
        elif isinstance(node, list):
            for i, item in enumerate(node):
                walk(f"{prefix}[{i}]", item)

    walk("", e2 or {})
    return out


__all__ = [
    "SCHEMA_VERSION", "STATE_KEYS", "merge_e2", "fuse_emotion",
    "to_vision_strings", "render_e2_json", "merge_vision_text",
    "flatten_confidence",
]
