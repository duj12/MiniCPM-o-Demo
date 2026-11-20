"""answers → 2.1-lite E2 —— **阈值、空值语义、档位折算都在这一处**。

输入是 Flide ``/v1/systemone`` 的 ``answers``：

    {"attn.gaze": {"type": "choice", "choice": "at_screen",
                   "probabilities": {...}, "confidence": 0.71},
     "act.type":  {"type": "choice", "choice": "nod", "probabilities": {...}},
     "act.count": {"type": "score", "score": 1.4, "probabilities": {...}}}

输出是 ``docs/Social_State_Schema_v2_1_Lite.md`` §4.3 的 E2 ``state`` 块。
**只填模型能给答案的槽位**：自由文本槽位（``FEATURE.appearance`` /
``ENV.objects`` / ``POSITION.relative_positions`` / ``description``）留
``None`` / ``[]``，由编排侧从 Qwen3-Omni 的文本描述里补 —— 这里是"模型能给
什么"的边界，不是"schema 长什么样"的边界。

## 三条不能改的语义（schema §5.1）

| 写法 | 含义 | 本模块的落点 |
|---|---|---|
| ``unknown`` | 枚举槽位证据不足 | 模型自己选 ``unknown``，或置信度低于阈值 |
| ``null`` | 本回合不可观察（**人不在画面**） | ``person_visible`` 判假时整组置空 |
| ``[]`` | 观察到了，但确实没有 | 人可见且没有任何动作过阈值 |

``ACTION.events=[]``（人可见、没动作）与 ``null``（看不到人）对策略完全
相反，所以 ``person_visible`` 是**第一问**，也是整份 E2 的总闸。

## 次数与时长：档位折算（schema §5.3 过渡期规则）

``score`` 返回的是**期望档位**（0..n-1 的浮点），不是概率。折算：

* ``act.count``      ``1 + round(expected)``，夹到 ``[1, 4]``；
* ``beh.duration``   查 ``DURATION_SECONDS``（档位代表值，单位秒、整数）。

字段名与单位全程不变 —— State Engine 上线后由引擎覆盖，下游无需区分来源。
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional


#: ``beh.duration`` 档位 → 代表秒数。取每档的**中位偏小**值而不是上限：
#: 引擎上线前这个数只是"本观测窗口内"的估计，报大了一旦和引擎数字对不上
#: 会显得像 bug；报小了最坏是保守。
DURATION_SECONDS: tuple = (1, 3, 8, 15)

#: 题集 id → E2 落点 ``(键, 子键)``（子键 ``None`` = 整键）。``act.type``
#: 落整键 ``ACTION``（``events[]`` 没有"半份"）。
#:
#: 与编排侧 ``orchestrator/omni/state_schema._SLOT_PATHS`` 逐条对应 ——
#: 两边各存一份是**故意**的（state head 不 import 编排代码），由
#: ``orchestrator/tests/test_state_merge.py`` 的契约段对齐。
_SLOT_PATHS: Dict[str, tuple] = {
    "act.type": ("ACTION", None),
    "attn.orientation": ("ATTENTION", "orientation"),
    "attn.gaze": ("ATTENTION", "gaze"),
    "emo.visual": ("EMOTION", None),
    "beh.posture": ("BEHAVIOR", "posture"),
    "beh.motion": ("BEHAVIOR", "motion"),
    "beh.activity": ("BEHAVIOR", "activity"),
    "beh.duration": ("BEHAVIOR", "duration_s"),
    "rel.person_count": ("RELATION", "person_count"),
    "rel.interaction": ("RELATION", "interaction"),
    "rel.target": ("RELATION", "target"),
    "env.scene": ("ENV", "scene"),
    "env.lighting": ("ENV", "lighting"),
    "pos.region": ("POSITION", "region"),
}


def _f(v: Any) -> Optional[float]:
    """宽松转 float —— 模型/引擎给 ``int`` / ``str`` / ``None`` 都要扛住。"""
    if v is None or isinstance(v, bool):
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _clip01(x: Optional[float]) -> Optional[float]:
    if x is None:
        return None
    return round(min(1.0, max(0.0, x)), 4)


def _enum(ans: Dict[str, Any], min_conf: float) -> Dict[str, Any]:
    """``choice`` → ``{"label": token|None, "confidence": p}``。

    置信度取**被选中那一项的原始概率**（``probabilities[label]``），
    而不是 Flide 的 ``confidence`` —— 后者是把 top prob 从均匀重标定到
    "全押"的刻度（``(p_max - 1/k)/(1 - 1/k)``），数值天然偏大，用它会
    系统性高估。原始概率与 ECE 校准是对齐的刻度。

    ``min_conf`` 以上才认这个 label，否则退 ``unknown`` —— 这是"模型给了
    低置信答案"与"没证据"之间的翻译，默认 0（即不拦），由部署调。
    """
    probs = ans.get("probabilities") if isinstance(ans, dict) else None
    label = ans.get("choice") if isinstance(ans, dict) else None
    p: Optional[float] = None
    if isinstance(probs, dict) and label in probs:
        p = _f(probs[label])
    if label is None and isinstance(probs, dict) and probs:
        # 老引擎只回 probabilities 不回 choice 时，取最大项
        label = max(probs, key=lambda k: _f(probs[k]) or 0.0)
        p = _f(probs[label])
    if label is None:
        return {"label": None, "confidence": None}
    label = str(label)
    if p is not None and min_conf > 0 and p < min_conf and label != "unknown":
        return {"label": "unknown", "confidence": _clip01(p)}
    return {"label": label, "confidence": _clip01(p)}


def _prob_of(ans: Any, label: str) -> Optional[float]:
    """取某个选项的原始概率（没有就返回 None）。"""
    if not isinstance(ans, dict):
        return None
    probs = ans.get("probabilities")
    if not isinstance(probs, dict):
        return None
    return _f(probs.get(label))


def _noul(ans: Dict[str, Any]) -> Optional[float]:
    if not isinstance(ans, dict):
        return None
    return _f(ans.get("noul"))


def _score(ans: Dict[str, Any]) -> Optional[float]:
    """``score`` 的期望档位（0..n-1）。也接受老引擎的 ``level`` 键。"""
    if not isinstance(ans, dict):
        return None
    for key in ("score", "level", "expected"):
        v = _f(ans.get(key))
        if v is not None:
            return v
    probs = ans.get("probabilities")
    if isinstance(probs, dict) and probs:
        # 只有概率向量时自己求期望：键是索引或档位名时按顺序取位置
        items = list(probs.items())
        try:
            total = sum((_f(p) or 0.0) for _, p in items) or 1.0
            return sum(i * (_f(p) or 0.0) for i, (_, p) in enumerate(items)) / total
        except Exception:  # noqa: BLE001
            return None
    return None


def _count_from_score(expected: Optional[float]) -> Optional[int]:
    if expected is None:
        return None
    return max(1, min(4, 1 + int(round(expected))))


def _duration_from_score(expected: Optional[float]) -> Optional[int]:
    if expected is None:
        return None
    idx = max(0, min(len(DURATION_SECONDS) - 1, int(round(expected))))
    return int(DURATION_SECONDS[idx])


def _collect_events(answers: Dict[str, Any], person_visible: bool,
                    action_p: float, max_events: int,
                    tier2: bool) -> Optional[List[Dict[str, Any]]]:
    """``act.type`` 一个多选一 → ``ACTION.events``。

    ``None`` = 看不到人；``[]`` = 人可见但没动作。**互斥**，别混。

    ``none`` 是"观察到了，确实没有"；选中项概率低于 ``action_p`` 时按
    "证据不足"处理，同样落 ``[]`` —— 宁可漏报一次瞬时动作，也不让噪声
    动作污染下游策略。``other`` 是 schema 的枚举值，原样出。

    ``max_events`` / ``tier2`` 保留在签名里只为兼容（动作合并成一题后
    一拍最多一条事件，也不再按长尾裁剪）。
    """
    if not person_visible:
        return None
    node = _enum(answers.get("act.type", {}), 0.0)
    label, p = node.get("label"), node.get("confidence")
    if label in (None, "none", "unknown") or (p is not None and p < action_p):
        return []
    count = _count_from_score(_score(answers.get("act.count")))
    return [{"type": str(label), "count": count, "confidence": _clip01(p)}]


def assemble(answers: Dict[str, Any], *, stage: str = "turn",
             tier2: bool = False, has_scene: bool = False,
             person_p: float = 0.5, action_p: float = 0.5,
             enum_min_conf: float = 0.0, max_events: int = 2,
             ) -> Dict[str, Any]:
    """answers → E2 ``state`` 块。

    ``has_scene`` 说明本轮题集里有没有 ``env.scene``（turn 轮没有）——
    没有时该槽位返回 ``None``（"本轮未观测"），**不是**沿用上一轮：
    E2 的缺键语义就是 ``null``（schema §5.2），沿用由编排侧缓存负责。
    """
    pv = _noul(answers.get("person_visible"))
    # ⚠️ 拿不到 person_visible 的概率（引擎异常/题被裁掉）时**按"人可见"处理**：
    #    宁可给出一份可能有误的完整状态，也不要因为一个门控故障把整份 E2
    #    退化成 null —— 后者的症状是"画面有人但系统当没人"。
    visible = True if pv is None else (pv >= person_p)

    if visible:
        attention = {
            "orientation": _enum(answers.get("attn.orientation", {}), enum_min_conf),
            "gaze": _enum(answers.get("attn.gaze", {}), enum_min_conf),
        }
        behavior = {
            "posture": _enum(answers.get("beh.posture", {}), enum_min_conf),
            "motion": _enum(answers.get("beh.motion", {}), enum_min_conf),
            "activity": _enum(answers.get("beh.activity", {}), enum_min_conf),
            "duration_s": _duration_from_score(_score(answers.get("beh.duration"))),
        }
        region = _enum(answers.get("pos.region", {}), enum_min_conf)
    else:
        # schema §5.1：``null`` = 本回合不可观察。**只有枚举槽位与数组槽位
        # 能置 null**，所以 EMOTION.visual 仍然走 ``unknown``（§3.3 明确
        # "人脸不可见时必须用 unknown"）。
        attention = {"orientation": {"label": None, "confidence": None},
                     "gaze": {"label": None, "confidence": None}}
        behavior = {
            "posture": {"label": None, "confidence": None},
            "motion": {"label": None, "confidence": None},
            "activity": {"label": None, "confidence": None},
            "duration_s": None,
        }
        region = {"label": None, "confidence": None}

    emo = _enum(answers.get("emo.visual", {}), enum_min_conf)
    if not visible:
        # §3.3：人脸不可见时必须是 ``unknown``，**不能沿用上一轮**，也不能
        # 因为这里是"看不到人"就写 ``null`` —— ``visual`` 是枚举槽位，
        # 而 §5.1 规定 null 表示"本回合不可观察"，情绪槽位用 unknown 表达
        # "证据不足"，语义更准（下游看到 null 会以为整条链路没跑）。
        emo = {"label": "unknown", "confidence": _clip01(
            _prob_of(answers.get("emo.visual"), "unknown"))}

    person_count = None
    if visible:
        pc = _enum(answers.get("rel.person_count", {}), 0.0).get("label")
        if pc is not None:
            person_count = 3 if pc == "3+" else int(pc) if pc.isdigit() else None

    scene = (_enum(answers.get("env.scene", {}), enum_min_conf)
             if has_scene else None)

    return {
        "ATTENTION": attention,
        "ACTION": {"events": _collect_events(
            answers, visible, action_p, max_events, tier2)},
        "EMOTION": {
            # ``label`` 的综合结论由编排侧的晚融合填（视觉 + ASR 情绪）；
            # state head 只给 ``visual``。这里先原样放一份，编排侧会覆盖。
            "label": dict(emo),
            "visual": emo,
            # ``None`` = 还没做融合（≠ 无冲突）。schema §3.3 的表格里
            # "无冲突"是 ``False``，"融合尚未发生"只能是 ``null``。
            "conflict": None,
        },
        "FEATURE": {"appearance": None, "accessories": []},
        "BEHAVIOR": behavior,
        "RELATION": {
            "person_count": person_count,
            "interaction": _enum(answers.get("rel.interaction", {}), enum_min_conf)
            if visible else {"label": None, "confidence": None},
            "target": _enum(answers.get("rel.target", {}), enum_min_conf)
            if visible else {"label": None, "confidence": None},
        },
        "ENV": {
            "scene": scene,
            "lighting": _enum(answers.get("env.lighting", {}), enum_min_conf),
            "objects": [],
        },
        "POSITION": {"region": region, "relative_positions": []},
    }


def project_slots(state: Dict[str, Any], asked) -> Dict[str, Any]:
    """把 E2 裁到**本轮实际问过的**槽位（``STATE_SLOTS`` 白名单）。

    ``asked`` 是题集 id 集合（``build_questions(...).keys()``）。白名单模式
    下题集本来就是裁过的，``assemble`` 却仍会给没问的槽位填一个
    ``{"label": None}``（那是"本轮未观测"的表示）—— 返回给编排侧的 E2 里
    留着这些空键没有信息量，还会让 ``conf`` 计数和 ``STATE_JSON`` 变长。

    ⚠️ 两种情况**不裁**：

    * ``asked`` 为空 —— 非白名单模式，老行为；
    * **人不在画面**（``ACTION.events is None``）—— 那是整份 ``null`` 读数，
      裁掉就等于把 §5.1 的 ``null`` / ``[]`` 门控丢了，而那正是 E2 最值钱
      的部分（编排侧的 ``filter_e2_slots`` 也是同一条规则）。
    """
    if not asked or not isinstance(state, dict) or not state:
        return state
    if (state.get("ACTION") or {}).get("events", "missing") is None:
        return state
    asked = frozenset(asked)
    out: Dict[str, Any] = {}
    for qid, (key, sub) in _SLOT_PATHS.items():
        if qid not in asked or key not in state:
            continue
        if sub is None:
            out[key] = state[key]
            continue
        node = state.get(key)
        if isinstance(node, dict) and sub in node:
            out.setdefault(key, {})[sub] = node[sub]
    return out


def flatten_confidence(state: Dict[str, Any]) -> Dict[str, float]:
    """抽出 ``{E2 路径: 概率}``，给日志/前端/Agent 的一眼可读版。

    自由文本与 ``person_count`` 这类无概率的槽位不出现 —— 一份"部分键"
    的扁平表比塞满 ``None`` 更好用。
    """
    out: Dict[str, float] = {}

    def walk(prefix: str, node: Any) -> None:
        if isinstance(node, dict):
            # 枚举槽位是 {"label","confidence"}；ACTION.events 的条目是
            # {"type","count","confidence"} —— 两种都要收。
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

    walk("", state)
    return out
