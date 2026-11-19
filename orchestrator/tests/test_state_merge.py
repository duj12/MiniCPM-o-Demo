#!/usr/bin/env python3
"""E2 状态的合并 / 融合 / 渲染 —— **纯离线**，不连 omni、不连 state head。

本脚本锁住 `orchestrator/omni/state_schema.py` 与
`interaction/downstream.py:_publish_vision` 的契约。它们是**改起来最容易
静默错**的一片：错了不会抛异常，只会把状态悄悄丢掉或悄悄膨胀。

锁住的六件事：

1. `merge_e2`：`full` 整体替换、`turn` 按键合并且 `None` 保旧值
   （`env.scene` 不在 turn 轮的问集里 —— 这是最容易踩的一条）；
2. **人不在画面**（`ACTION.events is None` 且 `attention.orientation is None`）
   时整份替换，但静态槽位 `ENV.scene` / `ENV.objects` 要留着；
3. `ACTION.events` 的 `null` 与 `[]` **语义相反**，渲染成中文必须可区分；
4. `fuse_emotion` 的规则表（一致 / 反向 / 视觉无证据 / 都无证据 / 惊讶）；
5. `merge_vision_text`：`ENV` / `POSITION` 是**两半拼起来**的，不能整键覆盖；
   文本侧那份 `_vlm_state` **不能被合成结果污染**（否则每发布一次就长一截）；
6. `StateClient` 拿不到服务时**返回 None 而不是抛异常**，并在连续失败后熔断
   （缺 httpx 时这一段跳过 —— 本地开发机常常没装，105/镜像里才有）；
7. **槽位白名单**（`ORCH_STATE_SLOTS`）：E2 只留白名单槽位、"人不在画面"
   的整份 null 不裁、半份键（ATTENTION/BEHAVIOR/RELATION）由文本拼接、
   E2 没产视觉情绪时**不冲掉**文本那份；外加与 `state_head` 的表一致性。

    python -m orchestrator.tests.test_state_merge
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from orchestrator.omni.state_schema import (  # noqa: E402
    DEFAULT_STATE_SLOTS, KEY_SLOTS, filter_e2_slots, merge_e2, fuse_emotion,
    to_vision_strings, merge_vision_text, parse_slots, partial_keys,
    render_e2_json, flatten_confidence,
)

_failures: list = []


def check(cond: bool, msg: str) -> None:
    print(f"  [{'OK' if cond else 'FAIL'}] {msg}")
    if not cond:
        _failures.append(msg)


def _node(label: str, conf: float = 0.9) -> dict:
    return {"label": label, "confidence": conf}


#: 一份典型的 full 轮 E2（形状取自 `state_head/assemble.py` 的输出）。
E2_FULL = {
    "ATTENTION": {"orientation": _node("facing_camera", 0.91),
                  "gaze": _node("at_screen", 0.77)},
    "ACTION": {"events": [{"type": "nod", "count": 2, "confidence": 0.82}]},
    "EMOTION": {"visual": _node("satisfied", 0.66)},
    "BEHAVIOR": {"posture": _node("standing", 0.9),
                 "motion": _node("stationary", 0.85),
                 "activity": _node("working", 0.4),
                 "duration_s": 6},
    "RELATION": {"person_count": 1, "interaction": _node("talking", 0.7),
                 "target": _node("camera_or_device", 0.8)},
    "ENV": {"scene": _node("office", 0.88), "lighting": _node("bright", 0.75)},
    "POSITION": {"region": _node("center", 0.72)},
}

#: 一份 turn 轮 E2：**没有** `env.scene`（turn 的问题集里就没有它）。
E2_TURN = {
    "ATTENTION": {"orientation": _node("back_to_camera", 0.6),
                  "gaze": _node("wandering", 0.55)},
    "ACTION": {"events": []},
    "ENV": {"scene": None, "lighting": _node("dim", 0.5)},
}


def t_merge_e2() -> None:
    print("\n[1] merge_e2：full 替换 / turn 保旧值")
    full = merge_e2({}, E2_FULL, "full")
    check(full["ENV"]["scene"]["label"] == "office", "full 轮写进 scene=office")

    turn = merge_e2(full, E2_TURN, "turn")
    check(turn["ENV"]["scene"]["label"] == "office",
          "turn 轮 patch 里 scene=None ⇒ **保旧值** office（不是被清成 None）")
    check(turn["ENV"]["lighting"]["label"] == "dim", "turn 轮真观测到的 lighting 覆盖")
    check(turn["ATTENTION"]["orientation"]["label"] == "back_to_camera",
          "turn 轮改掉了 orientation")
    check(turn["BEHAVIOR"]["posture"]["label"] == "standing",
          "turn 轮没问 BEHAVIOR ⇒ 旧值整块留着")
    check(turn["ACTION"]["events"] == [],
          "events 被显式改成 []（人在、没动作）⇒ 覆盖，不是保旧值")

    full2 = merge_e2(turn, {"EMOTION": {"visual": _node("angry", 0.7)}}, "full")
    check("BEHAVIOR" not in full2 or not full2.get("BEHAVIOR"),
          "full 轮是**整体替换**：旧键不残留（换人后 FEATURE 必须换掉）")


def t_person_missing() -> None:
    print("\n[2] 人不在画面：null 是读数，不是「没答」")
    full = merge_e2({}, E2_FULL, "full")
    gone = {
        "ATTENTION": {"orientation": {"label": None, "confidence": None},
                      "gaze": {"label": None, "confidence": None}},
        "ACTION": {"events": None},
        "EMOTION": {"visual": _node("unknown", 0.0)},
        "ENV": {"scene": None, "lighting": None},
    }
    out = merge_e2(full, gone, "turn")
    check(out["ACTION"]["events"] is None,
          "ACTION.events 是 **null**（不可观察），不是 []（人在、没动作）")
    check(out["ATTENTION"]["orientation"]["label"] is None,
          "ATTENTION.orientation 是 null")
    check(out["ENV"]["scene"]["label"] == "office",
          "静态槽位 ENV.scene 跨「人不见」保留")
    check(not out.get("BEHAVIOR"),
          "人物相关槽位（BEHAVIOR）随人物一起失效（整份替换，旧值不残留）")

    # 只把 events 置空、orientation 仍有值 ⇒ **不是**人不见（模型误写 [] ）
    lookalike = dict(gone)
    lookalike["ATTENTION"] = {"orientation": _node("facing_camera", 0.8)}
    keep = merge_e2(full, lookalike, "turn")
    check(keep["BEHAVIOR"]["posture"]["label"] == "standing",
          "events=null 但 orientation 有值 ⇒ 不按「人不见」清空（保住旧状态）")


def t_render() -> None:
    print("\n[3] E2 → 8 个字符串：null 与 [] 可区分")
    s = to_vision_strings(E2_FULL)
    check(s["ACTION"].startswith("nod×2(0.82)"), f"动作带次数与概率：{s['ACTION']}")
    check("orientation=facing_camera(0.91)" in s["ATTENTION"], "注意力带概率")

    empty = dict(E2_FULL, ACTION={"events": []})
    s2 = to_vision_strings(empty)
    check(s2["ACTION"].startswith("[]"), f"人在、无动作 ⇒ []：{s2['ACTION']}")

    missing = dict(E2_FULL, ACTION={"events": None},
                   ATTENTION={"orientation": {"label": None}})
    s3 = to_vision_strings(missing)
    check("null" in s3["ACTION"], f"人不见 ⇒ ACTION 渲染成 null：{s3['ACTION']}")
    check("ATTENTION" in s3 and "null" in s3["ATTENTION"],
          "人不见 ⇒ ATTENTION 也显式标 null（下游据此不保旧值）")

    # `with_conf=False` 只出标签
    s4 = to_vision_strings(E2_FULL, with_conf=False)
    check("(" not in s4["BEHAVIOR"], f"关掉置信度后不带括号：{s4['BEHAVIOR']}")

    js = render_e2_json(E2_FULL)
    check(js.startswith('STATE_JSON: {"schema_version":"2.1"'),
          "STATE_JSON 前缀可被 Agent 直接切")
    flat = flatten_confidence(E2_FULL)
    check(flat.get("ATTENTION.orientation") == 0.91, "拍平置信度按路径取值")
    check(flat.get("ACTION.events[0]") == 0.82, "事件 dict（type/count/confidence）也要拍平")


def t_fuse() -> None:
    print("\n[4] fuse_emotion 四条规则 + 惊讶 + 时效")
    visual = _node("satisfied", 0.8)
    label, conflict = fuse_emotion(visual, "satisfied", 0.9)
    check(label["label"] == "satisfied" and conflict is False,
          "两侧一致 ⇒ 同结论、conflict=False")

    label, conflict = fuse_emotion(visual, "angry", 0.7)
    check(label["label"] == "satisfied" and conflict is True,
          "视觉明确非中性、语音反向 ⇒ **视觉优先**且 conflict=True")

    label, conflict = fuse_emotion(_node("neutral", 0.5), "angry", 0.7)
    check(label["label"] == "angry" and conflict is False,
          "视觉中性、语音明确 ⇒ 采用语音")

    label, conflict = fuse_emotion(_node("unknown", 0.0), "happy", 0.6)
    check(label["label"] == "happy" and conflict is False, "视觉 unknown ⇒ 采用语音")

    label, conflict = fuse_emotion(visual, None, None)
    check(conflict is None, "语音侧没证据 ⇒ conflict=**None**（没做融合，不是 False）")

    label, conflict = fuse_emotion(visual, "angry", 0.7, asr_fresh=False)
    check(conflict is None, "ASR 情绪过期 ⇒ 当作没证据")

    label, conflict = fuse_emotion(visual, "surprised", 0.7)
    check(conflict is False and label["label"] == "satisfied",
          "一侧 surprised ⇒ 不计入冲突（瞬时惊讶与极性无关）")


def t_merge_text() -> None:
    print("\n[5] merge_vision_text：ENV/POSITION 两半拼接，文本侧不被污染")
    text = {"FEATURE": "青年女性，灰色短袖",
            "ENV": "桌上一只白杯子",
            "POSITION": "站在展台右侧"}
    e2 = {k: v for k, v in E2_FULL.items() if k in ("ENV", "POSITION")}
    text_copy = dict(text)
    merged = merge_vision_text(text, e2)
    check(merged["FEATURE"] == text["FEATURE"], "FEATURE 只有文本能给 ⇒ 留着")
    check(merged["ENV"].startswith("桌上一只白杯子") and "scene=office(0.88)" in merged["ENV"],
          f"ENV = 文本(物品) + E2(场景/光线)：{merged['ENV']}")
    check("region=center(0.72)" in merged["POSITION"]
          and merged["POSITION"].startswith("站在展台右侧"),
          f"POSITION = 文本(相对位置) + E2(九宫格)：{merged['POSITION']}")
    check(text == text_copy, "传入的 text_state 不被原地修改")

    again = merge_vision_text(text, e2)
    check(again == merged, "同一份输入合成两次结果相同（不会越拼越长）")


#: 下游 `_publish_vision` 的最小替身 —— 只装它真正读写的那几个字段。
def _mk_pub(state_enabled: bool = True, conf: bool = True,
            slots: str = ""):
    from orchestrator.interaction.downstream import InteractionDownstream

    d = InteractionDownstream.__new__(InteractionDownstream)
    d.session_id = "t-state-merge"
    d._state_enabled = state_enabled
    d._state_conf_in_text = conf
    d._state_slots = parse_slots(slots)
    d._state_concat = partial_keys(d._state_slots)
    d._asr_emotion = None
    d._asr_emotion_prob = None
    d._asr_emotion_at = 0.0
    d._vlm = ""
    d._vlm_state = {}
    d._vlm_text = ""
    d._state_e2 = {}
    d._state_at = 0.0
    d._state_stage = ""
    d._vlm_stage = ""
    # person_visible 两拍滞回的游标（`_on_state` 里的新字段）
    d._person_missing = None
    d._person_flip_pending = 0
    d.ic_calls = []
    d.rounds = []

    d._push_vision_to_ic = lambda state: d.ic_calls.append(dict(state))
    d._ic_owns_vision_round = lambda: False
    d._emit_visual_background = lambda n, have_state=False: d.rounds.append(
        (d._vlm, have_state))
    return d


def t_publish_vision() -> None:
    print("\n[6] _publish_vision：E2 缺席 = 老行为；在场时合成 & 不污染")
    # ---- E2 缺席：与 ORCH_STATE_HEAD=0 逐字节一致 ----
    d = _mk_pub(state_enabled=False)
    d._vlm_state = {"ATTENTION": "面向屏幕", "ACTION": "无"}
    d._vlm_text = "ATTENTION: 面向屏幕\nACTION: 无"
    d._publish_vision(parsed=True)
    check(d._vlm.startswith("ATTENTION: 面向屏幕"),
          f"没有 E2 ⇒ 只有 8 键渲染 + 原文：{d._vlm!r}")
    check("STATE_JSON" not in d._vlm, "没有 E2 ⇒ 不塞 STATE_JSON")
    check(d.ic_calls and d.ic_calls[-1].get("ATTENTION") == "面向屏幕",
          "IC 收到文本那份")

    # ---- E2 在场：STATE_JSON 打头，文本补自由文本槽位 ----
    d = _mk_pub()
    d._vlm_state = {"FEATURE": "青年女性，灰色短袖", "ENV": "桌上一只白杯子"}
    d._vlm_text = "FEATURE: 青年女性，灰色短袖\nENV: 桌上一只白杯子"
    d._state_e2 = E2_FULL
    d._state_at = time.monotonic()
    d._publish_vision()
    check(d._vlm.startswith("STATE_JSON: "), "E2 在场 ⇒ STATE_JSON 在最前面")
    check("FEATURE: 青年女性，灰色短袖" in d._vlm, "文本槽位进渲染")
    check("scene=office" in d._vlm, "E2 槽位进渲染")
    ic_state = d.ic_calls[-1]
    check("FEATURE" in ic_state and "ATTENTION" in ic_state,
          "IC 同时拿到自由文本与 E2 槽位")

    first = d._vlm
    d._publish_vision()
    check(d._vlm == first, "重复发布结果不变（`_vlm_state` 没被合成结果污染）")
    check(d._vlm_state.get("ENV") == "桌上一只白杯子",
          "文本侧 ENV 保持原样，不会累积 E2 那半段")

    # ---- E2 过期 ⇒ 退回文本（state 挂掉时 Agent 仍收得到画面）----
    d._state_at = time.monotonic() - 30.0
    d._publish_vision()
    check("STATE_JSON" not in d._vlm, "E2 超过 _STATE_FALLBACK_S ⇒ 退回纯文本")


def t_state_client() -> None:
    print("\n[7] StateClient：失败返回 None、连续失败熔断")
    try:
        import httpx  # noqa: F401
        from orchestrator.omni.state_client import StateClient
    except Exception as exc:  # noqa: BLE001
        print(f"  [SKIP] 本机没有 httpx（{exc}）—— 这一段在 105/镜像里跑")
        return

    import asyncio

    class _Boom:
        async def post(self, *a, **kw):
            raise RuntimeError("boom")

        async def aclose(self):
            pass

    c = StateClient("http://127.0.0.1:1", timeout_s=0.05,
                    fail_threshold=3, cooldown_s=60.0)
    c._client = _Boom()
    out = asyncio.run(c.post_state(["AAAA"], stage="turn"))
    check(out is None, "网络异常 ⇒ 返回 None（不抛）")
    check(c.stats["fail"] == 1, "记一次失败")
    check(asyncio.run(c.post_state(["AAAA"])) is None and c.stats["fail"] == 2,
          "第二次仍返回 None")
    check(asyncio.run(c.post_state(["AAAA"])) is None and c.tripped(),
          "达到阈值 ⇒ 熔断")
    before = dict(c.stats)
    check(asyncio.run(c.post_state(["AAAA"])) is None
          and c.stats["skipped"] == before["skipped"] + 1,
          "熔断期间直接跳过，不再发请求")
    check(asyncio.run(c.post_state([])) is None, "没有帧 ⇒ None")

    # 白名单**每次都发**：空列表也是显式的"问全套"，与服务端 env 默认值区分开
    class _Cap:
        def __init__(self) -> None:
            self.body = None

        async def post(self, url, json=None, **kw):
            self.body = json

            class _R:
                status_code = 200
                text = ""

                def json(self):
                    return {"state": {"ATTENTION": {"gaze": _node("at_screen")}},
                            "latency_ms": 5, "model": "onejev",
                            "n_questions": 1}

            return _R()

        async def aclose(self):
            pass

    for raw, want in (("attn.gaze, beh.activity", ["attn.gaze", "beh.activity"]),
                      ("", []), ("all", [])):
        cc = StateClient("http://127.0.0.1:1", slots=raw)
        cap = _Cap()
        cc._client = cap
        out = asyncio.run(cc.post_state(["AAAA"], stage="turn"))
        check(out is not None and cap.body.get("slots") == want,
              f"slots={raw!r} ⇒ 请求体 slots={cap.body.get('slots')}（期望 {want}）")


def t_slots() -> None:
    print("\n[8] 槽位白名单：裁剪 / 半份键拼接 / 情绪不被空值冲掉")
    slots = parse_slots(",".join(DEFAULT_STATE_SLOTS))
    check(set(slots) == set(DEFAULT_STATE_SLOTS), "默认白名单解析正确")
    _known = {i for ids in KEY_SLOTS.values() for i in ids}
    check(set(DEFAULT_STATE_SLOTS) <= _known,
          "默认白名单的每一项都属于某个 8 键（防拼写错误静默失效）")
    check(parse_slots("") == frozenset(), "空 ⇒ 空集合（E2 全出）")
    check(parse_slots("all") == frozenset() and parse_slots("*") == frozenset(),
          "all / * ⇒ 空集合（退老口径）")

    cut = filter_e2_slots(E2_FULL, slots)
    check(cut["ATTENTION"] == {"gaze": E2_FULL["ATTENTION"]["gaze"]},
          "ATTENTION 只剩 gaze（orientation 交回文本）")
    check("ACTION" not in cut, "ACTION 不在白名单 ⇒ 整键裁掉")
    check("EMOTION" not in cut, "EMOTION 不在白名单 ⇒ 整键裁掉")
    check(set(cut["BEHAVIOR"]) == {"activity"}, f"BEHAVIOR 只剩 activity：{cut['BEHAVIOR']}")
    check(set(cut["RELATION"]) == {"person_count", "interaction"},
          f"RELATION 只剩人数与互动：{cut['RELATION']}")
    check("ENV" not in cut and "POSITION" not in cut, "ENV/POSITION 也裁掉")

    half = partial_keys(slots)
    check(half == frozenset({"ATTENTION", "BEHAVIOR", "RELATION"}),
          f"半份键 = ATTENTION/BEHAVIOR/RELATION（ACTION 是整键）：{sorted(half)}")

    # 「人不在画面」是读数，**整份保留**（门控语义不能因为裁剪丢掉）
    gone = {"ACTION": {"events": None},
            "ATTENTION": {"orientation": {"label": None}, "gaze": {"label": "unknown"}},
            "BEHAVIOR": {"posture": {"label": "standing"}},
            "ENV": {"scene": {"label": "office"}}}
    check(filter_e2_slots(gone, slots) is gone,
          "人不见（events=null 且 orientation=null）⇒ 不裁，整份 null 读数保留")

    # 半份键：文本那半必须拼上，不能整键覆盖
    text = {"ATTENTION": "面向屏幕", "BEHAVIOR": "站立 6 秒",
            "RELATION": "单人，同伴"}
    merged = merge_vision_text(text, cut, concat_keys=half)
    check("gaze=at_screen" in merged["ATTENTION"]
          and merged["ATTENTION"].startswith("面向屏幕"),
          f"ATTENTION = 文本(朝向) + E2(视线)：{merged['ATTENTION']}")
    check("activity=working" in merged["BEHAVIOR"]
          and merged["BEHAVIOR"].startswith("站立 6 秒"),
          f"BEHAVIOR = 文本(姿态/时长) + E2(活动)：{merged['BEHAVIOR']}")
    # 不给 concat_keys 时是**整键覆盖**（老口径），确保两种口径真的不同
    overwritten = merge_vision_text(text, cut)
    check(not overwritten["ATTENTION"].startswith("面向屏幕"),
          "不传 concat_keys ⇒ 整键覆盖（对照组，证明上面那条真的靠 concat）")

    # 下游 `_on_state`：EMOTION 不在白名单时**不碰**文本那份情绪
    d = _mk_pub(slots=",".join(DEFAULT_STATE_SLOTS))
    d._vlm_state = {"EMOTION": "label=平静", "ATTENTION": "面向屏幕",
                    "BEHAVIOR": "站立 6 秒", "RELATION": "单人"}
    d._vlm_text = "EMOTION: label=平静"
    from orchestrator.downstream.interface import OmniState
    d._on_state(OmniState(t=0, stage="turn",
                          state=dict(E2_FULL, EMOTION={"visual": _node("neutral", 0.9)}),
                          latency_ms=1, model="onebit"))
    check("EMOTION" not in d._state_e2,
          "E2 没产视觉情绪（白名单不含 emo.visual）⇒ 不往缓存里塞空 EMOTION")
    check("label=平静" in d._vlm, f"文本那份情绪原样留着：{d._vlm.splitlines()}")
    check("gaze=at_screen" in d._vlm, "E2 的 gaze 进了快照")
    check(d._vlm_state.get("ATTENTION") == "面向屏幕",
          "文本侧 ATTENTION 未被污染（拼接结果不写回 _vlm_state）")

    # 契约：state_head 那份 `_SLOT_PATHS` 与编排侧逐条一致
    try:
        from state_head.assemble import _SLOT_PATHS as SH_PATHS
    except Exception as exc:  # noqa: BLE001
        print(f"  [SKIP] 本机 import 不到 state_head（{exc}）")
    else:
        from orchestrator.omni.state_schema import _SLOT_PATHS as ORCH_PATHS
        check(SH_PATHS == ORCH_PATHS,
              "state_head 与编排侧的槽位→E2 路径表逐条一致")


#: 人不在画面时的一份完整 full 读数（`state_head.assemble` 的形态）。
E2_GONE = {
    "ATTENTION": {"orientation": {"label": None, "confidence": None},
                  "gaze": {"label": None, "confidence": None}},
    "ACTION": {"events": None},
    "EMOTION": {"visual": {"label": "unknown", "confidence": 0.0}},
    "BEHAVIOR": {"posture": {"label": None, "confidence": None},
                 "motion": {"label": None, "confidence": None},
                 "activity": {"label": None, "confidence": None},
                 "duration_s": None},
    "RELATION": {"person_count": None,
                 "interaction": {"label": None, "confidence": None},
                 "target": {"label": None, "confidence": None}},
    "ENV": {"scene": {"label": "office", "confidence": 0.8},
            "lighting": {"label": "bright", "confidence": 0.7}},
    "POSITION": {"region": {"label": None, "confidence": None}},
}


def t_hysteresis() -> None:
    print("\n[9] person_visible 两拍滞回（1Hz 采样不抖）")
    from orchestrator.downstream.interface import OmniState

    def push(d, state: dict) -> None:
        d._on_state(OmniState(t=time.monotonic(), stage="full",
                              state=state, latency_ms=1, model="onebit"))

    d = _mk_pub()
    push(d, E2_FULL)
    check(d._person_missing is False, "首拍定基线：人在画面")
    baseline = d._state_e2
    n_pub = len(d.ic_calls)

    push(d, E2_GONE)
    check(d._state_e2 is baseline, "第 1 拍『人不见了』整份丢弃（缓存不动）")
    check(len(d.ic_calls) == n_pub, "待确认的那一拍不发布")
    check(d._person_flip_pending == 1, "翻转计数 1/2")

    push(d, E2_FULL)
    check(d._person_flip_pending == 0 and d._person_missing is False,
          "下一拍又见人 ⇒ 计数清零，不翻转")

    push(d, E2_GONE)
    check(d._person_flip_pending == 1, "重新计数 1/2")
    push(d, E2_GONE)
    check(d._person_missing is True, "连续 2 拍一致 ⇒ 翻转生效")
    check(d._state_e2.get("ACTION", {}).get("events") is None,
          "翻转后缓存的是 null 读数（看不到人）")

    # 反向同样要两拍：单拍"又看到人"不翻回
    push(d, E2_FULL)
    check(d._person_missing is True and d._person_flip_pending == 1,
          "从 null 翻回也要两拍（单拍不翻回）")


def t_desc_only_parse() -> None:
    print("\n[11] desc-only 三行解析（FEATURE / ENV / POSITION）")
    from orchestrator.omni.describe import parse_state, render_state

    three = ("FEATURE: 青年女性，灰色短袖，戴眼镜\n"
             "ENV: 白杯、绿植、木质展台\n"
             "POSITION: 站在展台右侧，偏画面中央")
    got = parse_state(three)
    check(set(got) == {"FEATURE", "ENV", "POSITION"} and len(got) == 3,
          f"三行 ⇒ 三个键：{sorted(got)}")
    check(got["FEATURE"].startswith("青年女性"), "FEATURE 值完整")

    wrapped = ("FEATURE: 青年女性，灰色短袖，戴眼镜\n"
               "ENV: 白杯、绿植\n"
               "木质展台\n"          # 模型换行续写，没有 KEY:
               "POSITION: 展台右侧")
    got2 = parse_state(wrapped)
    check(set(got2) == {"FEATURE", "ENV", "POSITION"},
          "多行（续行无 KEY:）⇒ 仍只出三个键，续行被忽略")
    check("木质展台" not in "".join(got2.values()),
          "续行不污染相邻字段（解析不到就丢）")

    eight = "\n".join(f"{k}: 值{k}" for k in
                      ("ATTENTION", "ACTION", "EMOTION", "FEATURE",
                       "BEHAVIOR", "RELATION", "ENV", "POSITION"))
    got3 = parse_state(eight)
    check(len(got3) == 8, "8 键形态仍可解析（ORCH_TEXT_DESC_ONLY=0 的回滚路径）")

    check(parse_state("好的，我看到了一个用户。") == {},
          "对话式回复 ⇒ 空 dict（调用方回退原文）")
    check(render_state(got) ==
          "FEATURE: " + got["FEATURE"] + "\nENV: " + got["ENV"] +
          "\nPOSITION: " + got["POSITION"], "按 STATE_KEYS 顺序渲染三行")


def t_state_head_contract() -> None:
    print("\n[12] state_head 题集 / assemble：schema §3 对齐 + 动作单题")
    try:
        from state_head.questions import build_questions
        from state_head.assemble import assemble
    except Exception as exc:  # noqa: BLE001
        print(f"  [SKIP] 本机 import 不到 state_head（{exc}）")
        return

    full = build_questions(stage="full")
    turn = build_questions(stage="turn")
    check(len(full) == 16, f"full = 16 问（§3 全枚举槽位）：{len(full)}")
    check(len(turn) == 15 and "env.scene" not in turn,
          f"turn = 15 问、不含 env.scene：{len(turn)}")
    check("act.type" in full and "env.scene" in full, "act.type / env.scene 都在 full 里")
    check(not any(q.split(".")[0] == "act" and q not in ("act.type", "act.count")
                  for q in full),
          "不再按动作类别拆 noul（没有 act.nod / act.wave …）")
    check(set(full["act.type"]["criteria"]) == {
        "nod", "shake_head", "wave", "point", "thumbs_up", "raise_hand",
        "clap", "hand_over_object", "other", "none"},
        "act.type 是 10 选 1（8 类 + other + none）")
    check(list(full["act.type"]["criteria"])[-2:] == ["other", "none"],
          "none 放最后（不让『没动作』占 argmax 先手）")

    base = {"person_visible": {"type": "noul", "noul": 0.92}}

    def _act(choice: str, p: float = 0.8, count: float = 0.0) -> dict:
        return {**base,
                "act.type": {"type": "choice", "choice": choice,
                             "probabilities": {choice: p}},
                "act.count": {"type": "score", "score": count}}

    out = assemble(_act("none", 0.9), stage="full", has_scene=True)
    check(out["ACTION"]["events"] == [],
          "act.type=none ⇒ events=[]（看到人、确实没动作）")

    out = assemble(_act("nod", 0.8), stage="full", has_scene=True)
    check(out["ACTION"]["events"] == [
        {"type": "nod", "count": 1, "confidence": 0.8}],
        f"nod p=0.8 ⇒ 单条事件：{out['ACTION']['events']}")
    check(len(out["ACTION"]["events"]) == 1, "一拍最多 1 条事件")

    out = assemble(_act("nod", 0.4), stage="full", has_scene=True, action_p=0.5)
    check(out["ACTION"]["events"] == [], "选中项低于 action_p ⇒ []（不让噪声动作入状态）")

    out = assemble({**base, "act.type": {"type": "choice", "choice": "other",
                                         "probabilities": {"other": 0.7}}},
                   stage="full", has_scene=True)
    check(out["ACTION"]["events"][0]["type"] == "other",
          "other 是 schema 枚举值，原样出（细节由文本描述补）")

    out = assemble({"person_visible": {"type": "noul", "noul": 0.1},
                    "act.type": {"type": "choice", "choice": "nod",
                                 "probabilities": {"nod": 0.9}}},
                   stage="full", has_scene=True)
    check(out["ACTION"]["events"] is None, "人不在画面 ⇒ events=null（不是 []）")

    # 白名单裁剪：人不见的整份 null 不裁（§5.1 门控读数要原样回编排侧）
    from state_head.assemble import project_slots
    cut = project_slots(out, {"act.type", "act.count"})
    check((cut.get("ACTION") or {}).get("events", "missing") is None,
          "人不见 ⇒ project_slots 不裁，ACTION.events=null 原样保留")


def main() -> int:
    t_merge_e2()
    t_person_missing()
    t_render()
    t_fuse()
    t_merge_text()
    t_publish_vision()
    t_state_client()
    t_slots()
    t_hysteresis()
    t_desc_only_parse()
    t_state_head_contract()
    print()
    if _failures:
        print(f"❌ {len(_failures)} 项失败：")
        for m in _failures:
            print("   -", m)
        return 1
    print("✅ 全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
