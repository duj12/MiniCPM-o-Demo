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
   （缺 httpx 时这一段跳过 —— 本地开发机常常没装，105/镜像里才有）。

    python -m orchestrator.tests.test_state_merge
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from orchestrator.omni.state_schema import (  # noqa: E402
    merge_e2, fuse_emotion, to_vision_strings, merge_vision_text,
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
def _mk_pub(state_enabled: bool = True, conf: bool = True):
    from orchestrator.interaction.downstream import InteractionDownstream

    d = InteractionDownstream.__new__(InteractionDownstream)
    d.session_id = "t-state-merge"
    d._state_enabled = state_enabled
    d._state_conf_in_text = conf
    d._vlm = ""
    d._vlm_state = {}
    d._vlm_text = ""
    d._state_e2 = {}
    d._state_at = 0.0
    d._state_stage = ""
    d._vlm_stage = ""
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


def main() -> int:
    t_merge_e2()
    t_person_missing()
    t_render()
    t_fuse()
    t_merge_text()
    t_publish_vision()
    t_state_client()
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
