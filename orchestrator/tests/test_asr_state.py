#!/usr/bin/env python3
"""ASR 状态快照（``AsrStateTracker``）验证 —— **透传契约**。

2026-10 起五个状态量（说 / 抢 / 信 / 完 + ``transcript``）全部由 ASR
服务端计算，放在每条消息的 ``state`` 里下发；编排层不再自己归纳。
本脚本锁住的是客户端仅剩的两件事：

1. **原样透传**服务端的 ``state``（档位归一、缺失补 NONE）；
2. 一轮结束后**分两拍清零**：
   - 拍一（tick）：``说``/``抢`` → NONE，``完``/``信``/``transcript`` 保留
     —— IC 的 ``fresh_turn → ANSWER`` 依赖这一拍
   - 拍二（tick）：全部清空
   （原 ``ORCH_ASR_TURN_HOLD_MS`` 的 1 秒保持窗口已删除）

    python -m orchestrator.tests.test_asr_state
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

_failures: list = []


def check(cond: bool, msg: str) -> None:
    print(f"  [{'OK' if cond else 'FAIL'}] {msg}")
    if not cond:
        _failures.append(msg)


# ====================================================================== #
#  构造消息（字段形状取自两个线上服务端的实录）
# ====================================================================== #

def server_state(**kw) -> dict:
    """服务端 ``state``（14 个 key 里只放 5 个基础 + transcript）。"""
    st = {
        "user_speaking_confidence": "HIGH",
        "barge_in_confidence": "HIGH",
        "transcript": "今天的天气怎么样",
        "asr_confidence": "HIGH",
        "turn_complete_confidence": "LOW",
    }
    st.update(kw)
    return st


def online(text: str, **st) -> dict:
    return {"mode": "2pass-online", "text": text, "is_final": False,
            "state": server_state(transcript=text, **st)}


def offline(text: str, **st) -> dict:
    state = server_state(transcript=text, turn_complete_confidence="HIGH")
    state.update(st)
    return {"mode": "2pass-offline", "text": text, "is_final": False,
            "state": state}


def final_empty() -> dict:
    """收尾帧：``is_final`` 且空文本（服务端 state 是兼容残留，不该透传）。"""
    return {"mode": "2pass-offline", "text": "", "is_final": True,
            "state": server_state(user_speaking_confidence="MEDIUM",
                                  barge_in_confidence="NONE",
                                  transcript="", asr_confidence="NONE",
                                  turn_complete_confidence="LOW")}


def turnsense(label: str) -> dict:
    """独立 turnsense 消息：**不带 state**（真实服务端会带，但那只是同一拍
    的重复快照；这里省略，专注验证 complete/incomplete 的清理判定）。"""
    return {"mode": "turnsense", "text": "", "is_final": False,
            "label": label, "probabilities": {"complete": 0.9, "incomplete": 0.05,
                                              "invalid": 0.05}}


# ====================================================================== #

def test_passthrough():
    from orchestrator.asr.client import AsrStateTracker

    print("\n[1] 原样透传服务端 state")
    tr = AsrStateTracker()
    s = tr.update(online("今天的天气", user_speaking_confidence="HIGH",
                         barge_in_confidence="LOW", asr_confidence="MEDIUM",
                         turn_complete_confidence="LOW"))
    check(s.user_speaking_confidence == "HIGH", "说 取自服务端")
    check(s.barge_in_confidence == "LOW", "抢 取自服务端")
    check(s.transcript == "今天的天气", "transcript 取自服务端")
    check(s.asr_confidence == "MEDIUM", "信 取自服务端")
    check(s.turn_complete_confidence == "LOW", "完 取自服务端")

    print("\n[2] 档位归一：未知/缺失 → NONE，小写 → 大写")
    tr2 = AsrStateTracker()
    s2 = tr2.update({"mode": "2pass-online", "text": "x",
                     "state": {"user_speaking_confidence": "high",
                               "barge_in_confidence": None,
                               "asr_confidence": "???",
                               "transcript": None}})
    check(s2.user_speaking_confidence == "HIGH", "小写 high → HIGH")
    check(s2.barge_in_confidence == "NONE", "null → NONE")
    check(s2.asr_confidence == "NONE", "未知值 → NONE")
    check(s2.turn_complete_confidence == "NONE", "缺失字段 → NONE")
    check(s2.transcript == "", "null transcript → 空串")

    print("\n[3] 没有 state 的消息（turnsense 控制帧等）不改快照")
    tr3 = AsrStateTracker()
    tr3.update(online("你好"))
    tr3.update({"mode": "turnsense", "label": "incomplete", "text": ""})
    check(tr3.state.transcript == "你好", "缺 state 的消息保留上一快照")


def test_turn_clearing():
    from orchestrator.asr.client import AsrStateTracker

    print("\n[4] 段末分两拍清零（无时间保持窗口）")
    tr = AsrStateTracker()
    tr.update(online("今天"))
    s = tr.update(offline("今天天气怎么样"))
    check(s.user_speaking_confidence == "HIGH" and
          s.turn_complete_confidence == "HIGH",
          "offline 那一条仍带服务端原值（说=HIGH / 完=HIGH）")
    check(s.transcript == "今天天气怎么样", "offline 那一条带本轮转写")

    tr.expire_if_idle()            # 拍一
    s = tr.state
    check(s.user_speaking_confidence == "NONE", "拍一：说 → NONE")
    check(s.barge_in_confidence == "NONE", "拍一：抢 → NONE")
    check(s.turn_complete_confidence == "HIGH", "拍一：完 保留（IC 靠它判 ANSWER）")
    check(s.asr_confidence == "HIGH", "拍一：信 保留")
    check(s.transcript == "今天天气怎么样", "拍一：transcript 保留")
    settled = s.to_dict()

    tr.expire_if_idle()            # 拍二
    s = tr.state
    # 拍二：五个基础字段回到默认，**感知字段也一起清成 None**
    # （它们是"刚说完那一段"的结论，段结束两拍后就没有意义了）
    check(all(v is None for k, v in s.to_dict().items()
              if k in ("language", "language_prob", "language_switch",
                       "emotion", "emotion_prob", "speech_event",
                       "noise", "overlap", "snr_db")),
          f"拍二：感知字段清成 None（实际 {s.to_dict()}）")
    check({k: s.to_dict()[k] for k in (
              "user_speaking_confidence", "barge_in_confidence", "transcript",
              "asr_confidence", "turn_complete_confidence")}
          == {"user_speaking_confidence": "NONE",
              "barge_in_confidence": "NONE", "transcript": "",
              "asr_confidence": "NONE",
              "turn_complete_confidence": "NONE"},
          f"拍二：五个基础字段全部清空（实际 {s.to_dict()}）")
    check(settled["turn_complete_confidence"] == "HIGH",
          "两拍之间确实存在'安静但已拍板'的快照")

    print("\n[5] 重复 tick 幂等")
    for _ in range(3):
        tr.expire_if_idle()
    check(tr.state.to_dict() == s.to_dict(), "清零后反复 tick 不再变化")

    print("\n[6] 新的一段开始会取消待清零")
    tr2 = AsrStateTracker()
    tr2.update(online("今天"))
    tr2.update(offline("今天天气怎么样"))
    tr2.update(online("明天"))          # 新段第一帧
    tr2.expire_if_idle()
    check(tr2.state.transcript == "明天", "新段流式帧到达后不再清零")
    tr2.expire_if_idle()
    check(tr2.state.transcript == "明天", "后续 tick 也不清（段还开着）")

    print("\n[7] turnsense=incomplete：段关了但没说完 → 不清零")
    tr3 = AsrStateTracker()
    tr3.update(online("我想去"))
    tr3.update(offline("我想去北京", **{"turn_complete_confidence": "MEDIUM"}))
    tr3.update(turnsense("incomplete"))
    tr3.expire_if_idle()
    tr3.expire_if_idle()
    check(tr3.state.transcript == "我想去北京",
          f"incomplete 后不清零（实际 {tr3.state.to_dict()}）")

    print("\n[8] turnsense=complete：走两拍清零")
    tr4 = AsrStateTracker()
    tr4.update(offline("我想去北京"))
    tr4.update(turnsense("complete"))
    tr4.expire_if_idle()
    check(tr4.state.turn_complete_confidence == "HIGH", "拍一保留完=HIGH")
    tr4.expire_if_idle()
    check(tr4.state.transcript == "" and
          tr4.state.turn_complete_confidence == "NONE", "拍二全部清空")

    print("\n[9] 收尾帧（is_final + 空文本）立即清零，不透传残留值")
    tr5 = AsrStateTracker()
    tr5.update(online("你好"))
    tr5.update(offline("你好世界"))
    s = tr5.update(final_empty())
    check(s.user_speaking_confidence == "NONE",
          f"收尾帧的 说=MEDIUM 不透传（实际 {s.user_speaking_confidence}）")
    check(s.transcript == "" and s.turn_complete_confidence == "NONE",
          "收尾帧直接全部归零")
    tr5.expire_if_idle()
    check(tr5.state.to_dict() == s.to_dict(), "收尾帧后 tick 无副作用")


def test_reset():
    from orchestrator.asr.client import AsrStateTracker

    print("\n[10] reset_turn 立即清空")
    tr = AsrStateTracker()
    tr.update(offline("你好世界"))
    tr.reset_turn()
    d = tr.state.to_dict()
    check(d["transcript"] == "" and d["turn_complete_confidence"] == "NONE",
          f"reset_turn 后全空（实际 {d}）")


def test_perception_passthrough():
    print("\n[10b] 感知字段透传（语种 / 情绪 / 事件 / 噪声 / 交叠 / SNR）")
    from orchestrator.asr.client import AsrStateTracker

    tr = AsrStateTracker()
    noise = {"calibrated": True, "floor_dbfs": -53.8,
             "burst": {"detected": True, "prob": 0.65, "type": "pop"}}
    overlap = {"detected": False, "prob": 0.098, "regions": []}
    s = tr.update(offline(
        "给我介绍一下星云吧",
        language="zh", language_prob=1, language_switch=False,
        emotion="neutral", emotion_prob=0.97, speech_event="speech",
        noise=noise, overlap=overlap, snr_db=17.8,
    ))
    check(s.language == "zh" and s.language_prob == 1.0, "语种字段透传")
    check(s.emotion == "neutral" and s.emotion_prob == 0.97, "语音情绪字段透传")
    check(s.speech_event == "speech", "声学事件字段透传")
    check(s.noise == noise and s.overlap == overlap and s.snr_db == 17.8,
          "噪声 / 交叠 / SNR 原样透传（嵌套 dict 不改写）")
    d = s.to_dict()
    check(d["language"] == "zh" and d["snr_db"] == 17.8,
          "to_dict() 带出感知字段（IC/UI 才拿得到）")

    # 拍一：与 信/完 同档保留（它们是"刚说完那一段"的结论）
    tr.expire_if_idle()
    check(tr.state.language == "zh" and tr.state.snr_db == 17.8,
          "拍一：感知字段保留")
    tr.expire_if_idle()
    check(tr.state.language is None and tr.state.snr_db is None,
          "拍二：感知字段清空")

    # 5 键服务端（感知关闭）：没有这些 key ⇒ 全为 None，不报错
    tr2 = AsrStateTracker()
    s2 = tr2.update(online("今天"))
    check(s2.language is None and s2.noise is None and s2.overlap is None,
          "服务端不带感知字段时全为 None（两种配置并存不炸）")


def test_perception_switch():
    import json

    from orchestrator.asr.client import AsrConfig

    print("\n[11] 首帧带感知开关（默认打开）")
    cfg = json.loads(AsrConfig().to_json("s1"))
    check(cfg["enable_perception"] is True, "默认 enable_perception=True")
    items = {k: v for k, v in cfg.items() if k.startswith("perception_")}
    check(len(items) == 6 and all(items.values()),
          f"6 个逐项开关默认全开（实际 {items}）")

    off = json.loads(AsrConfig(enable_perception=False).to_json("s1"))
    check(off["enable_perception"] is False, "显式关闭能下发 false")


def main() -> int:
    test_passthrough()
    test_turn_clearing()
    test_reset()
    test_perception_passthrough()
    test_perception_switch()
    print("=" * 68)
    if _failures:
        print(f"FAILED: {len(_failures)} 项")
        for f in _failures:
            print(f"  - {f}")
        return 1
    print("通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
