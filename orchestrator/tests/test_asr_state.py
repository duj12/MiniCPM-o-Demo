#!/usr/bin/env python3
"""ASR 状态归纳（``AsrStateTracker``）验证。

**背景**：编排服务要给下游提供五个离散状态量（用户是否出声 / 抢话把握 /
转写 / 转写置信 / 本轮说完把握）。它们的归纳逻辑住在
``orchestrator/asr/client.py`` 的 ``AsrStateTracker`` 里。

本脚本喂**假的消息序列**（不连服务端）验证归纳规则。里面掺了**真实帧**——
下面 ``REAL_*`` 常量是 2026-09 从两个线上服务端（31366 = asr-2pass/C++，
31323 = Fun-ASR-deploy/Python）实录的字段形状，用来锁住两个服务端的兼容性。

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
#  真实帧（从线上服务端实录，字段形状原样保留）
# ====================================================================== #

#: 31366（asr-2pass / C++）：流式置信度走**标量** online_confidence
REAL_ONLINE_CXX = {
    "end_time": 0, "index": 1, "is_final": False, "mode": "2pass-online",
    "online_confidence": 0.7588891983032227, "start_time": 290,
    "text": "今天的", "vad_segments": [], "wav_name": "probe",
}

#: 31323（Fun-ASR-deploy / Python）：流式置信度走 **confidence 对象**
REAL_ONLINE_PY = {
    "mode": "2pass-online", "text": "说出去看", "wav_name": "probe",
    "is_final": False, "start_time": 290, "end_time": 0,
    "vad_segments": [], "timestamp": "",
    "confidence": {"avg": 0.8897, "token": {"chars": ["说"], "scores": [0.891]}},
}

#: 31366 的独立 turnsense 消息（probabilities 是**数组**，带 prediction_id）
REAL_TURNSENSE_CXX = {
    "label": "incomplete", "mode": "turnsense", "prediction_id": 1,
    "probabilities": [0.2126, 0.5752, 0.2121],
    "segment_end": 6040, "segment_start": 290, "speech_duration": 5.75,
    "wav_name": "probe",
}

#: 31323 的独立 turnsense 消息（probabilities 是 **dict**，无 prediction_id）
REAL_TURNSENSE_PY = {
    "mode": "turnsense", "label": "incomplete",
    "probabilities": {"complete": 0.213, "incomplete": 0.5748, "invalid": 0.2122},
    "speech_duration": 6.12, "segment_start": 290, "segment_end": 6040,
}

#: 流结束收尾帧：turnsense **嵌在 offline 里**，且没有独立的 mode=turnsense
REAL_OFFLINE_EMBEDDED = {
    "confidence": {"avg": 0.95652, "token": {"chars": ["今"], "scores": [0.847]}},
    "dbfs": -22.29, "end_time": 5980, "index": 1, "is_final": True,
    "mode": "2pass-offline", "slice_type": 2,
    "start_time": 290, "text": "今天的说出去看的风景啊",
    "turnsense": {"label": "incomplete", "prediction_id": 1,
                  "probabilities": [0.2126, 0.5753, 0.2121],
                  "speech_duration": 5.69},
}


def online(text, conf=None, cxx=True):
    """造一条流式帧。``cxx=True`` 用标量字段名，否则用对象字段名。"""
    m = {"mode": "2pass-online", "text": text, "start_time": 290,
         "is_final": False}
    if conf is not None:
        if cxx:
            m["online_confidence"] = conf
        else:
            m["confidence"] = {"avg": conf, "token": {}}
    return m


def offline(text, conf=None, ts=None, is_final=False):
    m = {"mode": "2pass-offline", "text": text, "start_time": 290,
         "end_time": 5980, "is_final": is_final}
    if conf is not None:
        m["confidence"] = {"avg": conf, "token": {}}
    if ts is not None:
        m["turnsense"] = ts
    return m


def main() -> int:
    from orchestrator.asr.client import (
        AsrStateTracker, extract_turnsense, is_turnsense_complete,
        parse_online_confidence,
    )

    print("=" * 68)
    print("ASR 状态归纳验证")
    print("-" * 68)

    # ---------------- 兼容层 ----------------
    print("\n[兼容层] 两个服务端的字段差异")
    check(abs(parse_online_confidence(REAL_ONLINE_CXX) - 0.7588) < 1e-3,
          "C++ 标量 online_confidence 能解析")
    check(abs(parse_online_confidence(REAL_ONLINE_PY) - 0.8897) < 1e-4,
          "Python confidence 对象能解析")
    check(parse_online_confidence({"mode": "2pass-online", "text": "x"}) is None,
          "两者都缺失时返回 None")

    ts_cxx = extract_turnsense(REAL_TURNSENSE_CXX)
    check(ts_cxx["probabilities"] == [0.2126, 0.5752, 0.2121],
          "turnsense 数组形态原样保留")
    ts_py = extract_turnsense(REAL_TURNSENSE_PY)
    check(len(ts_py["probabilities"]) == 3
          and abs(ts_py["probabilities"][1] - 0.5748) < 1e-4,
          "turnsense dict 形态被归一成数组（顺序 complete/incomplete/invalid）")
    check(extract_turnsense({"mode": "2pass-online", "text": "x"}) is None,
          "普通流式帧里没有 turnsense -> None")

    emb = extract_turnsense(REAL_OFFLINE_EMBEDDED)
    check(emb is not None and emb["label"] == "incomplete",
          "嵌在 offline 里的 turnsense 能被挖出来")
    check(emb["segment_start_ms"] == 290 and emb["segment_end_ms"] == 5980,
          "嵌入形态缺 segment 起止时用宿主 offline 的 start/end 补上")

    check(is_turnsense_complete({"label": "complete"}) is True,
          "is_turnsense_complete: label=complete")
    check(is_turnsense_complete({"label": "", "prediction_id": 0}) is True,
          "is_turnsense_complete: 无 label 时回退 prediction_id==0")
    check(is_turnsense_complete(ts_cxx) is False, "incomplete 不算完整")
    check(is_turnsense_complete(None) is False, "None 不算完整")

    # ---------------- 转写累积 / 覆盖 ----------------
    print("\n[transcript] 增量拼接与最终覆盖")
    tr = AsrStateTracker()
    tr.update(online("今天"))
    tr.update(online("吃饭"))
    s = tr.update(online("了吗"))
    check(s.transcript == "今天吃饭了吗", "流式增量片段被拼成累积文本")

    s = tr.update(offline("今天吃饭了吗？", conf=0.96))
    check(s.transcript == "今天吃饭了吗？", "最终结果覆盖流式累积文本")

    # 下一段从空开始
    s = tr.update(online("明天"))
    check(s.transcript == "明天", "新一段从空开始（不残留上一段）")

    # ---------------- asr_confidence ----------------
    # 三档：>=0.8 HIGH / 0.6~0.8 MEDIUM / <0.6 LOW；无信号 NONE。
    # ⚠️ 早先只有两档、且流式拿 0.6 当 HIGH 线 —— 0.6~0.8 的低质量临时结果
    #    也被显示成 HIGH，说话时满屏"高置信"看不出哪些能信。
    print("\n[asr_confidence] 三档（0.8 / 0.6 两条线），流式与离线同一套口径")
    for conf, want in ((0.95, "HIGH"), (0.80, "HIGH"),
                       (0.79, "MEDIUM"), (0.70, "MEDIUM"), (0.60, "MEDIUM"),
                       (0.59, "LOW"), (0.30, "LOW")):
        tr = AsrStateTracker()
        s = tr.update(online("今天的", conf=conf))
        check(s.asr_confidence == want,
              f"流式 conf={conf} -> {want}（实际 {s.asr_confidence}）")

    for conf, want in ((0.95, "HIGH"), (0.80, "HIGH"), (0.79, "MEDIUM"),
                       (0.60, "MEDIUM"), (0.59, "LOW")):
        tr = AsrStateTracker()
        tr.update(online("今天的", conf=0.9))
        s = tr.update(offline("今天的天气", conf=conf))
        check(s.asr_confidence == want,
              f"离线 conf={conf} -> {want}（实际 {s.asr_confidence}）")

    # 流式文本被服务端过滤（返回了结果但文本空）→ LOW，不是 NONE
    tr = AsrStateTracker()
    s = tr.update(online("", conf=0.4))
    check(s.asr_confidence == "LOW", "空帧带低分 -> LOW（有信号，只是被丢）")

    tr2 = AsrStateTracker()
    check(tr2.state.asr_confidence == "NONE", "没有任何信号 -> NONE")

    # ⚠️ **空帧不带分数时，保留上一次的值**（实测 31366/C++：有文本的帧
    #    一定带 online_confidence、空帧一定不带）。早先"拿不到分就置 NONE"，
    #    于是每个空帧都擦掉上一帧刚给的分 → 信 在 HIGH/NONE 之间逐帧抖动。
    tr = AsrStateTracker()
    check(tr.update(online("好了现在", conf=0.854)).asr_confidence == "HIGH",
          "有文本 0.854 -> HIGH")
    check(tr.update(online("", conf=None)).asr_confidence == "HIGH",
          "空帧**无分** -> 保持 HIGH（不擦掉上一帧的分数）")
    check(tr.update(online("呃我今", conf=0.795)).asr_confidence == "MEDIUM",
          "下一帧有文本 0.795 -> MEDIUM（正常更新）")

    # 空帧**带高分**：文本因其它原因被丢，按分数正常分档、不凭空降级
    tr = AsrStateTracker()
    check(tr.update(online("", conf=0.85)).asr_confidence == "HIGH",
          "空帧带 0.85 -> HIGH（按分数分档，不因文本空而降级）")

    # ⚠️ **离线收尾帧同样**：空文本 + 不带分 → 保留上次（不擦成 NONE）。
    #    与流式空帧同一口径。收尾帧（is_final=true / text 空）只表示
    #    "这段结束了"，不代表这一轮没置信度。
    tr = AsrStateTracker()
    tr.update(online("今天", conf=0.9))
    check(tr.update(offline("", conf=None)).asr_confidence == "HIGH",
          "空文本 offline（无分）-> 保留上次 HIGH（收尾帧不擦分数）")
    tr = AsrStateTracker()
    tr.update(online("今天", conf=0.9))
    check(tr.update(offline("", conf=0.85)).asr_confidence == "HIGH",
          "空文本 offline（带 0.85）-> 按分数分档 HIGH")
    tr = AsrStateTracker()
    tr.update(online("今天", conf=0.9))
    check(tr.update(offline("今天天气", conf=0.5)).asr_confidence == "LOW",
          "有文本 offline（0.5）-> 正常更新 LOW")

    # ---------------- barge_in ----------------
    print("\n[barge_in_confidence] 按带文本的流式帧爬档")
    tr = AsrStateTracker()
    check(tr.state.barge_in_confidence == "NONE", "一帧都没来 -> NONE")
    check(tr.update(online("今天")).barge_in_confidence == "LOW", "第 1 帧 -> LOW")
    check(tr.update(online("吃饭")).barge_in_confidence == "MEDIUM", "第 2 帧 -> MEDIUM")
    check(tr.update(online("了吗")).barge_in_confidence == "HIGH", "第 3 帧 -> HIGH")
    check(tr.update(online("啊")).barge_in_confidence == "HIGH", "之后保持 HIGH")

    # ⚠️ **offline 那一拍 = HIGH**（新语义：拿到带转写的最终结果 = 确实
    #    说出了内容）。这是"至少让 IC 收到一次 HIGH"的一拍缓冲 ——
    #    offline 同时是「本轮结束」，若直接关段这个 HIGH 就永远观测不到。
    check(tr.update(offline("今天吃饭了吗？", conf=0.96)).barge_in_confidence == "HIGH",
          "offline（拿到带转写的最终结果）-> 抢 HIGH")
    # 下一拍（tick 消费缓冲；IC 已收到过 HIGH）→ 立即 NONE
    tr.expire_if_idle()
    check(tr.state.barge_in_confidence == "NONE",
          "下一拍 抢 -> NONE（本轮内量，轮结束即清）")
    # ⚠️ 跨句**不能累积**：这是实测踩过的 bug（早先只在 `_clear_all` 里重置，
    #    而那是懒触发的，静音期间不执行 → 句3 刚开始就 HIGH）。
    #    现在帧数在**段开始时**重置（见 `_on_online`），所以不依赖 offline 归零。

    # 三句连续 —— 每句都要从 LOW 起，不能累积
    tr = AsrStateTracker()
    firsts = []
    for i in (1, 2, 3):
        firsts.append(tr.update(online(f"句{i}")).barge_in_confidence)
        tr.update(offline(f"句{i}内容", conf=0.95))
    check(firsts == ["LOW", "LOW", "LOW"],
          f"连续三句各自从 LOW 起（实际 {firsts}）—— 不跨句累积")

    # 单句内仍能爬到 HIGH
    tr = AsrStateTracker()
    for _ in range(3):
        tr.update(online("字"))
    check(tr.state.barge_in_confidence == "HIGH", "单句内三帧仍爬到 HIGH")

    tr.reset_turn()
    check(tr.state.barge_in_confidence == "NONE", "reset_turn 后归零")

    # 空文本帧不计数
    tr = AsrStateTracker()
    check(tr.update(online("")).barge_in_confidence == "NONE",
          "空文本帧不计入 barge_in")

    # ---------------- user_speaking ----------------
    print("\n[user_speaking] 段状态机（不塌窗、不闪烁）")
    tr = AsrStateTracker()
    check(tr.state.user_speaking_confidence == "NONE", "没消息 -> NONE")
    # 盲窗：段已开但还没有文本 —— 这正是原方案读成 NONE 的地方
    check(tr.update(online("")).user_speaking_confidence == "MEDIUM",
          "空文本流式帧（盲窗）-> MEDIUM，不是 NONE")
    check(tr.update(online("今天")).user_speaking_confidence == "HIGH",
          "段内有文本 -> HIGH")
    # ⚠️ offline 那一拍 `说` 仍报 HIGH（一拍缓冲，让 IC 一定收到
    #    "用户刚才在说话 + 确实说出了内容"这个组合）；下一拍才 NONE。
    #    早先这里断言 LOW（"用户可能还在说下半句"的防闪烁保持窗口），
    #    那个语义已废弃（一轮结束了还在报 LOW 是错的）。
    check(tr.update(offline("今天", conf=0.96)).user_speaking_confidence == "HIGH",
          "offline 那一拍 -> 说 仍 HIGH（缓冲，保证 IC 收到）")
    tr.expire_if_idle()          # 下一拍（tick 消费缓冲）
    check(tr.state.user_speaking_confidence == "NONE",
          "下一拍 -> 说 NONE（本轮内量，轮结束即清）")
    # 低置信度被过滤 -> LOW
    tr2 = AsrStateTracker()
    tr2.update(online("今天", conf=0.9))
    s = tr2.update(offline("", conf=0.4))
    check(s.asr_confidence == "LOW", "被过滤的最终结果 -> asr_confidence=LOW")

    # ---------------- turn_complete ----------------
    print("\n[turn_complete_confidence] NONE/LOW/MEDIUM/HIGH")
    tr = AsrStateTracker()
    check(tr.state.turn_complete_confidence == "NONE", "无文本 -> NONE")
    check(tr.update(online("今天")).turn_complete_confidence == "LOW",
          "普通流式帧 -> LOW")

    # offline 是服务端对**这一段**的拍板 -> HIGH
    tr = AsrStateTracker()
    tr.update(online("今天的"))
    s = tr.update(offline("今天的说出去看的风景", conf=0.96))
    check(s.turn_complete_confidence == "HIGH",
          "2pass-offline（段拍板）-> HIGH")

    # 切分之后的流式帧：上一段已拍板、当前这段还在长 -> MEDIUM
    tr = AsrStateTracker()
    tr.update(online("今天的"))
    check(tr.update(offline("今天的", conf=0.96)).turn_complete_confidence == "HIGH",
          "第一段 offline -> HIGH")
    check(tr.update(online("下一段")).turn_complete_confidence == "MEDIUM",
          "VAD 切分之后的流式帧 -> MEDIUM（不是 HIGH）")
    check(tr.update(online("下一段更多")).turn_complete_confidence == "MEDIUM",
          "续接段保持 MEDIUM")

    # turnsense=incomplete 不拍板；等 offline 到了才 HIGH
    tr = AsrStateTracker()
    tr.update(online("今天的"))
    s = tr.update(REAL_TURNSENSE_CXX)
    check(s.turn_complete_confidence != "HIGH",
          "turnsense=incomplete 不拍板")
    s = tr.update(offline("今天的说出去看的风景", conf=0.96))
    check(s.turn_complete_confidence == "HIGH", "随后 offline 拍板 -> HIGH")

    # turnsense=complete 先拍板（此时 offline 可能还没回来）
    tr = AsrStateTracker()
    tr.update(online("今天"))
    s = tr.update({"mode": "turnsense", "label": "complete",
                   "prediction_id": 0, "probabilities": [0.9, 0.05, 0.05]})
    check(s.turn_complete_confidence == "HIGH", "turnsense=complete -> HIGH")
    s = tr.update(offline("今天。", conf=0.96))
    check(s.turn_complete_confidence == "HIGH", "complete 之后保持 HIGH")

    # invalid 之后可能永远没有 offline —— 状态不能悬空
    tr = AsrStateTracker()
    tr.update(online("今天"))
    tr.update({"mode": "turnsense", "label": "invalid",
               "prediction_id": 2, "probabilities": [0.1, 0.1, 0.8]})
    s = tr.update(online("新的一句"))
    check(s.turn_complete_confidence == "LOW",
          "invalid 后没有 offline：下一个流式帧能回到 LOW（状态不悬空）")

    # 嵌入形态（流结束收尾帧）
    tr = AsrStateTracker()
    tr.update(online("今天的"))
    s = tr.update(REAL_OFFLINE_EMBEDDED)
    check(s.transcript == "今天的说出去看的风景啊", "嵌 turnsense 的 offline 同样替换转写")
    check(s.turn_complete_confidence == "HIGH", "嵌 turnsense 的 offline 同样拍板 -> HIGH")

    # ---------------- 真实双句序列 ----------------
    print("\n[真实帧回放] 31366 句1+静音+句2（实录序列）")
    tr = AsrStateTracker()
    seq = [
        online("今天的", 0.7588),
        online("说出去看", 0.8916),
        REAL_TURNSENSE_CXX,
        offline("今天的说出去看的风景", conf=0.95652),
        online("今天的", 0.76),
        online("说出去看", 0.89),
        {**offline("今天的说出去看的风景", conf=0.95652, is_final=True),
         "turnsense": {"label": "incomplete", "prediction_id": 1,
                       "probabilities": [0.2126, 0.5753, 0.2121],
                       "speech_duration": 5.76}},
    ]
    seen = [tr.update(m) for m in seq]
    check(len(seen) == 7, "整条序列不抛异常")
    check([s.turn_complete_confidence for s in seen] ==
          ["LOW", "LOW", "LOW", "HIGH", "MEDIUM", "MEDIUM", "HIGH"],
          "turn_complete: 流式 LOW -> offline 拍板 HIGH -> 续接段 MEDIUM -> 再拍板 HIGH")
    check(seen[4].user_speaking_confidence == "HIGH",
          "新段起来后 user_speaking 回到 HIGH（不是卡在 LOW）")
    check(seen[-1].transcript == "今天的说出去看的风景",
          "第二轮转写正确覆盖")
    # ⚠️ barge 的**新语义**（本次重构）：
    #    · 段内按「带文本的流式帧」数爬档：1→LOW / 2→MEDIUM / 3+→HIGH
    #    · offline **拿到带转写的最终结果** → 本段 HIGH（不再归零！）
    #    · 段一关（本轮结束）→ **立即 NONE**
    #    所以要分两个视角看：`update()` 的**返回值**是"该消息处理后"的快照，
    #    而段关闭后 `说`/`抢` 已 NONE。
    # seen[2] 是 REAL_TURNSENSE_CXX —— turnsense 消息**关段但不带转写**，
    # 所以那一拍 `抢` 就是 NONE（段关了、且没有"说出内容"的证据）。
    # 这正是「空文本不置 _barge_final」该有的表现。
    check([s.barge_in_confidence for s in seen[:4]] ==
          ["LOW", "MEDIUM", "NONE", "HIGH"],
          f"句1：两帧爬到 MEDIUM；turnsense 关段 -> NONE；"
          f"offline（有转写）-> HIGH（实际 {[s.barge_in_confidence for s in seen[:4]]}）")
    check([s.barge_in_confidence for s in seen[4:]] == ["LOW", "MEDIUM", "HIGH"],
          f"句2：从 LOW 重新爬（不跨句累积）；offline -> HIGH"
          f"（实际 {[s.barge_in_confidence for s in seen[4:]]}）")

    # ---------------- 一句结束后状态要清空 ----------------
    # ⚠️ 这是实测踩过的 bug：状态机是**纯事件驱动**的，所有字段只在
    #    "收到下一条消息"时被覆盖。可一句话说完就**不再有消息**了 ——
    #    于是 `抢`/`信`/`完` 会**永远挂着旧值**（只有 `说` 会掉 NONE），
    #    表现是"早就不说话了，状态栏还写着置信度 HIGH"。
    print("\n[句末清空] 静音后四个状态量都要归 NONE（不能粘住）")
    tr = AsrStateTracker()
    tr.update(online("今天", conf=0.9))
    tr.update(offline("今天天气", conf=0.95))
    tr.expire_if_idle()      # 下一拍：tick 消费掉"刚拍板"那一拍的缓冲
    s = tr.state
    # 「本轮内」的量（`说`/`抢`）：一轮结束（+一拍缓冲被消费后）**立即** NONE。
    # 「本轮结论」（`信`/`完`/`transcript`）：**保留**一段时间供 UI 读。
    check(s.barge_in_confidence == "NONE", "offline 后 抢 立即 NONE（本轮内）")
    check(s.user_speaking_confidence == "NONE",
          "offline 后 说 立即 NONE（本轮内）")
    check(s.asr_confidence != "NONE", "offline 后 信 仍保留本段结果（本轮结论）")

    # 模拟 hold 窗口过期（不去真等 1.2s，直接压时间戳）
    tr._hold_until_ms = 0
    s = tr.state
    check(s.user_speaking_confidence == "NONE", "过期后 说 -> NONE")
    check(s.barge_in_confidence == "NONE", "过期后 抢 -> NONE（早先会粘住）")
    check(s.asr_confidence == "NONE", "过期后 信 -> NONE（早先会粘住）")
    check(s.turn_complete_confidence == "NONE", "过期后 完 -> NONE（早先会粘住）")
    check(s.transcript == "", "过期后 transcript 也清空")

    # ⚠️ 反向：**说话中不能被清**（段开着说明新一句已经开始）
    tr2 = AsrStateTracker()
    tr2.update(online("今天", conf=0.9))
    tr2._hold_until_ms = 0          # 即使 hold 已过期
    s = tr2.state
    check(s.user_speaking_confidence == "HIGH" and s.transcript == "今天",
          "段开着时读快照**不会**被清（新一句不被清坏）")

    # ⚠️ 反向：offline 后**立刻**来新一句，不能被清坏、也不该等 hold
    tr3 = AsrStateTracker()
    tr3.update(online("今天", conf=0.9))
    tr3.update(offline("今天天气", conf=0.95))
    tr3.update(online("明天", conf=0.9))
    tr3.update(online("开会", conf=0.9))
    s = tr3.state
    check(s.transcript == "明天开会", "新一句的 transcript 正确（旧句已被覆盖）")
    check(s.barge_in_confidence == "MEDIUM",
          "抢 从新句重新算（句2 两帧 -> MEDIUM，不带上句的计数）")

    # ---------------- state 快照 ----------------
    print("\n[快照] to_dict 字段齐全")
    d = AsrStateTracker().state.to_dict()
    check(set(d) == {"user_speaking_confidence", "barge_in_confidence",
                     "transcript", "asr_confidence", "turn_complete_confidence"},
          "五个字段齐全")

    # ---------------- 「本轮内」vs「本轮结论」两类语义 ----------------
    # 这是本次重构的核心约定，单独一节锁住，防止以后被实现成一样的。
    print("\n[两类语义] 本轮内（说/抢）立即清，本轮结论（完/信/text）保留")

    # ① 一拍缓冲：offline 那一拍说/抢仍 HIGH（保证 IC 一定收到）
    #
    # ⚠️ **必须模拟 recv_loop 的双读**：它对每条消息读两次快照 ——
    #    `self._tracker.update(msg)`（内部 `return self.state`）一次，
    #    紧接的 `on_message(msg)` 里再读一次，**后者才是发给 IC/UI 的**。
    #    早先只在 `update()` 的返回值上断言，恰好也是 HIGH，把这个 bug
    #    掩盖了（清空逻辑一度放在读快照路径里 → 第一次读就消耗掉缓冲，
    #    发给 IC 的已是 NONE）。用户一眼看出来的就是这个。
    tr = AsrStateTracker()
    tr.update(online("今天"))
    tr.update(offline("今天天气", conf=0.95))   # 内部那次读（被丢弃）
    s = tr.state                                # ← on_message 里给 IC 的那次
    check(s.user_speaking_confidence == "HIGH" and s.barge_in_confidence == "HIGH",
          "offline 那一拍：说/抢 仍 HIGH（经 recv_loop 双读后仍成立，IC 必达）")

    # ② 下一拍（tick）：说/抢 立即 NONE，但 完/信/text **仍在**
    tr.expire_if_idle()
    s = tr.state
    check(s.user_speaking_confidence == "NONE" and s.barge_in_confidence == "NONE",
          "下一拍：说/抢 立即 NONE（本轮内量）")
    check(s.turn_complete_confidence == "HIGH" and s.asr_confidence != "NONE"
          and s.transcript == "今天天气",
          "同一拍：完/信/text 仍在（本轮结论，保留供 UI 读）")

    # ③ 超过 TURN_HOLD_MS 后，本轮结论才清
    tr._hold_until_ms = 0          # 压时间戳，不等真时间
    tr.expire_if_idle()
    s = tr.state
    check(s.turn_complete_confidence == "NONE" and s.asr_confidence == "NONE"
          and s.transcript == "",
          "超过保留窗口：完/信/text 一并清空")
    check(s.user_speaking_confidence == "NONE" and s.barge_in_confidence == "NONE",
          "说/抢 仍是 NONE（不因保留窗口而复活）")

    # ④ `expire_if_idle()` 重复调用不改变结果（tick 50ms 跑一次，会重复调）
    tr2 = AsrStateTracker()
    tr2.update(online("你好"))
    tr2.update(offline("你好世界", conf=0.95))
    tr2.expire_if_idle()           # 第一次：消费缓冲
    a = tr2.state.to_dict()
    tr2.expire_if_idle()           # 第二次：无变化
    tr2.expire_if_idle()
    b = tr2.state.to_dict()
    check(a == b, "expire_if_idle 重复调用幂等（tick 会反复调它）")

    # ⑤ 空文本的最终结果**不置** barge=HIGH（那只是 VAD 收尾帧）
    tr3 = AsrStateTracker()
    tr3.update(online("喂"))
    s = tr3.update(offline("", conf=None))
    check(s.barge_in_confidence != "HIGH",
          f"空文本 final 不置 抢=HIGH（实际 {s.barge_in_confidence}）")

    # ⑥ TURN_HOLD_MS 可被环境变量覆盖
    import importlib
    import os as _os
    _os.environ["ORCH_ASR_TURN_HOLD_MS"] = "2500"
    import orchestrator.asr.client as _c
    importlib.reload(_c)
    check(_c.TURN_HOLD_MS == 2500, "ORCH_ASR_TURN_HOLD_MS 覆盖生效")
    _os.environ.pop("ORCH_ASR_TURN_HOLD_MS", None)
    importlib.reload(_c)

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
