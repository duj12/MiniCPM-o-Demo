#!/usr/bin/env python3
"""TTS 流式合成的下游侧验证（不连服务端）。

覆盖两个**实测踩过的真 bug**：

 ① **逐字喂导致「蹦字」**：服务端的 STREAM_SPLITTER 对每次收到的文本
    独立分句 —— 一次只喂 1~2 个字就被当成逐字成句，每个字都带句末语调
    再补静音。实测同一段 30 字文本：

        每次喂 1 字  → 28 帧 / 17.66s
        每次喂 8 字  →  4 帧 /  7.06s
        每次喂 15 字 →  2 帧 /  6.76s

    所以下游必须攒批。本脚本验证「攒到句末标点 / 够长 / 够久」三条边界。

 ② **第二轮没有回复**：`stream_id` 原本是个常量，于是下一轮的 delta 命中了
    **还没收尾的上一轮的流**（执行器按 `stream_id` 判等），文本被追加进旧流
    → 永远不开新流、永远不播。实测「两轮语音只出 1 次 tts.start」。
    本脚本验证每轮拿到**不同的** stream_id，且收尾后旧 id 不再被复用。

    python -m orchestrator.tests.test_tts_stream
"""
from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

_failures: list = []


def check(cond: bool, msg: str) -> None:
    print(f"  [{'OK' if cond else 'FAIL'}] {msg}")
    if not cond:
        _failures.append(msg)


def main() -> int:
    from orchestrator.downstream.interface import OmniDelta, OmniResponseDone
    from orchestrator.downstream.passthrough import PassthroughDownstream

    print("=" * 68)
    print("TTS 流式下游验证")
    print("-" * 68)

    def delta(t):
        return OmniDelta(t=0, delta_kind="text", text=t)

    def done():
        return OmniResponseDone(t=0, response_id="r", text="")

    # ---------------- ① 攒批 ----------------
    print("\n[攒批] 不能每字一发")
    d = PassthroughDownstream(mode="omni", streaming=True)

    async def feed_deltas(evs):
        out = []
        for e in evs:
            out.extend(await d.on_event(e))
        return out

    # 逐字喂一句，只有到句末标点才应吐出
    acts = asyncio.run(feed_deltas([delta(c) for c in "你好世界。"]))
    check(len(acts) == 1, "逐字喂到句末标点 → 只吐 1 个 Speak（不是 5 个）")
    check(acts and acts[0].text == "你好世界。", "攒出的文本是完整的句子")

    # 不到标点但够长 → 也要吐（否则长句首声一直等）
    d = PassthroughDownstream(mode="omni", streaming=True, flush_chars=14)
    acts = asyncio.run(feed_deltas([delta(c) for c in "一二三四五六七八九十甲乙丙丁"]))
    check(len(acts) == 1 and acts[0].text == "一二三四五六七八九十甲乙丙丁",
          f"无标点但够 {d.flush_chars} 字 → 也吐出去")

    # 逗号**不是**句末，不该在那儿切（切了语调会怪）
    d = PassthroughDownstream(mode="omni", streaming=True)
    acts = asyncio.run(feed_deltas([delta("今天天气不错，"), delta("我们去公园。")]))
    check(len(acts) == 1 and acts[0].text == "今天天气不错，我们去公园。",
          "逗号不切句，一路攒到句号才发")

    # 兜底超时 → LLM 卡住时也要把已有内容发出去。
    # ⚠️ 超时是在**下一个 delta 到达时**判定的（事件驱动，没有定时器），
    #    所以要喂第二个 delta 才会触发；这也正是真实场景 —— LLM 卡住后
    #    接着吐字，那一刻才需要把积压的先发出去。
    d = PassthroughDownstream(mode="omni", streaming=True, flush_seconds=0.15)
    acts = asyncio.run(feed_deltas([delta("半句话没写完")]))
    check(not acts, "刚喂进去（未超时）不会立刻发")
    time.sleep(0.2)
    acts = asyncio.run(feed_deltas([delta("接着写")]))
    check(len(acts) == 1 and acts[0].text == "半句话没写完",
          "卡住超过 flush_seconds 后的下一个 delta → 把积压的先发出去")

    # ---------------- ② 每轮 stream_id 唯一 ----------------
    print("\n[跨轮] 每轮必须是**新的** stream_id")
    d = PassthroughDownstream(mode="omni", streaming=True)

    async def one_turn(text_parts):
        acts = []
        for p in text_parts:
            acts.extend(await d.on_event(delta(p)))
        acts.extend(await d.on_event(done()))
        return acts

    t1 = asyncio.run(one_turn(["第一轮。"]))
    t2 = asyncio.run(one_turn(["第二轮。"]))
    sid1 = {a.stream_id for a in t1 if a.text}
    sid2 = {a.stream_id for a in t2 if a.text}
    check(len(sid1) == 1 and len(sid2) == 1, "每轮内部只用同一个 stream_id")
    check(sid1 != sid2, f"两轮 stream_id 不同（{sid1} vs {sid2}）—— 第二轮才会开新流")
    check(all(s for s in sid1 | sid2), "stream_id 非空（否则执行器走整段合成路径）")

    # 收尾信号带着**本轮**的 id
    fin1 = [a for a in t1 if a.is_final]
    check(len(fin1) == 1 and fin1[0].stream_id in sid1,
          "ResponseDone → 收尾 Speak 带着本轮的 stream_id")

    # 尾文本没到标点也要补发（否则最后半句丢掉）
    d = PassthroughDownstream(mode="omni", streaming=True)
    t = asyncio.run(one_turn(["没有句号结尾"]))
    check(any(a.text == "没有句号结尾" for a in t),
          "收尾时补发未到标点的尾巴（最后一个字不会丢）")

    # 连续多轮都不串
    d = PassthroughDownstream(mode="omni", streaming=True)
    sids = []
    for i in range(4):
        acts = asyncio.run(one_turn([f"第{i}轮。"]))
        sids.append([a.stream_id for a in acts if a.text][0])
    check(len(set(sids)) == 4, f"连续 4 轮各拿不同 id：{sids}")

    # ---------------- ③ 关闭流式时行为不变 ----------------
    print("\n[回归] 关闭流式 → 仍是整段一个 Speak")
    d = PassthroughDownstream(mode="omni", streaming=False)
    acts = asyncio.run(d.on_event(
        OmniResponseDone(t=0, response_id="r", text="整段回复。")))
    check(len(acts) == 1 and acts[0].text == "整段回复。"
          and acts[0].stream_id == "" and acts[0].is_final,
          "非流式：ResponseDone → 一个整段 Speak（stream_id 为空）")

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
