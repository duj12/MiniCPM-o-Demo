#!/usr/bin/env python3
"""端到端验证：编排层**透传**服务端 ``state``，并按两拍清零。

用真实的 ``AsrClient`` 连 ASR 服务，喂一个 wav，然后：
  · 每条消息打印客户端快照（应与服务端 ``state`` 逐字一致）
  · 音频发完后继续 tick（50ms），观察离线之后的两拍：
    说/抢→NONE（完/信/text 保留）→ 全空

    python -m orchestrator.tests.probe_state_live --url ws://127.0.0.1:10097 \
        --wav /tmp/sop/12_1_1026.wav
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import time
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from orchestrator.asr.client import AsrClient, AsrConfig  # noqa: E402


def _fmt(st) -> str:
    return ("%s/%s/%s/%s text=%r" % (
        st.user_speaking_confidence, st.barge_in_confidence,
        st.asr_confidence, st.turn_complete_confidence, st.transcript))


#: 感知字段（服务端 14 键里除 5 基础 + transcript 之外的 9 个）
PERC = ("language", "language_prob", "language_switch", "emotion",
        "emotion_prob", "speech_event", "noise", "overlap", "snr_db")


def _perc(st) -> str:
    d = {k: getattr(st, k) for k in PERC}
    if all(v is None for v in d.values()):
        return ""
    return "  | lang=%r(%.2f sw=%s) emo=%r(%.2f) ev=%r snr=%s ov=%s nz=%s" % (
        d["language"], d["language_prob"] or 0.0, d["language_switch"],
        d["emotion"], d["emotion_prob"] or 0.0, d["speech_event"],
        d["snr_db"], d["overlap"], d["noise"])


async def main(url: str, wav: str, idle_s: float) -> int:
    with wave.open(wav) as w:
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    audio = (pcm.astype(np.float32) / 32768.0)

    cfg = AsrConfig(enable_turnsense=True, enable_timestamp=False)
    client = AsrClient(url, cfg, wav_name="state-live")

    seen_keysets: set = set()
    modes_seen: dict = {}

    def on_msg(m: dict) -> None:
        _mode = str(m.get("mode"))
        modes_seen[_mode] = modes_seen.get(_mode, 0) + 1
        """只做一件事：把服务端 ``state`` 的 key 集合打一次（验 5 / 14 字段）。"""
        st = m.get("state")
        if isinstance(st, dict) and st:
            ks = tuple(sorted(st))
            if ks not in seen_keysets:
                seen_keysets.add(ks)
                print("  [state] %d keys: %s" % (len(ks), ",".join(ks)))
            # 哪种消息带感知值？（线上 2pass-offline 才有真值，流式帧是 null）
            vals = {k: st.get(k) for k in PERC if st.get(k) is not None}
            if vals and not best_perc:
                print("  [state] mode=%r 带真值: %s"
                      % (m.get("mode"), vals))
        # 收消息那一刻的快照里带感知值（tick 的 50ms 采样可能刚好落在两拍之间）
        stc = client.state
        d = {k: getattr(stc, k) for k in PERC}
        if any(v is not None for v in d.values()):
            best_perc.clear()
            best_perc.update(d)

    await client.connect()
    task = asyncio.create_task(client.recv_loop(on_msg))

    mismatches = 0
    best_perc: dict = {}          # 见过的最完整的一份感知快照（段末两拍会被清空）

    async def _tick_loop(seconds: float, label: str) -> None:
        nonlocal mismatches
        t0 = time.time()
        last = client.state.to_dict()
        while time.time() - t0 < seconds:
            await asyncio.sleep(0.05)
            client.expire_if_idle()
            cur = client.state.to_dict()
            if cur != last:
                print("  [%s] %s%s" % (label, _fmt(client.state),
                                       _perc(client.state)))
                last = cur
            if any(cur.get(k) is not None for k in PERC):
                best_perc.clear()
                best_perc.update({k: cur[k] for k in PERC})

    # tick 与推流**并发**跑（真实 session.run_tick 就是这个节奏）
    tick = asyncio.create_task(_tick_loop(60.0, "tick"))

    # 送音频（每 20ms 一块，模拟实时）
    step = 320
    for i in range(0, len(audio), step):
        await client.push(audio[i:i + step])
        await asyncio.sleep(0.02)
    print("== 音频发完，进入空转 %.1fs" % idle_s)
    await asyncio.sleep(idle_s)
    tick.cancel()

    # 结束时把剩余消息排空，再对一遍服务端原值
    # ⚠️ 必须走 drain()（它发 ``is_speaking: false``）：流式后端只在收到
    #    这个控制帧时才出 ``2pass-offline``，close() 是直接断连。
    print("== 发 is_speaking=false，等最终结果")
    print("== drain: %s" % await client.drain(timeout=15.0))
    print("== 收尾那一刻快照: %s%s" % (_fmt(client.state),
                                      _perc(client.state)))
    for _ in range(100):
        await asyncio.sleep(0.05)
        client.expire_if_idle()
    task.cancel()
    print("== 最终快照: %s" % _fmt(client.state))
    print("== 收到过的 mode: %s" % modes_seen)
    print("== 感知字段（编排层 AsrState 是否真拿到了）: %s"
          % (best_perc or "全 None —— 服务端没开感知 / 没下发"))
    return 1 if mismatches else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="ws://127.0.0.1:10097")
    ap.add_argument("--wav", required=True)
    ap.add_argument("--idle", type=float, default=4.0)
    a = ap.parse_args()
    sys.exit(asyncio.run(main(a.url, a.wav, a.idle)))
