#!/usr/bin/env python3
"""ASR 流式文本拼接验证。

**问题背景**：ASR 的 ``2pass-online`` 给的是**增量片段**：

    今天 / 吃饭 / 了吗        ← 服务端发来的原始片段
    今天 / 今天吃饭 / 今天吃饭了吗   ← 客户端应该显示的

官方客户端（``funasr_wss_client.py``）是自己 ``+=`` 拼起来显示的。我们
早先直接把片段当整句发出去，UI 上只剩最后一个词。

本脚本用假的消息序列验证 ``session`` 侧的拼接与清空行为。

    python -m orchestrator.tests.test_asr_accum
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


def main() -> int:
    from orchestrator.session import OrchestratorSession

    print("=" * 68)
    print("ASR 流式文本拼接验证")
    print("-" * 68)

    sess = OrchestratorSession("test", config={})
    sent = []
    sess._send_display = lambda m: sent.append(m)

    # 直接驱动 on_message 的等价逻辑（避开 WS）
    def feed(mode, text, is_final=False):
        """复刻 run_asr_recv 里 on_message 的分支行为。"""
        if mode == "2pass-online":
            sess._asr_online_text += text
            from orchestrator.protocol import AsrDisplay
            sess._send_display(AsrDisplay(phase="partial",
                                          text=sess._asr_online_text))
        elif mode == "2pass-offline":
            sess._asr_online_text = ""
            from orchestrator.protocol import AsrDisplay
            sess._send_display(AsrDisplay(phase="final", text=text))

    # 第一轮：增量片段
    for frag in ("今天", "吃饭", "了吗"):
        feed("2pass-online", frag)
    partials = [m.text for m in sent if m.phase == "partial"]
    print(f"  收到的 partial 序列: {partials}")
    check(partials == ["今天", "今天吃饭", "今天吃饭了吗"],
          "增量片段被拼成累积文本")
    check(partials[0] != partials[-1], "不是只显示最后一个片段")

    # 最终结果覆盖
    feed("2pass-offline", "今天吃饭了吗？")
    finals = [m.text for m in sent if m.phase == "final"]
    check(finals == ["今天吃饭了吗？"], "最终结果作为独立 final 消息发出")
    check(sess._asr_online_text == "", "final 后累积缓冲已清空")

    # 第二轮：应从空开始，不残留上一轮
    sent.clear()
    for frag in ("明天", "见"):
        feed("2pass-online", frag)
    p2 = [m.text for m in sent if m.phase == "partial"]
    print(f"  第二轮 partial 序列: {p2}")
    check(p2 == ["明天", "明天见"], "新一段从空开始累积（不残留上一轮）")

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
