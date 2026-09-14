#!/usr/bin/env python3
"""前端 TS 改动的最小静态检查。

没有 Node/tsc 时的替代验证：括号平衡 + 关键符号存在性 + 开关值。
**不是**类型检查，只用于确认手工编辑没把结构改坏。

    python orchestrator/tests/check_ts.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

TARGET = Path(__file__).resolve().parents[2] / "frontend/mobile/src/mobile-duplex.ts"

_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.S)
_LINE_COMMENT = re.compile(r"//[^\n]*")
_TEMPLATE = re.compile(r"`(?:\\.|[^`\\])*`", re.S)
_SQUOTE = re.compile(r"'(?:\\.|[^'\\])*'")
_DQUOTE = re.compile(r'"(?:\\.|[^"\\])*"')


def strip_literals(s: str) -> str:
    s = _BLOCK_COMMENT.sub("", s)
    s = _LINE_COMMENT.sub("", s)
    s = _TEMPLATE.sub('""', s)
    s = _SQUOTE.sub('""', s)
    s = _DQUOTE.sub('""', s)
    return s


def main() -> int:
    if not TARGET.is_file():
        print(f"找不到 {TARGET}")
        return 2
    src = TARGET.read_text(encoding="utf-8")
    stripped = strip_literals(src)

    fails = []
    for op, cl in (("{", "}"), ("(", ")"), ("[", "]")):
        a, b = stripped.count(op), stripped.count(cl)
        tag = "OK" if a == b else "MISMATCH"
        if a != b:
            fails.append(f"{op}{cl} 不平衡 {a} vs {b}")
        print(f"  {op}{cl}: {a} vs {b}  {tag}")

    print()
    for sym in ("frameOmniBase64", "omniFrameCounter", "captureFaceFrame",
                "captureOmniFrame", "MobileChunk"):
        print(f"  {sym:22s} {src.count(sym)} 处")

    # 关键开关必须是 false（云端 AEC 是权威，浏览器不得抢先）
    for key in ("echoCancellation", "noiseSuppression", "autoGainControl"):
        m = re.search(rf"{key}:\s*(\w+)", src)
        val = m.group(1) if m else "NOT FOUND"
        ok = val == "false"
        if not ok:
            fails.append(f"{key} = {val}（应为 false）")
        print(f"  {key:22s} = {val}  {'OK' if ok else 'FAIL'}")

    # 分块必须是 100ms 量级而非 1 秒
    if "this.sampleRate / 10" not in src and "sampleRate / 10" not in src:
        fails.append("未找到 sampleRate/10（100ms 分块）")
        print("  chunkSize 计算        未找到 sampleRate/10  FAIL")
    else:
        print("  chunkSize 计算        sampleRate/10  OK")

    print()
    if fails:
        print(f"FAILED: {len(fails)} 项")
        for f in fails:
            print(f"  - {f}")
        return 1
    print("通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
