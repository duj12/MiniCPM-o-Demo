#!/usr/bin/env python3
"""验证页的内联 JS 结构检查（无 Node 时的替代）。

检查括号平衡 + 关键符号存在性 + 已知修复点是否落实。

    python orchestrator/tests/check_html.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

PAGE = Path(__file__).resolve().parents[2] / "static/orchestrator-test.html"

_JS = re.compile(r"<script>(.*?)</script>", re.S)
_BLOCK = re.compile(r"/\*.*?\*/", re.S)
_LINE = re.compile(r"//[^\n]*")
_TMPL = re.compile(r"`(?:\\.|[^`\\])*`", re.S)
_SQ = re.compile(r"'(?:\\.|[^'\\])*'")
_DQ = re.compile(r'"(?:\\.|[^"\\])*"')


def strip(s: str) -> str:
    s = _BLOCK.sub("", s)
    s = _LINE.sub("", s)
    s = _TMPL.sub('""', s)
    s = _SQ.sub('""', s)
    s = _DQ.sub('""', s)
    return s


def main() -> int:
    if not PAGE.is_file():
        print(f"找不到 {PAGE}")
        return 2
    html = PAGE.read_text(encoding="utf-8")
    code = "\n".join(_JS.findall(html))
    s = strip(code)

    fails = []
    for op, cl in (("{", "}"), ("(", ")"), ("[", "]")):
        a, b = s.count(op), s.count(cl)
        tag = "OK" if a == b else "MISMATCH"
        if a != b:
            fails.append(f"{op}{cl} 不平衡 {a} vs {b}")
        print(f"  {op}{cl}: {a} vs {b}  {tag}")

    print()
    syms = ("PcmPlayer", "diagUpdate", "btnUnlock", "btnTestTone",
            "audiostat", "src_w", "src_h", "createConstantSource",
            "statechange", "grab(320, 240")
    for sym in syms:
        print(f"  {sym:22s} {html.count(sym)}")

    print()
    # 已知修复点必须都在
    checks = [
        ("坐标用服务端下发的 src_w（不再写死 640）",
         "t.src_w" in html and "const G1_W = 640" not in html),
        ("播放器手动重采样到 ctx.sampleRate",
         "Math.abs(ctxRate - srcRate)" in html),
        ("有音频解锁按钮", "btnUnlock" in html and "audioCtx.resume()" in html),
        ("有 AudioContext 状态诊断", "diagState" in html or "diag.ctxState" in html),
        ("suspended 时有明确提示", "解锁音频" in html),
    ]
    for desc, ok in checks:
        print(f"  [{'OK' if ok else 'FAIL'}] {desc}")
        if not ok:
            fails.append(desc)

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
