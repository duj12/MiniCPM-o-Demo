#!/usr/bin/env python3
"""定位 HTML 内联 JS 的括号不平衡位置（辅助排查用）。"""
from __future__ import annotations

import re
import sys
from pathlib import Path

PAGE = Path(__file__).resolve().parents[2] / "static/orchestrator-test.html"

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
    src = PAGE.read_text(encoding="utf-8")
    blocks = re.findall(r"<script>(.*?)</script>", src, re.S)
    if not blocks:
        print("没有 <script> 块")
        return 2
    code = "\n".join(blocks)
    base = src[: src.find("<script>")].count("\n") + 1
    c = strip(code)

    depth = 0
    line = 1
    report = []
    # 记录「每行结束时的深度」，用于定位未闭合的块
    depth_at_eol = {}
    for ch in c:
        if ch == "\n":
            depth_at_eol[line] = depth
            line += 1
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth < 0:
                report.append((line, depth))
                depth = 0
    print(f"JS 起始于 HTML 第 {base} 行")
    print(f"最终括号深度: {depth}")
    if report:
        for ln, d in report[:5]:
            print(f"  多余的 '}}' 在 JS 第 {ln} 行（HTML 第 {base + ln - 1} 行）")

    if depth != 0:
        # 找出最后一次「深度回零」的行 —— 未闭合的块从那之后开始
        last_zero = 0
        for ln in sorted(depth_at_eol):
            if depth_at_eol[ln] == 0:
                last_zero = ln
        print(f"最后回到深度 0 的位置: JS 第 {last_zero} 行"
              f"（HTML 第 {base + last_zero - 1} 行）")
        # 打印该行之后到结尾各行的深度变化（只显示回升处）
        prev = 0
        for ln in sorted(depth_at_eol):
            if ln <= last_zero:
                continue
            d = depth_at_eol[ln]
            if d > prev:
                src_line = code.splitlines()[ln - 1] if ln - 1 < len(code.splitlines()) else ""
                print(f"   JS:{ln:4d} 深度 {prev}→{d}  | {src_line.strip()[:70]}")
            prev = d
    return 1 if (depth != 0 or report) else 0


if __name__ == "__main__":
    sys.exit(main())
