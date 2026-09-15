#!/usr/bin/env python3
"""把验证页的内联 JS 抽出来，交给 node --check 做真正的语法检查。

比手写正则数括号可靠得多（正则容易被模板字符串里的 ``${}`` 和
URL 里的 ``//`` 误导）。

    python orchestrator/tests/extract_js.py /tmp/check.js
    node --check /tmp/check.js
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

PAGE = Path(__file__).resolve().parents[2] / "static/orchestrator-test.html"


def main() -> int:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("check.js")
    src = PAGE.read_text(encoding="utf-8")
    blocks = re.findall(r"<script>(.*?)</script>", src, re.S)
    # 只要内联脚本（跳过引外部文件的空标签）
    code = "\n".join(b for b in blocks if b.strip())
    out.write_text(code, encoding="utf-8")
    print(f"已抽出 {len(code)} 字节 → {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
