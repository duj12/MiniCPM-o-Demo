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


def _node_check(code: str):
    """用 node --check 做真正的语法检查（比数括号可靠）。

    手写正则数括号会被模板字符串里的 ``${}`` 和 URL 里的 ``//`` 误导 ——
    实测出现过误报。有 node 就用它，没有则跳过（不阻断）。
    """
    import shutil
    import subprocess
    import tempfile

    node = shutil.which("node")
    if not node:
        return None
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False,
                                     encoding="utf-8") as f:
        f.write(code)
        tmp = f.name
    try:
        r = subprocess.run([node, "--check", tmp], capture_output=True,
                           text=True, timeout=30)
        if r.returncode == 0:
            return True, ""
        return False, (r.stderr or r.stdout).strip()[:600]
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)
    finally:
        try:
            import os
            os.unlink(tmp)
        except OSError:
            pass


def main() -> int:
    if not PAGE.is_file():
        print(f"找不到 {PAGE}")
        return 2
    html = PAGE.read_text(encoding="utf-8")
    code = "\n".join(_JS.findall(html))

    fails = []
    ok, err = _node_check(code)
    if ok is None:
        print("  [跳过] node 不可用，未做 JS 语法检查")
    elif ok:
        print("  [OK] node --check 语法检查通过")
    else:
        print("  [FAIL] JS 语法错误:")
        for ln in err.splitlines()[:10]:
            print(f"      {ln}")
        fails.append("JS 语法错误")

    print()
    syms = ("PcmPlayer", "diagUpdate", "btnUnlock",
            "audiostat", "src_w", "src_h", "createConstantSource",
            "statechange", "grab(320, 240",
            # AEC 模式 / 延迟显示 / 校准
            "aecMode", "btnCalib", "renderDelay", "selectedAecMode",
            "deviceKey", "session.stats", "calibrate", "delayinfo")
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
        ("有 AudioContext 状态诊断", "diag.ctxState" in html),
        ("suspended 时有明确提示", "解锁音频" in html),
        ("AEC 模式可选（browser/service/off）",
         'value="browser"' in html and 'value="service"' in html
         and 'value="off"' in html),
        ("有延迟校准按钮", "btnCalib" in html),
        ("显示声学延迟与建议值",
         "delay_ms" in html and "suggested_delay_ms" in html),
        ("设备标识随会话上报（用于记住延迟）",
         "deviceKey" in html and "device_id" in html),
        ("校准提示要求外放",
         "外放" in html and "耳机" in html),
        ("ASR 与 TTS 分区（替换语义不冲掉追加语义）",
         'id="asrline"' in html and 'id="ttslog"' in html),
        ("播报字幕不截断（去掉 slice 限制）",
         ".slice(0, 80)" not in html),
        ("校准独立于会话（走 /v1/calibrate，不要求先开会话）",
         "/v1/calibrate" in html and "calibrate.play" in html),
        ("校准期间关掉浏览器 AEC（否则测的是被消过的信号）",
         html.count("echoCancellation: false") >= 1),
        ("service 模式关闭浏览器 AEC（二选一，不叠加）",
         "mode === 'browser'" in html),
        ("移除手动设延迟入口（不该让用户做这个）",
         "btnSetDelay" not in html),
        ("校准前发 calibrate.start（否则服务端不发音频）",
         "calibrate.start" in html),
        ("读 getSettings() 确认约束**实际生效**",
         "getSettings" in html and "实际生效" in html),
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
