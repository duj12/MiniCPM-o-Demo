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
# 会话客户端：armed 承诺与采集时间戳都在这里发出去，必须一起守
SESSION_JS = (Path(__file__).resolve().parents[2]
              / "static/duplex/lib/orchestrator-session.js")
CAPTURE_JS = (Path(__file__).resolve().parents[2]
              / "static/duplex/lib/capture-processor.js")

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
        # ⚠️ 必须返回**二元组**：调用方是按 `ok, err = _node_check(...)`
        #    解包的，裸 None 会炸成 TypeError 而不是"跳过"。106 上没装
        #    node，这条路径一直没被走到过（本地 Windows 有 node）。
        return None, "node 不可用"
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
    syms = ("PcmPlayer", "diagUpdate",
            "audiostat", "src_w", "src_h", "createConstantSource",
            "statechange", "grab(320, 240",
            # AEC 模式 / 延迟与锚点显示
            "aecMode", "renderDelay", "selectedAecMode",
            "deviceKey", "session.stats", "delayinfo",
            # 播放锚点与采集时间戳（本次修复的核心）
            "beginResponse", "start_ctx", "stop_ctx", "playback_anchor",
            "resampleChunk16k", "audioEpoch")
    for sym in syms:
        print(f"  {sym:22s} {html.count(sym)}")

    print()
    # 已知修复点必须都在
    checks = [
        ("坐标用服务端下发的 src_w（不再写死 640）",
         "t.src_w" in html and "const G1_W = 640" not in html),
        ("播放器手动重采样到 ctx.sampleRate",
         "Math.abs(ctxRate - srcRate)" in html),
        ("有 AudioContext 状态诊断", "diag.ctxState" in html),
        ("AEC 模式可选（browser/service/off）",
         'value="browser"' in html and 'value="service"' in html
         and 'value="off"' in html),
        ("显示声学延迟与落位锚点来源",
         "delay_ms" in html and "anchor_source" in html),
        ("设备标识随会话上报（用于记住延迟）",
         "deviceKey" in html and "device_id" in html),
        ("ASR 与 TTS 分区（替换语义不冲掉追加语义）",
         'id="asrline"' in html and 'id="ttslog"' in html),
        ("播报字幕不截断（去掉 slice 限制）",
         ".slice(0, 80)" not in html),
        ("service 模式关闭浏览器 AEC（二选一，不叠加）",
         "mode === 'browser'" in html),
        ("移除手动设延迟入口（不该让用户做这个）",
         "btnSetDelay" not in html),
        ("读 getSettings() 确认约束**实际生效**",
         "getSettings" in html and "实际生效" in html),
        # ---- 本次修复的核心契约 ----
        ("四个无效按钮已移除（解锁/校准/导出诊断/导出校准录音）",
         not any(f'btn{x}' in html
                 for x in ("Unlock", "Calib", "Export", "CalibWav"))),
        ("校准通道的前端代码已移除（服务端端点保留给离线工具）",
         "/v1/calibrate" not in html and "calibrate.start" not in html),
        ("声明 playback_anchor 能力位（否则服务端退回预测落位）",
         "'playback_anchor'" in html or '"playback_anchor"' in html),
        # ⚠️ 承诺点必须在**第一块音频到达时**，不能在 tts.start 时。
        #    真机踩过：tts.start 在合成之前发出，那时算的"现在 + 提前量"
        #    到音频真到时早已过去 340ms → 浏览器只能"马上播"，而参考轨仍按
        #    过期时刻落位 → 比实际回声早 340ms → 回声完全消不掉。
        ("起播承诺在第一块音频到达时计算（不能在 tts.start 时算）",
         "_armNow()" in html
         and "this._armed = this.ctx.currentTime + this._leadMs" in html
         and "beginResponse" in html
         and "this._armed = 0;" in html),
        # ⚠️ `_armNow()` 每次调用返回**同一个**缓存承诺值。若它出现在
        #    `at` 的表达式里被每块求值，所有块都会被排到同一时刻、
        #    叠在一起同时播 —— 听感是"一闪而过、只听到开头和结尾"。
        #    只有本句**首块**能取它，后续块必须走 nextAt 顺排。
        ("只有首块取承诺时刻，后续块按 nextAt 顺排（否则整段音频折叠同播）",
         "if (!this._firstAt) {" in html
         and "at = this._armNow() || (this.ctx.currentTime + 0.02);" in html
         and "const at = this._armNow() ||" not in html),
        ("采集块带上 AudioContext 时刻（服务端据此做精确换算）",
         "ev.data.t0" in html and "ctxTime" in html),
        # ⚠️ 断言的是**函数定义**，不是调用点。曾经这里只查调用点，
        #    结果定义在删旧 handler 时被误删 —— 调用点还在、断言照样通过，
        #    而页面在每个音频块上抛 ReferenceError：音频和视频**都发不出去**
        #    （两者在同一个 worklet 回调里），表现为"点击开始后既没识别也没
        #    人脸"。node --check 只查语法，查不出未定义引用，必须靠这条。
        ("采样率不是 16k 时先重采样再回传（否则服务端截断、时间轴错 3 倍）",
         "function resampleChunk16k(" in html and "const _rs16 = {" in html
         and "resampleChunk16k(ev.data.audio, audioCtx.sampleRate)" in html),
        ("AudioContext 代号随会话上报（换 context 时锚点作废）",
         "audioEpoch" in html and "epoch: audioEpoch" in html),
        ("AudioContext 挂起时有恢复路径（手势监听兜底）",
         "pointerdown" in html and "audioCtx.resume()" in html),
    ]
    for desc, ok in checks:
        print(f"  [{'OK' if ok else 'FAIL'}] {desc}")
        if not ok:
            fails.append(desc)

    # ---- 内联脚本里的"调用了但从未定义"的本地函数 ----
    #
    # ⚠️ 这条是被真机故障逼出来的：`resampleChunk16k` 的定义在删除旧按钮
    #    handler 时被一并删掉，调用点还在 —— 页面在每个音频块上抛
    #    ReferenceError，音频和视频（同一个 worklet 回调里）**都发不出去**。
    #    而当时的检查只断言调用点存在，照样通过；`node --check` 也只查语法，
    #    查不出未定义引用。这类 bug 只能靠"调用点 vs 定义点"的集合比对。
    print()
    print("== 内联脚本：调用了但未定义的本地函数 ==")
    stripped = strip(code)
    defined = set(re.findall(r"function\s+([A-Za-z_$][\w$]*)\s*\(", stripped))
    defined |= set(re.findall(r"(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=",
                              stripped))
    defined |= set(re.findall(r"class\s+([A-Za-z_$][\w$]*)\s*\{", stripped))
    # 类方法（缩进的 `name(args) {`）也算已定义 —— 否则 PcmPlayer 的
    # playBase64/stop 之类会被误报（它们是通过实例属性调用的）
    defined |= set(re.findall(r"^\s+([A-Za-z_$][\w$]*)\s*\([^)]*\)\s*\{",
                              stripped, re.M))
    # 函数**参数**也是合法的可调用标识符（回调里 `ok(t)` 这种），
    # 不算未定义 —— 否则每个接受回调的函数签名都会误报
    for params in re.findall(r"\(([^)]*)\)\s*(?:=>|\{)", stripped):
        for p in params.split(","):
            p = p.strip().lstrip(".").strip()
            if re.fullmatch(r"[A-Za-z_$][\w$]*", p):
                defined.add(p)
    defined |= set(re.findall(r"([A-Za-z_$][\w$]*)\s*=>", stripped))
    # ⚠️ 只收集**裸调用**（前面不是 `.`）—— `this.player.stop()` 里的
    #    `stop` 是成员方法，本地作用域里本来就不该有定义
    called = set(re.findall(r"(?<![\w$.])([A-Za-z_$][\w$]*)\s*\(", stripped))
    # 浏览器/JS 内置与全局，不算未定义
    builtins = {
        "if", "for", "while", "switch", "catch", "return", "typeof", "new",
        "function", "of", "in", "do", "else", "try", "delete", "void",
        "Math", "Number", "String", "Boolean", "Array", "Object", "JSON",
        "Float32Array", "Int16Array", "Uint8Array", "ArrayBuffer", "DataView",
        "Blob", "URL", "Date", "Promise", "Set", "Map", "Error", "parseInt",
        "parseFloat", "isNaN", "isFinite", "setTimeout", "clearTimeout",
        "setInterval", "clearInterval", "requestAnimationFrame", "atob",
        "btoa", "fetch", "console", "document", "window", "navigator",
        "location", "localStorage", "WebSocket", "AudioContext",
        "AudioWorkletNode", "FileReader", "Image", "Infinity", "NaN",
        "undefined", "null", "true", "false", "this", "super", "await",
        "async", "yield", "instanceof", "class", "extends", "get", "set",
        "static", "constructor", "globalThis", "EventSource", "Notification",
    }
    missing = sorted(c for c in called
                     if c not in defined and c not in builtins
                     and not c[0].isupper())     # 大写开头大概率是全局类
    if missing:
        print(f"  [FAIL] 以下函数被调用但没有定义：{missing}")
        for m in missing:
            print(f"      {m}()")
        fails.append(f"未定义的本地函数: {missing}")
    else:
        print(f"  [OK] 未发现未定义的本地函数"
              f"（定义 {len(defined)} 个 / 调用 {len(called)} 个）")

    # ---- 会话客户端 / 采集 worklet：锚点协议的发送端 ----
    # 这两处是"把时刻从浏览器问出来"的发送端，配套的服务端换算在
    # orchestrator/clock.py。任一侧被改回去，整条对齐就静默失效
    # （页面照常出声，只是回声再也消不掉）—— 所以必须一起守。
    print()
    print("== 锚点协议发送端（session 客户端 + 采集 worklet）==")
    js = SESSION_JS.read_text(encoding="utf-8") if SESSION_JS.is_file() else ""
    cap = CAPTURE_JS.read_text(encoding="utf-8") if CAPTURE_JS.is_file() else ""
    lib_checks = [
        ("session 客户端：tts.start 立即回 armed（承诺起播时刻）",
         "'armed'" in js and "start_ctx: at" in js),
        ("session 客户端：started/cancelled 带上实际时刻与 epoch",
         "_firstAt()" in js and "stop_ctx" in js and "epoch: this.epoch" in js),
        ("session 客户端：采集块带上 ctx_time + epoch",
         "ctx_time: chunk.ctxTime" in js),
        ("session 客户端：存下服务端下发的 lead_ms",
         "msg.lead_ms" in js),
        ("采集 worklet：每个块带上首采样的 ctx 时刻",
         "t0: t0" in cap and "this._t0 = t0 + this._chunkSize / sampleRate" in cap),
    ]
    for desc, ok in lib_checks:
        print(f"  [{'OK' if ok else 'FAIL'}] {desc}")
        if not ok:
            fails.append(desc)

    # ---- 服务端：必须先发音频、再等承诺 ----
    # 浏览器是在第一块音频到达时才算承诺时刻的，所以服务端要是先等承诺
    # 再发音频，就会死等到超时 —— 或者（更早的版本）让浏览器在 tts.start
    # 时承诺一个到音频到达时已经过期的时刻，导致参考轨早 340ms。
    print()
    print("== 服务端落位时序（先发音频，再等承诺）==")
    exe = (Path(__file__).resolve().parents[1] / "actions/executor.py")
    ex = exe.read_text(encoding="utf-8") if exe.is_file() else ""
    i_audio = ex.find("TtsAudio.from_int16")
    i_resolve = ex.find("_resolve_play_at(session, response_id, arm)")
    i_place = ex.find("session.ref_track.place(")
    seq_checks = [
        ("先发 TtsAudio，再等 armed 承诺",
         i_audio > 0 and i_resolve > i_audio),
        ("承诺拿到之后才 place() 参考轨",
         i_resolve > 0 and i_place > i_resolve),
        ("首块音频后让出事件循环（否则回执晚一个 RTT 才被处理）",
         "await asyncio.sleep(0)" in ex),
        ("等承诺的超时小于 lead+D 预算（否则参考轨写晚于回声）",
         "ARM_TIMEOUT_MS = 250" in ex),
    ]
    for desc, ok in seq_checks:
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
