"""同一段视频、同一张帧，逐秒对比两条视觉状态链。

    JEV  —— ``POST /v1/state``（``stage=full``，全套 16 问，出全部枚举槽位）
    OMNI —— Qwen3-Omni 实时 WS，每拍一轮描述文本（默认用**线上**的
            desc-only 两行提示词，即 ``orchestrator.omni.describe`` 里那份）

两边看的是**同一张帧**：omni 的 ``recent_frames_b64`` 返回的是已经进了 KV
的帧，原样再发给 state head —— 所以延迟差里不含抽帧/缩放差异。

用法（在仓库根目录、装了 orchestrator 依赖的 python 下）：

    # 默认：线上两行描述 vs Jev，20 拍
    python -m orchestrator.tools.bench_compare /data/.../s-xxx-raw.mkv

    # 老行为对照：Omni 吐全量 8 键，看当年的字数和耗时
    python -m orchestrator.tools.bench_compare clip.mkv --variant full

    # 只看 Jev、不跑 omni（--variant none），或把明细落盘
    python -m orchestrator.tools.bench_compare clip.mkv --variant none --out /tmp/cmp.jsonl

## 读数要注意

* 两侧都是**整轮时间**：含送帧（omni 还要送 1s 音频块）+ 等 JSON/解码完成。
  Jev 的数字是 state head 自报的 ``latency_ms``，不含网络。
* ``max_new_tokens`` 在 omni 的这条路径上**实测不一定生效**（16/32/48 都不
  截断），真正控制输出长度的是提示词里的字数要求。
* 提示词模板里**不能出现符号**（``——`` 会被当成字段内分隔符照抄），示例
  也**不能贴近常见场景**（否则模型一字不差照抄示例）。这两条都是 2026-11-20
  实测踩出来的，详见 ``orchestrator/omni/describe.py`` 的注释。
"""
from __future__ import annotations

import argparse
import asyncio
import glob
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

import numpy as np

#: 仓库根（``orchestrator/tools/`` 往上三层）。被 import 或被 -m 调用都成立。
REPO = str(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from orchestrator.omni.client import OmniClient            # noqa: E402
from orchestrator.omni.describe import (                    # noqa: E402
    DESC_ONLY_INSTRUCTION, DESC_ONLY_SYSTEM_PROMPT, STATE_SYSTEM_PROMPT,
    TURN_INSTRUCTION, parse_state)

STATE_URL = os.environ.get("STATE_URL", "http://127.0.0.1:22520")
OMNI_URL = os.environ.get(
    "OMNI_URL", "wss://127.0.0.1:8006/v1/realtime?mode=video")
SR = 16000

#: 各 variant：system prompt / 每轮注入指令 / max_new_tokens。
VARIANTS = {
    # 线上形态：文本只出 FEATURE / ENV 两行关键词
    "desc": (DESC_ONLY_SYSTEM_PROMPT, DESC_ONLY_INSTRUCTION, 32),
    # 老形态：文本吐全量 8 键（用来对照字数和耗时）
    "full": (STATE_SYSTEM_PROMPT, TURN_INSTRUCTION, None),
}


def extract_frames(video: str, outdir: str, start: float, ticks: int,
                   px: int = 640, fps: float = 1.0) -> list[str]:
    subprocess.run([
        "ffmpeg", "-loglevel", "error", "-y",
        "-ss", str(start), "-i", video,
        "-vf", f"fps={fps},scale='min({px},iw)':-2",
        "-frames:v", str(ticks),
        os.path.join(outdir, "f%03d.jpg"),
    ], check=True)
    return sorted(glob.glob(os.path.join(outdir, "f*.jpg")))


async def post_state(frames_b64: list[str], timeout_s: float = 12.0) -> dict:
    body = json.dumps({"frames_b64": frames_b64, "stage": "full",
                       "slots": []}).encode()

    def _post() -> dict:
        req = urllib.request.Request(
            STATE_URL + "/v1/state", data=body,
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout_s) as r:
            return json.loads(r.read())

    last: Exception | None = None
    for _ in range(3):
        try:
            return await asyncio.to_thread(_post)
        except (urllib.error.HTTPError, urllib.error.URLError, OSError) as exc:
            last = exc
            await asyncio.sleep(1.0)
    raise RuntimeError(f"state head 三次都失败: {last}")


class _Collector:
    """收集 omni 这一轮回复：text delta 拼接 + response.done 的全文。"""

    def __init__(self) -> None:
        self._delta = ""
        self.last_done = ""
        self.done_evt = asyncio.Event()

    def on_event(self, ev: dict) -> None:
        t = ev.get("type")
        if t == "response.output.delta" and ev.get("kind") == "text":
            self._delta += ev.get("text") or ""
        elif t == "response.done":
            text = (ev.get("text") or self._delta or "").strip()
            self._delta = ""
            self.last_done = text
            self.done_evt.set()

    def begin(self) -> None:
        self._delta = ""
        self.last_done = ""
        self.done_evt.clear()


def _fmt_e2(state: dict) -> str:
    """把 E2 压成一行：只显示有结论的槽位。"""
    out = []
    for key in ("ATTENTION", "ACTION", "EMOTION", "BEHAVIOR", "RELATION",
                "FEATURE", "ENV", "POSITION"):
        node = state.get(key)
        if node is None:
            out.append(f"{key}=null")
            continue
        if not isinstance(node, dict):
            continue
        bits = []
        for k, v in node.items():
            if isinstance(v, dict) and "label" in v:
                bits.append(f"{k}={v.get('label')}"
                            + (f"({v['confidence']:.2f})"
                               if v.get("confidence") is not None else ""))
            elif isinstance(v, list):
                if v:
                    bits.append(f"{k}=" + ",".join(
                        str(x.get("type", x)) if isinstance(x, dict) else str(x)
                        for x in v))
            elif v is not None:
                bits.append(f"{k}={v}")
        out.append(f"{key}{{{'; '.join(bits)}}}")
    return " ".join(out)


def _stats(vals: list[int]) -> str:
    vals = sorted(v for v in vals if v is not None)
    if not vals:
        return "n/a"
    n = len(vals)
    med = vals[n // 2] if n % 2 else (vals[n // 2 - 1] + vals[n // 2]) // 2
    p95 = vals[min(n - 1, int(n * 0.95))]
    return f"min {vals[0]} / 中位 {med} / p95 {p95} / max {vals[-1]} ms"


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("video", help="含视频的录像（-raw.mkv）")
    ap.add_argument("--start", type=float, default=2.0, help="起始秒")
    ap.add_argument("--ticks", type=int, default=20, help="拍数")
    ap.add_argument("--px", type=int, default=640, help="送帧长边像素")
    ap.add_argument("--fps", type=float, default=1.0, help="抽帧帧率")
    ap.add_argument("--variant", choices=("desc", "full", "none"),
                    default="desc",
                    help="desc = 线上两行；full = 老 8 键；none = 只跑 Jev")
    ap.add_argument("--max-tokens", type=int, default=0,
                    help="覆盖该 variant 的 max_new_tokens（0 = 用默认）")
    ap.add_argument("--audio-ms", type=int, default=1000,
                    help="每拍随帧送的音频长度（<1000 绕过 push_audio 的 1s 聚合）")
    ap.add_argument("--out", default="", help="JSONL 明细输出路径")
    args = ap.parse_args()

    if not os.path.isfile(args.video):
        print(f"视频不存在: {args.video}", file=sys.stderr)
        return 2

    tmpdir = tempfile.mkdtemp(prefix="state_cmp_")
    frames = extract_frames(args.video, tmpdir, args.start, args.ticks,
                            args.px, args.fps)
    print(f"# {os.path.basename(args.video)} 从 {args.start:g}s 起，"
          f"{len(frames)} 拍（{args.fps:g}fps）", flush=True)
    if not frames:
        print("没有抽到帧", file=sys.stderr)
        return 2

    run_omni = args.variant != "none"
    col = _Collector()
    omni = None
    recv = None
    instruction = ""
    max_tokens = None
    if run_omni:
        system_prompt, instruction, max_tokens = VARIANTS[args.variant]
        if args.max_tokens > 0:
            max_tokens = args.max_tokens
        # ⚠️ 必须用 asr 触发模式（编排侧的默认值）：``turn_decision="model"``
        #    才能让 ``trigger_reply(force_listen=False)`` 真的触发解码。
        #    "turnsense" 下服务端等 VAD 判定话轮，纯静音永远等不到 response.done。
        omni = OmniClient(OMNI_URL, system_prompt=system_prompt,
                          on_event=col.on_event, verify_ssl=False,
                          turn_trigger="asr")
        await omni.connect()
        recv = asyncio.create_task(omni.recv_loop())

    silence = np.zeros(SR, dtype=np.float32)
    rows = []
    try:
        for i, path in enumerate(frames, 1):
            with open(path, "rb") as fh:
                jpeg = fh.read()
            col.begin()
            t0 = time.monotonic()
            kv_frames = [jpeg]
            if omni is not None:
                # 帧随音频块一起进 KV（与编排里 offer_frame + 音频块同一机制）
                omni.offer_frame(jpeg)
                if args.audio_ms >= 1000:
                    await omni.push_audio(silence)
                else:
                    await omni._send(np.zeros(
                        SR * args.audio_ms // 1000, dtype=np.float32))
                kv_frames = omni.recent_frames_b64(2) or [jpeg]

            st_task = asyncio.create_task(post_state(kv_frames))

            omni_text = ""
            if omni is not None:
                ok = await omni.trigger_reply(instruction,
                                              max_new_tokens=max_tokens)
                if ok:
                    try:
                        await asyncio.wait_for(col.done_evt.wait(), timeout=30)
                        omni_text = col.last_done
                    except asyncio.TimeoutError:
                        omni_text = "(omni 30s 未返回)"
                else:
                    omni_text = "(omni 触发失败)"
            omni_ms = int((time.monotonic() - t0) * 1000) if run_omni else None

            try:
                st = await asyncio.wait_for(st_task, timeout=30)
                e2 = st.get("state") or {}
                ms = st.get("latency_ms")
                st_n = st.get("n_questions")
            except Exception as exc:  # noqa: BLE001
                e2, ms, st_n = {}, None, None
                print(f"  !! state 失败: {exc}", file=sys.stderr)

            parsed = parse_state(omni_text) if omni_text else {}
            rows.append({"t": float(i - 1), "jev": e2, "jev_ms": ms,
                         "jev_nq": st_n, "omni_text": omni_text,
                         "omni_parsed": parsed, "omni_ms": omni_ms})

            print(f"\n[{i:02d}s] JEV  {ms}ms ({st_n} 问)  {_fmt_e2(e2)}",
                  flush=True)
            if run_omni:
                print(f"       OMNI {omni_ms}ms  {len(omni_text)}字  "
                      f"{omni_text.replace(chr(10), ' | ')}", flush=True)
            await asyncio.sleep(0.3)
    finally:
        if omni is not None:
            # 只 cancel 收包循环，**不要**手动置 ``closed = True`` ——
            # ``OmniClient.close()`` 见到 closed 会直接 return，WS 就不关了，
            # 退出时 asyncio 会报 "Fatal error on SSL transport / Event loop
            # is closed"。
            recv.cancel()
            try:
                await asyncio.gather(recv, return_exceptions=True)
                await omni.close()
                await asyncio.sleep(0.1)   # 让关闭帧发出去
            except Exception:  # noqa: BLE001
                pass

    print(f"\n=== 延迟汇总（{len(rows)} 拍，整轮时间）===", flush=True)
    print(f"  JEV   {_stats([r['jev_ms'] for r in rows])}", flush=True)
    if run_omni:
        print(f"  OMNI  {_stats([r['omni_ms'] for r in rows])}", flush=True)
        chars = [len(r["omni_text"]) for r in rows]
        if chars:
            print(f"  OMNI 文本长度  {min(chars)}~{max(chars)} 字"
                  f"（中位 {sorted(chars)[len(chars) // 2]}）", flush=True)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"\n# 明细已写入 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
