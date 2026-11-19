"""state head 的测量台架 —— 延迟/并发曲线 + SOP 逐槽位准确率。

两种用法互不干扰：

    # 1) 延迟与并发曲线（容器内跑，量的是整条链：uvicorn → httpx → flide）
    python -m state_head.bench_sop --still /app/assets/frame.jpg --repeat 50
    python -m state_head.bench_sop --still /app/assets/frame.jpg --repeat 200 \
        --concurrency 1,2,4,8,16

    # 2) SOP 逐槽位准确率（06–11 片段）
    python -m state_head.bench_sop --clips /data/.../assets/video/sop \
        --stage both --out /app/logs/sop_state.json

## 为什么默认打 ``/v1/state`` 而不是直接打 flide

验收标准是"端到端 P95 ≤ 500ms、8 路并发不排队"，那必须包含 uvicorn 的
JSON 编解码、缩图、以及组装 —— 只量 flide 会把这部分藏起来。要看 flide
本身的极限再追加 ``--direct``（它只发 ``/v1/systemone``，响应不组装）。

## 关于准确率的判定

期望值表（``SOP_EXPECT``）来自 SOP 片段本身，只判"模型该不该选到这一类"，
不判细粒度。命中率低于阈值**不代表服务坏了** —— V1 是零样本，这张表的作用
是给出"哪些槽位值得退回 Qwen3-Omni 文本、哪些可以直接用"的依据。
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import httpx

#: SOP 片段 → 期望（只写"必须满足"的槽位，见计划 §六.2）。
#: 键是片段文件名去扩展名后的**前缀**匹配（``06_1`` 命中 ``06_1_xxx.mkv``）。
SOP_EXPECT: Dict[str, Dict[str, Any]] = {
    "06_1": {"attn.orientation": {"back_to_camera"}},
    "06_2": {"attn.gaze": {"at_screen", "at_camera"},
             "beh.activity": {"using_phone"}},
    "07_1": {"_argmax_action": "thumbs_up"},
    "07_2": {"_argmax_action": "shake_head", "_min_count": 2},
    "08_1": {"emo.visual": {"dissatisfied", "impatient", "confused",
                            "angry", "sad"}},
    "08_2": {"emo.visual": {"dissatisfied", "impatient", "confused",
                            "angry", "sad"}},
    "10_2": {"beh.motion": {"moving_away"}, "_min_duration_s": 2},
    "11": {"rel.person_count": {2},
           "rel.interaction": {"talking", "facing_each_other"},
           "rel.target": {"another_person"}},
}

MIME = ".jpg"


# --------------------------------------------------------------------------- #
#  输入：单帧 / 片段抽帧
# --------------------------------------------------------------------------- #

def _b64_file(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


def extract_frames(clip: Path, fps: float = 1.0, max_frames: int = 8) -> List[str]:
    """``.mkv/.mp4`` → 每秒一帧的 base64 列表（需要宿主机/容器内有 ffmpeg）。"""
    if shutil.which("ffmpeg") is None:
        raise SystemExit("需要 ffmpeg 抽帧（或先用 ffmpeg 手动抽好，"
                         "把 --clips 指到帧目录）")
    out_dir = Path(tempfile.mkdtemp(prefix="sop_"))
    try:
        subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(clip),
             "-vf", f"fps={fps}", "-frames:v", str(max_frames),
             "-q:v", "2", str(out_dir / f"%03d{MIME}")],
            check=True)
        frames = sorted(out_dir.glob(f"*{MIME}"))
        return [_b64_file(p) for p in frames]
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)


def clip_key(name: str) -> Optional[str]:
    for k in SOP_EXPECT:
        if name.startswith(k):
            return k
    return None


# --------------------------------------------------------------------------- #
#  请求
# --------------------------------------------------------------------------- #

async def post_state(client: httpx.AsyncClient, base: str, frames: List[str],
                     stage: str, session_id: str = "") -> Dict[str, Any]:
    resp = await client.post(
        f"{base}/v1/state",
        json={"frames_b64": frames, "stage": stage, "session_id": session_id})
    resp.raise_for_status()
    return resp.json()


async def post_direct(client: httpx.AsyncClient, flide: str,
                      frames: List[str]) -> Dict[str, Any]:
    from .backends import build_media
    from .questions import build_questions

    st, media = build_media(frames)
    resp = await client.post(f"{flide}/v1/systemone",
                             json={"state": st, "questions":
                                   build_questions("turn"), "media": media})
    resp.raise_for_status()
    return resp.json()


# --------------------------------------------------------------------------- #
#  延迟 / 并发
# --------------------------------------------------------------------------- #

def _pct(xs: Sequence[float], q: float) -> float:
    if not xs:
        return float("nan")
    s = sorted(xs)
    i = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
    return s[i]


async def bench_latency(base: str, frames: List[str], repeat: int,
                        concurrency: int, stage: str) -> Dict[str, Any]:
    """``repeat`` 次请求，``concurrency`` 路并发；返回时延与吞吐。"""
    async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as client:
        lat: List[float] = []
        errs = 0
        sem = asyncio.Semaphore(concurrency)

        async def one(i: int) -> None:
            nonlocal errs
            async with sem:
                t0 = time.monotonic()
                try:
                    await post_state(client, base, frames, stage, f"bench{i}")
                except Exception as exc:  # noqa: BLE001
                    errs += 1
                    if errs <= 3:
                        print(f"  ! {exc}", file=sys.stderr)
                    return
                lat.append((time.monotonic() - t0) * 1000.0)

        t_start = time.monotonic()
        await asyncio.gather(*(one(i) for i in range(repeat)))
        wall = time.monotonic() - t_start
    return {
        "concurrency": concurrency, "n": repeat, "errors": errs,
        "p50_ms": round(_pct(lat, 0.50), 1),
        "p95_ms": round(_pct(lat, 0.95), 1),
        "max_ms": round(max(lat), 1) if lat else None,
        "qps": round(len(lat) / wall, 2) if wall > 0 else 0.0,
    }


# --------------------------------------------------------------------------- #
#  SOP 准确率
# --------------------------------------------------------------------------- #

def _slot_of(state: Dict[str, Any], qid: str) -> Any:
    """``attn.orientation`` → ``state["ATTENTION"]["orientation"]["label"]``。"""
    group, _, field = qid.partition(".")
    key = {"attn": "ATTENTION", "beh": "BEHAVIOR", "emo": "EMOTION",
           "rel": "RELATION", "env": "ENV", "pos": "POSITION"}.get(group)
    if key is None:
        return None
    node = state.get(key, {})
    if qid == "emo.visual":
        node = node.get("visual", {})
        return node.get("label") if isinstance(node, dict) else None
    if not isinstance(node, dict):
        return None
    v = node.get(field)
    if isinstance(v, dict):
        return v.get("label")
    return v


def _argmax_action(state: Dict[str, Any]) -> Optional[str]:
    events = (state.get("ACTION") or {}).get("events") or []
    if not events:
        return None
    return max(events, key=lambda e: e.get("confidence") or 0.0).get("type")


def check_expected(state: Dict[str, Any], expect: Dict[str, Any]) -> List[Tuple[str, bool]]:
    """返回 ``[(检查项, 是否通过)]``。"""
    out: List[Tuple[str, bool]] = []
    for k, want in expect.items():
        if k == "_argmax_action":
            out.append((k, _argmax_action(state) == want))
        elif k == "_min_count":
            ev = (state.get("ACTION") or {}).get("events") or []
            got = max([e.get("count") or 0 for e in ev], default=0)
            out.append((k, got >= int(want)))
        elif k == "_min_duration_s":
            got = (state.get("BEHAVIOR") or {}).get("duration_s") or 0
            out.append((k, got >= int(want)))
        else:
            out.append((k, _slot_of(state, k) in want))
    return out


async def bench_sop(clips: Path, base: str, stage_mode: str,
                    out_path: Optional[Path]) -> int:
    files = sorted([p for p in clips.rglob("*")
                    if p.suffix.lower() in (".mkv", ".mp4", ".mov")])
    if not files:
        print(f"没在 {clips} 找到片段", file=sys.stderr)
        return 2

    stages = ["full", "turn"] if stage_mode == "both" else [stage_mode]
    results: List[Dict[str, Any]] = []
    async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as client:
        for clip in files:
            key = clip_key(clip.name)
            expect = SOP_EXPECT.get(key or "", {})
            try:
                frames = extract_frames(clip)
            except subprocess.CalledProcessError as exc:
                print(f"  ! 抽帧失败 {clip.name}: {exc}", file=sys.stderr)
                continue
            if not frames:
                continue
            for stage in stages:
                t0 = time.monotonic()
                try:
                    r = await post_state(client, base, frames[-2:], stage,
                                         session_id=f"sop-{clip.stem}-{stage}")
                except Exception as exc:  # noqa: BLE001
                    print(f"  ! {clip.name} {stage}: {exc}", file=sys.stderr)
                    continue
                dt = (time.monotonic() - t0) * 1000.0
                checks = check_expected(r.get("state", {}), expect) if expect else []
                results.append({
                    "clip": clip.name, "key": key, "stage": stage,
                    "latency_ms": round(dt, 1), "n_questions": r.get("n_questions"),
                    "state": r.get("state"), "checks": checks,
                    "passed": sum(1 for _, ok in checks if ok),
                    "total": len(checks),
                })
                mark = "—" if not checks else f"{sum(1 for _, ok in checks if ok)}/{len(checks)}"
                print(f"{clip.name:<28} {stage:<5} {dt:7.1f}ms  {mark}")
                for name, ok in checks:
                    if not ok:
                        print(f"    ✗ {name}")

    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2),
                            encoding="utf-8")
        print(f"\n明细已写入 {out_path}")

    scored = [r for r in results if r["total"]]
    if scored:
        p = sum(r["passed"] for r in scored)
        t = sum(r["total"] for r in scored)
        print(f"\nSOP 命中 {p}/{t} = {p / t:.1%}")
    lats = [r["latency_ms"] for r in results]
    if lats:
        print(f"延迟 p50={_pct(lats, .5):.0f}ms p95={_pct(lats, .95):.0f}ms "
              f"max={max(lats):.0f}ms（{len(lats)} 次）")
    return 0 if not scored or p == t else 1


# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="state head 基准")
    ap.add_argument("--url", default=os.environ.get(
        "STATE_HEAD_URL", "http://127.0.0.1:22520"),
        help="state_head 地址（默认 http://127.0.0.1:22520）")
    ap.add_argument("--direct", metavar="FLIDE_URL",
        help="改打 flide 的 /v1/systemone（不组装，只量模型）")
    ap.add_argument("--still", type=Path, help="延迟基准用的单帧图片")
    ap.add_argument("--repeat", type=int, default=50)
    ap.add_argument("--concurrency", default="1",
        help="逗号分隔，如 1,2,4,8,16")
    ap.add_argument("--stage", default="turn", choices=("turn", "full"))
    ap.add_argument("--clips", type=Path, help="SOP 片段目录（.mkv/.mp4）")
    ap.add_argument("--stage-mode", default="both", choices=("turn", "full", "both"))
    ap.add_argument("--out", type=Path, help="SOP 明细 JSON 输出路径")
    args = ap.parse_args(argv)

    if args.clips:
        return asyncio.run(bench_sop(args.clips, args.url, args.stage_mode, args.out))

    if not args.still:
        ap.error("需要 --still <图片> 或 --clips <目录>")
    frames = [_b64_file(args.still)]

    if args.direct:
        async def _direct() -> None:
            async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as c:
                for _ in range(args.repeat):
                    await post_direct(c, args.direct, frames)
        t0 = time.monotonic()
        asyncio.run(_direct())
        print(f"direct {args.repeat} 次：{(time.monotonic() - t0) * 1000 / args.repeat:.1f} ms/次")
        return 0

    conc = [int(x) for x in str(args.concurrency).split(",") if x.strip()]
    rows = []
    for c in conc:
        row = asyncio.run(bench_latency(args.url, frames, args.repeat, c, args.stage))
        rows.append(row)
        print(f"并发 {c:>2}  请求 {row['n']:>4}  错误 {row['errors']:>2}  "
              f"p50 {row['p50_ms']:>7}ms  p95 {row['p95_ms']:>7}ms  "
              f"max {row['max_ms']}ms  {row['qps']} QPS")
    worst = max((r["p95_ms"] for r in rows if r["p95_ms"] == r["p95_ms"]), default=0)
    print(f"\n最差 p95 = {worst:.0f}ms（预算 500ms）")
    return 0 if worst <= 500 else 1


if __name__ == "__main__":
    raise SystemExit(main())
