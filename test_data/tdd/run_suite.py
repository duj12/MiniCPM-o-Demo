#!/usr/bin/env python3
"""Replay every status=runnable clip and score its observations.

Blocked clips stay out of the run. agent_gate does not fail a clip.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CASES = ROOT / "test_data" / "tdd" / "cases.json"
PY = "/root/miniconda3/envs/orch105/bin/python"
#: Shadow orchestrators run on another port (run_orch_105.sh ORCH_PORT).
PORT = os.environ.get("ORCH_REPLAY_PORT", "8100")
OUT = Path(os.environ.get("ORCH_TDD_OUT", "/tmp/sop_tdd_v2"))

sys.path.insert(0, str(ROOT / "test_data" / "tdd"))
from score import blocked_cases, load_cases, runnable_cases, score_paths  # noqa: E402


def log(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with (OUT / "progress.txt").open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def replay(clip: dict) -> tuple[Path, Path, int]:
    cid = clip["id"]
    events = OUT / f"{cid}.events.jsonl"
    rlog = OUT / f"{cid}.replay.log"
    video = ROOT / "test_data" / clip["video"]
    cmd = [
        PY, "-u", str(ROOT / "orchestrator_replay.py"),
        "--video", str(video),
        "--host", "127.0.0.1", "--port", PORT,
        "--ic-events-out", str(events),
        "--no-show", "--mute",
        "--face-size", "1280x720",
        "--drain-s", "40",
        "--wait-reply-s", "20",
    ]
    t0 = time.monotonic()
    try:
        with rlog.open("w", encoding="utf-8") as fo:
            proc = subprocess.run(
                cmd, cwd=str(ROOT), stdout=fo, stderr=subprocess.STDOUT, timeout=700,
            )
        code = proc.returncode
    except subprocess.TimeoutExpired:
        code = 124
    log(
        f"REPLAY {cid} exit={code} wall={time.monotonic() - t0:.0f}s "
        f"bytes={events.stat().st_size if events.is_file() else 0}"
    )
    return events, rlog, code


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    cases = load_cases(CASES)
    wanted = set(sys.argv[1:])
    clips = runnable_cases(cases)
    if wanted:
        clips = [c for c in clips if c["id"] in wanted or c["suite"] in wanted]
    blocked = blocked_cases(cases)
    (OUT / "blocked.json").write_text(
        json.dumps(
            [
                {"id": c["id"], "suite": c["suite"], "blocked_reason": c.get("blocked_reason")}
                for c in blocked
            ],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    if not clips:
        log("no runnable clips")
        return 2
    summary = OUT / "summary.jsonl"
    if not wanted:
        summary.write_text("", encoding="utf-8")
        (OUT / "progress.txt").write_text("", encoding="utf-8")
    log(f"START runnable={len(clips)} blocked={len(blocked)}")
    fails = 0
    counts: dict[str, dict[str, int]] = {}
    for i, clip in enumerate(clips, 1):
        video = ROOT / "test_data" / (clip.get("video") or "")
        log(f"RUN {i}/{len(clips)} {clip['id']}")
        if not video.is_file():
            report = {
                "id": clip["id"],
                "suite": clip["suite"],
                "verdict": "INCONCLUSIVE",
                "observations": [],
                "reason": f"missing video {video}",
            }
        else:
            events, rlog, code = replay(clip)
            report = score_paths(clip, events, rlog)
            report["replay_exit"] = code
        obs_path = OUT / f"{clip['id']}.observation.json"
        obs_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        report["observation_json"] = str(obs_path)
        verdict = report["verdict"]
        bucket = counts.setdefault(clip["suite"], {"PASS": 0, "FAIL": 0, "INCONCLUSIVE": 0})
        bucket[verdict] = bucket.get(verdict, 0) + 1
        if verdict != "PASS":
            fails += 1
        failed_obs = [o for o in report.get("observations") or [] if not o.get("ok")]
        log(f"SCORE {clip['id']} {verdict} failed={failed_obs}")
        with summary.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(report, ensure_ascii=False) + "\n")
    rollup = {
        "runnable": {suite: sum(v.values()) for suite, v in counts.items()},
        "verdicts": counts,
        "fails": fails,
        "blocked": [
            {"id": c["id"], "suite": c["suite"], "blocked_reason": c.get("blocked_reason")}
            for c in blocked
        ],
    }
    (OUT / "rollup.json").write_text(
        json.dumps(rollup, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log(f"DONE fails={fails} rollup={rollup['verdicts']}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
