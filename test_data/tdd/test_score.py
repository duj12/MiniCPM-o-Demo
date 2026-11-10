"""Scorer bans. These stay red when a loose gate would go green."""
from __future__ import annotations

import json
from pathlib import Path

from score import load_cases, resume_closes_pause_at_or_after, runnable_cases, score_clip

CASES = Path(__file__).with_name("cases.json")


def _answer(t_ms: int, text: str, emotion: str = "", role: str | None = None) -> dict:
    event = {
        "t_ms": t_ms,
        "action": "ANSWER",
        "speech": {"transcript": text, "emotion": emotion},
        "probes": {},
        "session": {},
    }
    if role is not None:
        event["probes"]["utterance_role"] = role
    return event


def _yield(t_ms: int, text: str, barge_class: str) -> dict:
    return {
        "t_ms": t_ms,
        "action": "YIELD",
        "sop": "07",
        "speech": {"transcript": text},
        "probes": {"barge_class": barge_class},
        "session": {"mode": "LISTENING", "user_spoke_since_greet": True},
    }


def test_runnable_set_keeps_a7_split_and_blocked_out():
    cases = load_cases(CASES)
    runnable = runnable_cases(cases)
    ids = [c["id"] for c in runnable]
    assert ids.count("v1-A07") == 1
    for n in range(2, 9):
        assert f"v1-A07-{n}" in ids
    assert "v1-A01" not in ids
    assert "v1-A02" not in ids
    assert "supp-75" not in ids
    assert "v2-05-1" in ids
    assert "v2-05-2" in ids
    by_suite = {}
    for clip in runnable:
        by_suite[clip["suite"]] = by_suite.get(clip["suite"], 0) + 1
    assert by_suite == {"v2": 32, "v1_new": 21, "supplement": 2}


def test_asr_neighbor_names_aigc():
    clip = {
        "id": "v2-03-1",
        "observations": [{"kind": "answer_target_contains", "any_of": ["AIGC", "aigc"]}],
    }
    assert score_clip(clip, [_answer(11300, "你好,  AIG是什么呀?")], "")["verdict"] == "PASS"
    assert score_clip(clip, [_answer(8700, "你好,  AITC是什么呀?")], "")["verdict"] == "PASS"
    assert score_clip(clip, [_answer(1000, "哼哼")], "")["verdict"] == "FAIL"
    assert score_clip(clip, [_answer(1000, "今天天气怎么样")], "")["verdict"] == "FAIL"
    assert score_clip(clip, [_answer(1000, "AI 是什么")], "")["verdict"] == "FAIL"


def test_one_edit_names_a_chinese_target():
    clip = {
        "id": "room",
        "observations": [{"kind": "answer_target_regex", "pattern": "会议室"}],
    }
    assert score_clip(clip, [_answer(1000, "帮我订个会议是")], "")["verdict"] == "PASS"
    assert score_clip(clip, [_answer(1000, "哼哼")], "")["verdict"] == "FAIL"


def test_any_answer_does_not_pass_target():
    clip = {
        "id": "x",
        "observations": [{"kind": "answer_target_regex", "pattern": "会议室"}],
    }
    report = score_clip(clip, [_answer(1000, "哼哼")], "")
    assert report["verdict"] == "FAIL"
    assert report["observations"][0]["ok"] is False


def test_hmm_does_not_pass_schedule():
    clip = {
        "id": "v2-11-2",
        "observations": [
            {"kind": "answer_target_regex", "pattern": "下班|日程"},
            {"kind": "no_answer_transcript_regex", "pattern": "^哼+"},
        ],
    }
    report = score_clip(clip, [_answer(1000, "哼哼")], "")
    assert report["verdict"] == "FAIL"
    assert [row["ok"] for row in report["observations"]] == [False, False]


def test_historical_resume_does_not_close_a_later_soft_pause():
    log = "\n".join([
        "[3000ms ASR] 早",
        "[tts.pause greet1] 停在 1.00s",
        "[tts.resume greet1] 从 1.00s 接着播",
        "[22000ms ASR] 嗯那",
    ])
    assert resume_closes_pause_at_or_after(log, 22000) is False
    log_ok = log + "\n[tts.pause later] 停在 2.00s\n[tts.resume later] 从 2.00s 接着播"
    assert resume_closes_pause_at_or_after(log_ok, 22000) is True


def test_false_barge_ignores_greet_resume_and_false_barge_yield():
    clip = {
        "id": "v2-04-1",
        "observations": [
            {"kind": "answer_target_contains", "any_of": ["展厅"]},
            {"kind": "false_barge_soft_resume_or_ignore"},
        ],
    }
    events = [
        _answer(19000, "给我介绍一下你们的展厅吧。"),
        {
            "t_ms": 22200,
            "action": "LISTEN",
            "speech": {"transcript": "嗯那。"},
            "session": {"barge_phase": "soft", "barge_pause_reason": "barge"},
            "probes": {},
        },
    ]
    log = "\n".join([
        "[4000ms face]",
        "[tts.pause greet] 停在 1.00s",
        "[tts.resume greet] 从 1.00s 接着播",
    ])
    report = score_clip(clip, events, log)
    assert report["observations"][1]["ok"] is False
    events.append(_yield(23000, "嗯那。", "false_barge"))
    report = score_clip(clip, events, log)
    assert report["observations"][1]["ok"] is False


def test_greet_process_yield_is_not_other_place_yield():
    clip = {
        "id": "v2-04-1",
        "observations": [
            {"kind": "answer_target_contains", "any_of": ["展厅"]},
            {"kind": "no_hard_yield_on_false_barge_script"},
        ],
    }
    greeting = {
        "t_ms": 6400,
        "action": "HOLD",
        "speech": {},
        "probes": {},
        "session": {"mode": "GREETING", "playback_active": True},
    }
    cutting = _yield(6600, "给", "false_barge")
    answer = _answer(9000, "给我介绍一下你们的展厅吧。")
    answer["session"] = {"mode": "SPEAKING"}
    report = score_clip(clip, [greeting, cutting, answer], "")
    assert report["observations"][1]["ok"] is True
    later = _yield(23000, "嗯那。", "false_barge")
    later["session"] = {"mode": "SPEAKING", "playback_active": True}
    report = score_clip(clip, [greeting, cutting, answer, later], "")
    assert report["observations"][1]["ok"] is False


def test_backchannel_yield_during_greet_does_not_fail():
    clip = {
        "id": "v2-03-1",
        "observations": [{"kind": "no_hard_yield_on_backchannel"}],
    }
    greeting = {
        "t_ms": 4000,
        "action": "HOLD",
        "speech": {},
        "probes": {},
        "session": {"mode": "GREETING"},
    }
    greet_yield = _yield(4200, "嗯嗯", "backchannel")
    report = score_clip(clip, [greeting, greet_yield], "")
    assert report["observations"][0]["ok"] is True
    later = _yield(20000, "嗯嗯", "backchannel")
    later["session"] = {"mode": "SPEAKING"}
    answer = _answer(9000, "你好，AIGC是什么呀")
    answer["session"] = {"mode": "SPEAKING"}
    report = score_clip(clip, [greeting, greet_yield, answer, later], "")
    assert report["observations"][0]["ok"] is False


def test_false_barge_yield_is_not_a_true_interrupt():
    clip = {"id": "v1-A07", "observations": [{"kind": "has_yield"}]}
    report = score_clip(clip, [_yield(6700, "给", "false_barge")], "")
    assert report["verdict"] == "FAIL"
    report = score_clip(clip, [_yield(20200, "别说了", "true_interrupt")], "")
    assert report["verdict"] == "PASS"


def test_answer_without_wait_does_not_pass_a06():
    clip = {"id": "v1-A06", "observations": [{"kind": "has_wait_or_deferred_answer"}]}
    assert score_clip(clip, [_answer(9000, "附近的麦当劳")], "")["verdict"] == "FAIL"
    waited = {"t_ms": 4000, "action": "WAIT", "sop": "06", "speech": {}, "session": {}, "probes": {}}
    assert score_clip(clip, [waited, _answer(9000, "附近的麦当劳")], "")["verdict"] == "PASS"


def test_cases_roundtrip_version():
    raw = json.loads(CASES.read_text(encoding="utf-8"))
    assert raw["version"] == 2
