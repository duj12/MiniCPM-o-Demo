"""Scorer bans. These stay red when a loose gate would go green."""
from __future__ import annotations

import json
from pathlib import Path

from score import load_cases, resume_closes_pause_at_or_after, runnable_cases, score_clip

CASES = Path(__file__).with_name("cases.json")


def _answer(t_ms: int, text: str, emotion: str = "", reply: bool | None = None) -> dict:
    event = {
        "t_ms": t_ms,
        "action": "ANSWER",
        "speech": {"transcript": text, "emotion": emotion},
        "probes": {},
        "session": {},
    }
    if reply is not None:
        event["probes"]["expects_reply"] = reply
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
    # SOP04 只保留 04_1026.mkv 这一条。
    assert [cid for cid in ids if cid.startswith("v2-04-")] == ["v2-04-1"]
    by_suite = {}
    for clip in runnable:
        by_suite[clip["suite"]] = by_suite.get(clip["suite"], 0) + 1
    assert by_suite == {"v2": 30, "v1_new": 21, "supplement": 3}


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
    answer = _answer(19000, "给我介绍一下你们的展厅吧。")
    log = "\n".join([
        "[4000ms face]",
        "[tts.pause greet] 停在 1.00s",
        "[tts.resume greet] 从 1.00s 接着播",
    ])
    report = score_clip(clip, [answer], log)
    assert report["observations"][1]["ok"] is False
    heard = {
        "t_ms": 22200,
        "action": "LISTEN",
        "speech": {"transcript": "嗯那。"},
        "session": {},
        "probes": {},
    }
    report = score_clip(clip, [answer, heard], log)
    assert report["observations"][1]["ok"] is True
    report = score_clip(clip, [answer, heard, _yield(23000, "嗯那。", "false_barge")], log)
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


def test_yield_does_not_require_the_old_three_way():
    clip = {"id": "v1-A07", "observations": [{"kind": "has_yield"}]}
    assert score_clip(clip, [_yield(6700, "给", "false_barge")], "")["verdict"] == "PASS"
    assert score_clip(clip, [_yield(20200, "别说了", "true_interrupt")], "")["verdict"] == "PASS"
    assert score_clip(clip, [_answer(1000, "别说了")], "")["verdict"] == "FAIL"


def test_ask_again_is_the_spoken_line():
    clip = {"id": "v2-01-1", "observations": [{"kind": "utter_sop", "sop": "01"}]}
    numbered = {
        "t_ms": 1000,
        "action": "UTTER",
        "sop": "01",
        "text": "别的话",
        "speech": {},
        "session": {},
        "probes": {},
    }
    assert score_clip(clip, [numbered], "")["verdict"] == "FAIL"
    spoken = {
        "t_ms": 1000,
        "action": "UTTER",
        "text": "抱歉，能请您再说一遍吗？",
        "speech": {},
        "session": {},
        "probes": {},
    }
    assert score_clip(clip, [spoken], "")["verdict"] == "PASS"


def test_filler_is_sent_by_the_agent_not_uttered_by_ic():
    clip = {"id": "v1-A26", "observations": [{"kind": "agent_filler_to_tts"}]}
    uttered = {
        "t_ms": 1000,
        "action": "UTTER",
        "text": "我查一下",
        "speech": {},
        "session": {},
        "probes": {},
    }
    assert score_clip(clip, [uttered], "[tts.start a] 我查一下")["verdict"] == "FAIL"
    inserted = {"t_ms": 1000, "action": "INSERT", "speech": {}, "session": {}, "probes": {}}
    assert score_clip(clip, [inserted], "")["verdict"] == "FAIL"
    held = {"t_ms": 1000, "action": "HOLD", "speech": {}, "session": {}, "probes": {}}
    assert score_clip(clip, [held], "[tts.end b] 我查一下")["verdict"] == "PASS"


def test_closing_line_ends_in_interactioncore():
    clip = {"id": "v1-A39", "observations": [{"kind": "closing_end_by_ic"}]}
    said = {
        "t_ms": 1000,
        "action": "HOLD",
        "speech": {"transcript": "不用了。"},
        "session": {},
        "probes": {},
    }
    farewell = {
        "t_ms": 1100,
        "action": "UTTER",
        "sop": "39",
        "text": "好的，再见。",
        "speech": {"transcript": "不用了。"},
        "session": {},
        "probes": {},
    }
    ended = {
        "t_ms": 1400,
        "action": "END",
        "sop": "39",
        "speech": {},
        "session": {},
        "probes": {},
    }
    left = {"t_ms": 2000, "action": "END", "sop": "24", "speech": {}, "session": {}, "probes": {}}
    log = "[tts.start c] 好的，再见。"
    assert score_clip(clip, [said, farewell, ended], log)["verdict"] == "PASS"
    assert score_clip(clip, [said, ended], log)["verdict"] == "FAIL"
    assert score_clip(clip, [said, farewell, ended], "")["verdict"] == "FAIL"
    assert score_clip(clip, [said, farewell, left], log)["verdict"] == "FAIL"
    handed = _answer(1000, "不用了")
    assert score_clip(clip, [handed, farewell, ended], log)["verdict"] == "FAIL"


def test_leave_requires_end_not_session_end_pending():
    clip = {"id": "v1-D24", "observations": [{"kind": "has_end_or_session_close"}]}
    pending = {
        "t_ms": 4000,
        "action": "HOLD",
        "speech": {},
        "session": {},
        "probes": {},
        "agent": {"session_end_pending": True},
    }
    assert score_clip(clip, [pending], "")["verdict"] == "FAIL"
    ended = {"t_ms": 4000, "action": "END", "speech": {}, "session": {}, "probes": {}}
    assert score_clip(clip, [ended], "")["verdict"] == "PASS"


def test_answer_without_wait_does_not_pass_a06():
    clip = {"id": "v1-A06", "observations": [{"kind": "has_wait_or_deferred_answer"}]}
    assert score_clip(clip, [_answer(9000, "附近的麦当劳")], "")["verdict"] == "FAIL"
    waited = {"t_ms": 4000, "action": "WAIT", "sop": "06", "speech": {}, "session": {}, "probes": {}}
    assert score_clip(clip, [waited, _answer(9000, "附近的麦当劳")], "")["verdict"] == "PASS"


def test_answer_target_count_requires_both_repeats():
    clip = {
        "id": "supp-9554",
        "observations": [
            {"kind": "answer_target_count", "pattern": "交互体验", "n": 2},
        ],
    }
    one = [_answer(1000, "给我介绍魔珐科技"), _answer(16000, "跟你的交互体验是这个")]
    assert score_clip(clip, one, "")["verdict"] == "FAIL"
    two = one + [_answer(30000, "跟你的交互体验是这个")]
    assert score_clip(clip, two, "")["verdict"] == "PASS"


def test_tts_answers_played_rejects_start_cancel():
    clip = {
        "id": "supp-9554",
        "observations": [{"kind": "tts_answers_played", "n": 2}],
    }
    events = [_answer(500, "回答")]
    ok = "\n".join([
        "[1000ms] [tts.start a] 你好",
        "[tts.end a] 你好",
        "[2000ms] [tts.start b] 好的",
        "[tts.end b] 好的",
    ])
    assert score_clip(clip, events, ok)["verdict"] == "PASS"
    assert score_clip(
        clip, events, "[1000ms] [tts.start a] 你好"
    )["verdict"] == "FAIL"
    cut = ok + "\n[3000ms] [tts.cancel b] 已播 0.00s"
    assert score_clip(clip, events, cut)["verdict"] == "FAIL"


def test_tts_answers_played_ignores_tts_before_the_first_answer():
    clip = {
        "id": "supp-9554",
        "observations": [{"kind": "tts_answers_played", "n": 1}],
    }
    events = [_answer(1000, "回答")]
    log = "\n".join([
        "[200ms] [tts.start greeting] 您好，欢迎光临。",
        "[1500ms] [tts.start answer] 这是回答。",
    ])
    assert score_clip(clip, events, log)["verdict"] == "PASS"

    only_greeting = "[200ms] [tts.start greeting] 您好，欢迎光临。"
    assert score_clip(clip, events, only_greeting)["verdict"] == "FAIL"


def test_tts_answers_played_ignores_agent_filler_after_the_answer():
    clip = {
        "id": "supp-9554",
        "observations": [{"kind": "tts_answers_played", "n": 1}],
    }
    events = [_answer(1000, "回答")]
    for filler in ("嗯，让我看一下", "好的，我查一下", "让我看看", "稍等", "我看看"):
        log = f"[1500ms] [tts.start filler] {filler}"
        assert score_clip(clip, events, log)["verdict"] == "FAIL", filler


def test_tts_answers_played_counts_three_answer_segments():
    clip = {
        "id": "supp-9554",
        "observations": [{"kind": "tts_answers_played", "n": 3}],
    }
    events = [_answer(1000, "回答")]
    log = "\n".join([
        "[200ms] [tts.start greeting] 您好，欢迎光临。",
        "[1500ms] [tts.start a] 第一段回答",
        "[3000ms] [tts.start b] 第二段回答",
        "[5000ms] [tts.start c] 第三段回答",
    ])
    assert score_clip(clip, events, log)["verdict"] == "PASS"


def _orchestrator_replay():
    """Import the replay client even when this env has no numpy installed."""
    import importlib
    import sys
    import types

    try:
        return importlib.import_module("orchestrator_replay")
    except ModuleNotFoundError:
        if "numpy" in sys.modules:
            raise
        sys.modules["numpy"] = types.ModuleType("numpy")
        return importlib.import_module("orchestrator_replay")


def test_ic_event_record_preserves_the_turn_block():
    replay = _orchestrator_replay()
    turn = {"pending_target": "answer", "pending_waited_ms": 42}
    rec = replay._ic_event_record({
        "action": "ANSWER",
        "t_ms": 1234,
        "turn": turn,
    })
    assert rec["turn"] == turn


def test_cases_roundtrip_version():
    raw = json.loads(CASES.read_text(encoding="utf-8"))
    assert raw["version"] == 2
