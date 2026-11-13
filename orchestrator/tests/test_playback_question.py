"""播报中要不要问，由 InteractionCore 决定。编排器只把问发出去，并把「是」或「不是」写回。"""

from __future__ import annotations

import asyncio

from interaction.decision_client import DecisionAnswer
from interaction.state import Confidence, SessionMode
from orchestrator.interaction.inprocess import InProcessICClient


def _playing(engine, text: str) -> None:
    engine.apply_face(Confidence.HIGH, 0.2)
    engine.apply_track("t1", 3000)
    engine.apply_lip(Confidence.HIGH)
    engine.apply_vad(Confidence.HIGH, Confidence.NONE)
    engine.state.session.mode = SessionMode.SPEAKING
    engine.state.session.greet_spoken = True
    engine.state.session.subject_track_id = "t1"
    engine.state.session.answered_since_greet = True
    engine.apply_playback_active(True)
    engine.apply_asr(text, Confidence.HIGH, Confidence.LOW)


def _open_client(monkeypatch, sent: list[dict], choice: str | None):
    def systemone(self, questions, state, *, think_mode="never"):
        sent.append(questions)
        if choice is None:
            return DecisionAnswer({})
        return DecisionAnswer(
            {
                "对着当前会话": {
                    "type": "choice",
                    "choice": choice,
                    "confidence": 0.9,
                }
            }
        )

    monkeypatch.setattr(
        "interaction.decision_client.DecisionClient.systemone",
        systemone,
    )
    monkeypatch.setenv("ORCH_IC_DECISION_PROBE", "1")
    events: list[dict] = []
    client = InProcessICClient("s", on_expression_event=events.append)
    assert client.connect(), client.error
    return client, events


def test_playing_long_line_asks_this_session_and_writes_yes(monkeypatch) -> None:
    """满三个字、不是应声、不是叫停、还没有记录：编排器只问对着当前会话，并把「是」写回。"""
    sent: list[dict] = []
    client, events = _open_client(monkeypatch, sent, "是")
    try:
        _playing(client.engine, "今天天气怎么样")
        action = asyncio.run(client.tick(0))
        names = [name for questions in sent for name in questions]
        assert "barge_class" not in names
        assert "对着当前会话" in names
        question = next(
            questions["对着当前会话"]
            for questions in sent
            if "对着当前会话" in questions
        )
        assert question["type"] == "choice"
        assert set(question["criteria"]) == {"是", "不是"}
        assert client.engine.state.probes.addressed_to_session == "是"
        assert action.type.value != "YIELD"
        assert [event["type"] for event in events] == ["pause"]
        assert events[0]["fade_ms"] > 0
    finally:
        client.close()


def test_no_answer_resumes_on_the_next_beat(monkeypatch) -> None:
    """记录写成「不是」之后，下一拍立刻继续原来的播报，不再问。"""
    sent: list[dict] = []
    client, events = _open_client(monkeypatch, sent, "不是")
    try:
        _playing(client.engine, "今天天气怎么样")
        first = asyncio.run(client.tick(0))
        assert client.engine.state.probes.addressed_to_session == "不是"
        assert first.type.value != "YIELD"
        second = asyncio.run(client.tick(0))
        names = [name for questions in sent for name in questions]
        assert names.count("对着当前会话") == 1
        assert "barge_class" not in names
        assert second.type.value != "YIELD"
        assert [event["type"] for event in events] == ["pause", "resume"]
        assert events[0]["fade_ms"] > 0
        assert client.engine.state.session.playback_active is True
    finally:
        client.close()


def test_backchannel_short_and_stop_do_not_ask(monkeypatch) -> None:
    """应声、不足三个字、叫停，编排器都不问三分法，也不问对着当前会话。"""
    sent: list[dict] = []

    for text in ("嗯嗯", "好的"):
        client, events = _open_client(monkeypatch, sent, None)
        try:
            _playing(client.engine, text)
            action = asyncio.run(client.tick(0))
            names = [name for questions in sent for name in questions]
            assert "对着当前会话" not in names, text
            assert "barge_class" not in names, text
            assert action.type.value != "YIELD", text
            assert events == [], text
        finally:
            client.close()
        sent.clear()

    client, events = _open_client(monkeypatch, sent, None)
    try:
        _playing(client.engine, "别说了")
        action = asyncio.run(client.tick(0))
        names = [name for questions in sent for name in questions]
        assert "对着当前会话" not in names
        assert "barge_class" not in names
        assert action.type.value == "YIELD"
        assert [event["type"] for event in events] == ["stop"]
        assert client.engine.state.probes.addressed_to_session is None
    finally:
        client.close()
