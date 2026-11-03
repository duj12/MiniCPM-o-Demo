"""Local IC state and callback isolation; no remote IC protocol is involved."""
import asyncio

import pytest

from interaction.state import AgentStatus, Confidence
from orchestrator.interaction.inprocess import InProcessICClient


@pytest.fixture(autouse=True)
def no_probe_network(monkeypatch):
    monkeypatch.setenv("ORCH_IC_DECISION_PROBE", "0")


@pytest.fixture
def clients():
    made = []
    def make(sid):
        client = InProcessICClient(sid)
        assert client.connect(), client.error
        made.append(client)
        return client
    yield make
    for client in made:
        client.close()


def test_state_and_agent_projection_are_session_local(clients):
    first, second = clients("A"), clients("B")
    assert first.engine is not second.engine
    first.apply("apply_asr", transcript="A说的话", asr_confidence="HIGH")
    second.apply("apply_asr", transcript="B说的话", asr_confidence="LOW")
    first.apply("apply_identity", identity_id="uid-A", identity_confidence="HIGH", display_name="甲")
    first.apply_agent(status="pending_announce", session_end_pending=True)
    assert first.engine.state.speech.transcript == "A说的话"
    assert second.engine.state.speech.transcript == "B说的话"
    assert second.engine.state.person.identity_id is None
    assert first.engine.state.agent.status is AgentStatus.PENDING_ANNOUNCE
    assert second.engine.state.agent.status is AgentStatus.IDLE
    assert second.engine.state.agent.session_end_pending is False


@pytest.mark.parametrize("status", ["BUSY", "busy", "Busy"])
def test_agent_projection_normalizes_status_and_preserves_omitted_fields(clients, status):
    client = clients("state")
    client.apply_agent(status=status)
    client.apply_agent(session_end_pending=True)
    assert client.engine.state.agent.status is AgentStatus.BUSY
    assert client.engine.state.agent.session_end_pending is True
    client.apply_agent(status="unknown")
    client.apply_agent()
    assert client.engine.state.agent.status is AgentStatus.BUSY
    assert client.engine.state.agent.session_end_pending is True


def test_confidence_strings_support_real_policy_ticks(clients):
    client = clients("confidence")
    client.apply("apply_face", face_present_confidence="HIGH", bbox_area_ratio=0.12)
    client.apply("apply_track", track_id="7", dwell_ms=2500)
    client.apply("apply_lip", lip_speaking_confidence="LOW")
    client.apply("apply_identity", identity_id="uid-x", identity_confidence="HIGH", display_name="甲")
    client.apply("apply_vad", user_speaking_confidence="HIGH", barge_in_confidence="MEDIUM")
    client.apply("apply_asr", transcript="你好", asr_confidence="HIGH", turn_complete_confidence="HIGH")
    state = client.engine.state
    stored = [state.person.face_present_confidence, state.person.lip_speaking_confidence,
              state.person.identity_confidence, state.speech.user_speaking_confidence,
              state.speech.barge_in_confidence, state.speech.asr_confidence]
    assert all(isinstance(value, Confidence) for value in stored)
    async def drive():
        return [await client.tick(50) for _ in range(30)]
    actions = asyncio.run(drive())
    assert client.bad_ticks == 0, client.error
    assert all(action is not None for action in actions)


def test_end_callbacks_and_reset_belong_to_their_host_session():
    first_events, second_events = [], []
    first = InProcessICClient("A", on_agent_event=first_events.append,
                              on_expression_event=first_events.append)
    second = InProcessICClient("B", on_agent_event=second_events.append,
                               on_expression_event=second_events.append)
    assert first.connect() and second.connect()
    try:
        second.apply("apply_asr", transcript="保留B", asr_confidence="HIGH")
        assert first.end_session()
        assert [(event["session_id"], event["type"]) for event in first_events] == [
            ("A", "stop"), ("A", "end")]
        assert second_events == []
        assert second.engine.state.speech.transcript == "保留B"
    finally:
        first.close()
        second.close()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
