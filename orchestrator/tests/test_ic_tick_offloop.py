"""IC tick only enqueues callbacks; slow host output cannot hold algorithm execution."""
import asyncio
import socket

import pytest

from interaction.state import SessionMode
from orchestrator.interaction.inprocess import InProcessICClient


@pytest.mark.asyncio
async def test_end_tick_enqueues_while_host_output_waits(monkeypatch):
    monkeypatch.setenv("ORCH_IC_DECISION_PROBE", "0")
    def unexpected_connection(*args, **kwargs):
        raise AssertionError("Local IC must not open a transport")
    monkeypatch.setattr(socket, "create_connection", unexpected_connection)
    events = asyncio.Queue()
    client = InProcessICClient("local", on_agent_event=events.put_nowait,
                               on_expression_event=events.put_nowait)
    assert client.connect(), client.error
    started, release = asyncio.Event(), asyncio.Event()
    consumed = []
    async def host_output():
        event = await events.get()
        consumed.append(event)
        started.set()
        await release.wait()
    consumer = asyncio.create_task(host_output())
    try:
        client.engine.state.session.mode = SessionMode.SPEAKING
        client.engine.state.session.absent_ms = 10_000
        action = await asyncio.wait_for(client.tick(50), 0.5)
        await asyncio.wait_for(started.wait(), 0.5)
        assert action.type.value == "END"
        assert consumed[0] == {"type": "stop", "session_id": "local", "reason": "ic_stop"}
        assert not consumer.done()
        # A second tick still returns while expression execution is deliberately held.
        assert await asyncio.wait_for(client.tick(50), 0.5) is not None
        assert client.tick_calls == 2
        assert (await events.get())["type"] == "end"
    finally:
        release.set()
        await consumer
        client.close()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
