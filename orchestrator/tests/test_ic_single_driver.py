"""Regression: overlapping local IC sessions all keep driving their own state.

The former remote-IC single-driver takeover guard has been retired.
"""
import asyncio

import pytest

from orchestrator.interaction.inprocess import InProcessICClient


@pytest.mark.asyncio
async def test_overlapping_sessions_are_not_suspended_or_retargeted(monkeypatch):
    monkeypatch.setenv("ORCH_IC_DECISION_PROBE", "0")
    clients = [InProcessICClient(f"room-{index}") for index in range(8)]
    try:
        assert all(client.connect() for client in clients)
        assert len({id(client.engine) for client in clients}) == len(clients)
        for index, client in enumerate(clients):
            client.apply("apply_asr", transcript=f"独立文本{index}", asr_confidence="HIGH")
        async def drive(client):
            for _ in range(5):
                assert await client.tick(50) is not None
                await asyncio.sleep(0)
        await asyncio.gather(*(drive(client) for client in clients))
        assert [client.tick_calls for client in clients] == [5] * len(clients)
        assert all(not client.suspended and client.suspended_ticks == 0 for client in clients)
        assert [client.engine.state.speech.transcript for client in clients] == [
            f"独立文本{index}" for index in range(len(clients))]
        clients[0].close()
        clients[1].apply("apply_asr", transcript="仍然可写", asr_confidence="HIGH")
        assert await clients[1].tick(50) is not None
        assert clients[1].engine.state.speech.transcript == "仍然可写"
        assert clients[1].available
    finally:
        for client in clients:
            client.close()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
