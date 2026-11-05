"""Host/IC contract and actual independent Agent gRPC loopback regressions."""
import asyncio
from contextlib import asynccontextmanager
import json
import os
from pathlib import Path

import pytest

from agent_protocol import agent_pb2 as pb
from orchestrator.downstream.interface import Cancel, PauseTts, ResumeTts, Speak
from orchestrator.interaction.agent import AgentClient
from orchestrator.interaction.downstream import InteractionDownstream


@pytest.fixture(autouse=True)
def no_remote_probe(monkeypatch):
    monkeypatch.setenv("ORCH_IC_DECISION_PROBE", "0")


def host(target="127.0.0.1:1", sid="room-a"):
    value = InteractionDownstream("inprocess", target, session_id=sid)
    assert value.ic.connect()
    return value


def action(client, kind, **fields):
    assert client.submit({"type": kind, **fields})
    message = client._queue.get_nowait()
    client._queue.task_done()
    return message


def test_control_and_vision_correlation_and_optional_payload():
    client = AgentClient("127.0.0.1:1", session_id="room", on_event=lambda event: None,
                         mcp_servers=[{"name": "tools", "transport": "http", "url": "http://tools/mcp"}])
    try:
        first = action(client, "answer", transcript="你好", identity_id="p1", display_name="甲",
                       affect={"label": "happy", "confidence": 0.8}, language="zh")
        assert first.act == pb.AgentAction.ANSWER and first.session_id == "room"
        assert first.affect.label == "happy" and first.language == "zh"
        assert first.mcp_servers.servers[0].name == "tools"
        for kind in ("vision", "insert", "yield", "end"):
            msg = action(client, kind, vlm="ATTENTION: 看镜头")
            assert (msg.ci, msg.epoch) == (first.ci, first.epoch)
        visual = action(client, "vision", vlm="ENV: 新场景")
        assert visual.epoch > first.epoch and visual.ci != first.ci
        second = action(client, "answer", transcript="新话题")
        assert second.epoch > visual.epoch and second.vlm == ""
    finally:
        client.close()


def test_sink_events_are_local_and_sessions_have_independent_state():
    a, b = host(sid="a"), host(sid="b")
    try:
        a.ic.apply("apply_asr", transcript="只属于A", asr_confidence="HIGH")
        assert b.ic.engine.state.speech.transcript != "只属于A"
        sink = a.ic.engine.expression_sink
        sink.play_template("您好")
        sink.pause(reason="soft", fade_ms=100)
        sink.resume()
        sink.stop()
        out = a._drain_events()
        assert [type(item) for item in out] == [Speak, PauseTts, ResumeTts, Cancel]
        assert out[0].text == "您好" and out[1].fade_ms == 100
        assert b._drain_events() == []
    finally:
        a.ic.close()
        b.ic.close()
        a.agent.close()
        b.agent.close()


def test_stale_text_is_filtered_at_host_consume_and_done_is_not_playback():
    value = host()
    try:
        first = action(value.agent, "answer", transcript="第一轮")
        value._events.put(("agent", {"type": "agent.text", "kind": "answer", "ci": first.ci,
                                      "epoch": first.epoch, "stream_id": "s", "text": "旧正文", "is_final": True}))
        second = action(value.agent, "answer", transcript="第二轮")
        assert value._drain_events() == []
        value.ic.apply("apply_playback_active", playback_active=True)
        value._events.put(("agent", {"type": "agent.done", "ci": second.ci, "epoch": second.epoch}))
        assert value._drain_events() == []
        assert value.ic.engine.state.session.playback_active is True
        action(value.agent, "yield")
        value._events.put(("agent", {"type": "agent.text", "kind": "answer", "ci": second.ci,
                                      "epoch": second.epoch, "stream_id": "s", "text": "取消正文", "is_final": True}))
        assert value._drain_events() == []
    finally:
        value.ic.close()
        value.agent.close()


async def wait_for(predicate, timeout=5):
    end = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() > end:
            raise AssertionError("Timed out waiting for Agent events")
        await asyncio.sleep(0.01)


@asynccontextmanager
async def real_service():
    source = os.environ.get("AGENT_TEST_SOURCE")
    python = os.environ.get("AGENT_TEST_PYTHON")
    if not source or not python:
        pytest.skip("Set AGENT_TEST_SOURCE and AGENT_TEST_PYTHON for the real Agent service loopback")
    harness = Path(__file__).with_name("agent_loopback_server.py")
    process = await asyncio.create_subprocess_exec(
        python, "-u", str(harness), str(Path(source).resolve()),
        stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE)
    try:
        line = await asyncio.wait_for(process.stdout.readline(), 20)
        if not line:
            error = (await process.stderr.read()).decode(errors="replace")
            raise AssertionError("Agent fixture did not start: " + error[-3000:])
        target = "127.0.0.1:" + str(json.loads(line)["port"])
        async def stats():
            process.stdin.write(b"stats\n")
            await process.stdin.drain()
            return json.loads(await asyncio.wait_for(process.stdout.readline(), 5))
        yield target, stats
    finally:
        if process.returncode is None:
            process.stdin.write(b"quit\n")
            await process.stdin.drain()
            try:
                await asyncio.wait_for(process.wait(), 10)
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()


@pytest.mark.asyncio
async def test_actual_agent_multi_session_insert_yield_end_and_stream_output():
    async with real_service() as (target, stats):
        one, two = host(target, "one"), host(target, "two")
        received = {"one": [], "two": []}
        def pump():
            for item in (one, two):
                received[item.session_id].extend(item._drain_events())
            return True
        def spoken(sid):
            pump()
            return "".join(x.text for x in received[sid] if isinstance(x, Speak))
        try:
            one.agent.start()
            two.agent.start()
            one.ic.engine.agent_sink.on_answer(identity_id="p1", display_name="甲", transcript="slow-query")
            two.ic.engine.agent_sink.on_answer(identity_id="p2", display_name="乙", transcript="fast-two")
            pump()
            await wait_for(lambda: "正在查询" in spoken("one") and "正文:fast-two" in spoken("two"))
            await wait_for(lambda: (pump() and one.ic.engine.state.agent.status.value == "pending_announce"))
            assert "正文:slow-query" not in spoken("one")
            one.ic.engine.agent_sink.on_insert()
            pump()
            await wait_for(lambda: "正文:slow-query" in spoken("one"))
            assert "fast-two" not in spoken("one") and "slow-query" not in spoken("two")
            one.ic.apply("apply_playback_active", playback_active=True)
            await wait_for(lambda: (pump() and one.ic.engine.state.agent.status.value == "idle"))
            assert one.ic.engine.state.session.playback_active is True
            one.ic.engine.agent_sink.on_answer(identity_id=None, display_name=None, transcript="blocked-old")
            pump()
            await wait_for(lambda: "正在处理" in spoken("one"))
            one.ic.engine.agent_sink.on_yield()
            pump()
            await wait_for(lambda: (pump() and one.ic.engine.state.agent.status.value == "idle"))
            snapshot = await stats()
            assert snapshot["cancelled"] >= 1
            before = len(snapshot["starts"])
            one.ic.engine.agent_sink.on_answer(identity_id=None, display_name=None, transcript="fast-after-yield")
            pump()
            await wait_for(lambda: "正文:fast-after-yield" in spoken("one"))
            assert len((await stats())["starts"]) == before
            one.ic.engine.agent_sink.on_end()
            pump()
            await asyncio.sleep(0.1)
            assert (await stats())["deleted"]
            one.ic.engine.agent_sink.on_vision("ENV: 新场景")
            pump()
            one.ic.engine.agent_sink.on_answer(identity_id=None, display_name=None, transcript="fast-new-context")
            pump()
            await wait_for(lambda: "正文:fast-new-context" in spoken("one"))
            assert len((await stats())["starts"]) > before
            assert one.agent._thread.is_alive()  # END kept this same RPC alive.
        finally:
            for item in (one, two):
                await asyncio.to_thread(item.agent.close)
                item.ic.close()


@pytest.mark.asyncio
async def test_actual_agent_disconnect_is_terminal_and_does_not_replay():
    async with real_service() as (target, stats):
        value = host(target, "disconnect")
        try:
            value.agent.start()
            value.ic.engine.agent_sink.on_answer(identity_id=None, display_name=None, transcript="blocked-disconnect")
            value._drain_events()
            await wait_for(lambda: value.agent._active is not None)
            await asyncio.sleep(0.1)
            value.agent._call.cancel()
            await wait_for(lambda: value.agent._failed)
            out = value._drain_events()
            assert any(isinstance(item, Cancel) and item.reason == "agent_disconnect" for item in out)
            assert not value.agent.submit({"type": "answer", "transcript": "no-replay"})
            await asyncio.sleep(0.1)
            assert "no-replay" not in (await stats())["calls"]
            old_connection = value.agent.connection_id
            value.ic.engine.agent_sink.on_answer(identity_id=None, display_name=None, transcript="fast-new-rpc")
            value._drain_events()
            assert value.agent.connection_id != old_connection
            outputs = []
            def fresh_reply():
                outputs.extend(value._drain_events())
                return any(isinstance(item, Speak) and "fast-new-rpc" in item.text for item in outputs)
            await wait_for(fresh_reply)
            assert len((await stats())["starts"]) == 2
        finally:
            await asyncio.to_thread(value.agent.close)
            value.ic.close()


@pytest.mark.asyncio
async def test_actual_agent_error_and_rejected_answer_clear_projection():
    async with real_service() as (target, stats):
        value = host(target, "errors")
        out = []
        def pump():
            out.extend(value._drain_events())
            return True
        try:
            value.agent.start()
            value.ic.engine.agent_sink.on_answer(identity_id=None, display_name=None, transcript="error-turn")
            pump()
            await wait_for(lambda: (pump() and any(isinstance(x, Cancel) and x.reason == "agent_error" for x in out)))
            assert value.ic.engine.state.agent.status.value == "idle"
            assert all(x.guard is None or not x.guard() for x in out if isinstance(x, Speak))
            assert value.agent._thread.is_alive()
            out.clear()
            # Deliberately violate the high-water mark to exercise a real server rejection.
            value.agent._epoch -= 1
            value.ic.apply_agent(status="BUSY", session_end_pending=True)
            value.ic.engine.agent_sink.on_answer(identity_id=None, display_name=None, transcript="rejected-stale")
            pump()
            await wait_for(lambda: (pump() and any(isinstance(x, Cancel) and x.reason == "agent_rejected" for x in out)))
            assert value.ic.engine.state.agent.status.value == "idle"
            assert value.ic.engine.state.agent.session_end_pending is False
        finally:
            await asyncio.to_thread(value.agent.close)
            value.ic.close()


@pytest.mark.asyncio
async def test_host_pause_wait_does_not_block_tick_or_hard_stop():
    gate = asyncio.Event()
    seen = []
    async def output(item):
        seen.append(item.kind)
        if isinstance(item, PauseTts):
            await gate.wait()
        if isinstance(item, Cancel):
            gate.set()
    value = InteractionDownstream("inprocess", "127.0.0.1:1", session_id="controls", on_output=output)
    assert value.ic.connect()
    try:
        sink = value.ic.engine.expression_sink
        sink.pause(fade_ms=1000)
        await value._deliver(value._drain_events())
        await asyncio.sleep(0)
        assert "pause_tts" in seen and not gate.is_set()
        await asyncio.wait_for(value.ic.tick(50), 0.1)
        sink.stop()
        await value._deliver(value._drain_events())
        await wait_for(gate.is_set)
        assert "cancel" in seen
    finally:
        await value.on_session_end("test")


def test_template_guard_and_failed_apply_are_safe():
    value = host()
    try:
        value.ic.apply("apply_asr", unexpected_keyword=True)
        assert value.ic.failures["apply_asr"] == 1
        value.ic.engine.expression_sink.play_template("过期模板")
        value.ic.engine.expression_sink.stop()
        out = value._drain_events()
        assert isinstance(out[0], Speak) and out[0].guard is not None
        assert out[0].guard() is False
    finally:
        value.ic.close()
        value.agent.close()


def test_cancelled_turn_discards_queued_and_late_busy_state():
    value = host()
    try:
        first = action(value.agent, "answer", transcript="旧轮")
        revision = value.agent._revision
        value._feedback[(first.ci, "partial-feedback")] = "未完成片段"
        value._events.put(("action", {"type": "yield"}))
        value._events.put(("agent", {"type": "agent.state", "ci": first.ci, "epoch": first.epoch,
                                      "revision": revision, "status": "BUSY", "session_end_pending": True}))
        value._drain_events()
        assert value._feedback == {}
        assert value.ic.engine.state.agent.status.value == "idle"
        assert value.ic.engine.state.agent.session_end_pending is False
        captured = []
        value.agent._on_event = captured.append
        value.agent._call = iter([
            pb.AgentEvent(session_id=value.session_id, ci=first.ci, epoch=first.epoch,
                          act=pb.AgentAction.ANSWER,
                          state=pb.AgentState(status=pb.AgentState.BUSY, session_end_pending=True)),
            pb.AgentEvent(session_id=value.session_id, ci=first.ci, epoch=first.epoch,
                          act=pb.AgentAction.ANSWER,
                          state=pb.AgentState(status=pb.AgentState.IDLE, session_end_pending=False)),
        ])
        value.agent._receive()
        states = [event for event in captured if event["type"] == "agent.state"]
        assert len(states) == 1 and states[0]["status"] == "IDLE"
        assert states[0]["session_end_pending"] is False
        assert value.agent._failed and value.agent._queue.get_nowait() is None
        value.agent._call = None
    finally:
        value.ic.close()
        value.agent.close()
