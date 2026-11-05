"""Host-owned bidirectional Agent transport; InteractionCore remains local."""
from __future__ import annotations
import logging
from collections import defaultdict, deque
import queue
import threading
import uuid
from typing import Callable

logger = logging.getLogger(__name__)


class AgentClient:
    """One RPC per host session. A failed RPC never replays its action queue."""

    def __init__(self, target: str, *, session_id: str,
                 on_event: Callable[[dict], None], mcp_servers: list | None = None):
        if "://" in target:
            raise ValueError("ORCH_AGENT_GRPC must be a host:port gRPC target")
        self.target, self.session_id = target, session_id
        self.connection_id = uuid.uuid4().hex
        self._on_event = on_event
        self._mcp_servers = mcp_servers or []
        self._queue = queue.Queue(maxsize=256)
        self._lock = threading.RLock()
        self._thread = self._channel = self._call = None
        self._closed = self._failed = False
        self._epoch = self._revision = 0
        self._active = None
        self._accept_text = self._text_done = self._ended = False
        self._vision = ""
        self._pending_acks = defaultdict(deque)
        self.error = None

    def _emit(self, event):
        self._on_event({"connection_id": self.connection_id, "revision": self._revision, **event})

    def start(self) -> None:
        with self._lock:
            if self._thread is not None:
                return
            if self._closed:
                raise RuntimeError("Agent client already closed")
            import grpc
            from agent_protocol.agent_pb2_grpc import AgentServiceStub
            self._channel = grpc.insecure_channel(self.target)
            self._call = AgentServiceStub(self._channel).Interact(self._requests())
            self._thread = threading.Thread(target=self._receive, daemon=True,
                                            name=f"agent-{self.session_id[:8]}")
            self._thread.start()

    def _requests(self):
        while True:
            item = self._queue.get()
            try:
                if item is None:
                    return
                yield item
            finally:
                self._queue.task_done()

    def _finish_requests(self):
        # No replay on EOF, cancellation or close. Wake a blocked gRPC iterator.
        while True:
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except queue.Empty:
                break
        self._queue.put_nowait(None)

    def submit(self, event: dict) -> bool:
        """Assign wire identity in the host; controls reference the active ANSWER."""
        from agent_protocol import agent_pb2 as pb
        kind = event.get("type")
        with self._lock:
            if self._closed or self._failed:
                return False
            if kind == "vision":
                self._vision = str(event.get("vlm") or "")
                act, transcript = pb.AgentAction.ANSWER, ""
                if not self._vision:
                    return False
            else:
                acts = {"answer": pb.AgentAction.ANSWER, "insert": pb.AgentAction.INSERT,
                        "yield": pb.AgentAction.YIELD, "end": pb.AgentAction.END}
                act = acts.get(kind)
                if act is None:
                    return False
                transcript = str(event.get("transcript") or "")
            if kind not in ("answer", "vision") and self._active is None:
                return False
            if kind == "answer" or self._active is None or (kind == "vision" and self._ended):
                self._epoch += 1
                self._active = uuid.uuid4().hex
                self._pending_acks.clear()
                self._revision += 1
            if kind == "answer":
                self._accept_text = True
                self._text_done = self._ended = False
            elif kind in ("yield", "end"):
                self._revision += 1
                self._accept_text = False
                self._text_done = True
                self._ended = kind == "end"
            elif kind == "vision":
                self._ended = False
            if kind == "end":
                self._vision = ""
            ci, epoch = self._active, self._epoch
            msg = pb.AgentAction(act=act, session_id=self.session_id, ci=ci,
                                 epoch=epoch, transcript=transcript, vlm=self._vision if kind == "vision" else "")
            for key in ("identity_id", "display_name", "language", "app_id", "ai_intent", "ai_style"):
                if event.get(key) is not None:
                    setattr(msg, key, str(event[key]))
            affect = event.get("affect")
            if affect:
                msg.affect.label = str(affect.get("label") or "")
                msg.affect.confidence = float(affect.get("confidence") or 0.0)
            # ASR 感知（P05）：IC 只在测到时才带这几个键。存在性要单独判断 ——
            # 落成默认值（False / "" / 0.0）会把「这段没测出来」伪装成
            # 「测了，没噪声」，语义正好反了。
            if event.get("noise_burst_detected") is not None:
                msg.noise_burst_detected = bool(event["noise_burst_detected"])
            if event.get("noise_burst_type"):
                msg.noise_burst_type = str(event["noise_burst_type"])
            if event.get("snr_db") is not None:
                msg.snr_db = float(event["snr_db"])
            for server in self._mcp_servers:
                msg.mcp_servers.servers.add(**server)
            try:
                self._queue.put_nowait(msg)
                self._pending_acks[(ci, epoch, act)].append(kind)
            except queue.Full:
                self.error = "Agent request queue full"
                self._failed = True
                self._active = None
                self._finish_requests()
                if self._call is not None:
                    self._call.cancel()
                else:
                    self._emit({"type": "agent.disconnect", "error": self.error})
                return False
            return True

    def _receive(self) -> None:
        from agent_protocol import agent_pb2 as pb
        try:
            for event in self._call:
                with self._lock:
                    if (event.session_id != self.session_id or event.ci != self._active
                            or event.epoch != self._epoch or self._closed):
                        continue
                    envelope = {"ci": event.ci, "epoch": event.epoch}
                    payload = event.WhichOneof("payload")
                    if payload == "text":
                        if not self._accept_text or self._text_done:
                            continue
                        chunk = event.text
                        kind = pb.TextChunk.Kind.Name(chunk.kind).lower()
                        if kind in ("feedback", "answer"):
                            self._emit({"type": "agent.text", **envelope, "kind": kind,
                                        "stream_id": chunk.stream_id or event.ci,
                                        "text": chunk.text, "is_final": chunk.is_final})
                    elif payload == "state":
                        state = event.state
                        status = pb.AgentState.Status.Name(state.status) if state.HasField("status") else None
                        pending = state.session_end_pending if state.HasField("session_end_pending") else None
                        # Control actions reuse ci/epoch: reject late generation state after cancel.
                        if not self._accept_text:
                            status = status if status == "IDLE" else None
                            pending = False if pending is False else None
                        if status is not None or pending is not None:
                            self._emit({"type": "agent.state", **envelope, "status": status,
                                        "session_end_pending": pending})
                    elif payload == "ack":
                        key = (event.ci, event.epoch, event.act)
                        pending = self._pending_acks.get(key)
                        action_type = pending.popleft() if pending else None
                        if pending is not None and not pending:
                            self._pending_acks.pop(key, None)
                        if action_type == "answer" and not event.ack.accepted:
                            self._accept_text = False
                            self._text_done = True
                            self._revision += 1
                        self._emit({"type": "agent.ack", **envelope, "act": pb.AgentAction.Type.Name(event.act), "action_type": action_type, "accepted": event.ack.accepted,
                                    "reason": event.ack.reason})
                    elif payload == "error":
                        self._accept_text = False
                        self._text_done = True
                        self._revision += 1
                        self._emit({"type": "agent.error", **envelope, "error": event.error.message})
                    elif payload == "done":
                        self._text_done = True
                        # Generation completion says nothing about real playback.
                        self._emit({"type": "agent.done", **envelope})
        except Exception as exc:
            self.error = str(exc)
        finally:
            with self._lock:
                self._failed = True
                self._active = None
                self._finish_requests()
                if not self._closed:
                    self._emit({"type": "agent.disconnect", "error": self.error})

    def is_current(self, event: dict, *, text: bool = False) -> bool:
        with self._lock:
            return (not self._closed and not self._failed
                    and event.get("connection_id", self.connection_id) == self.connection_id
                    and event.get("ci") == self._active and event.get("epoch") == self._epoch
                    and event.get("revision", self._revision) == self._revision
                    and (not text or self._accept_text))

    def close(self, timeout: float = 2.0) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._active = None
            self._finish_requests()
            call, channel, thread = self._call, self._channel, self._thread
        if call is not None:
            call.cancel()
        if channel is not None:
            channel.close()
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout)
