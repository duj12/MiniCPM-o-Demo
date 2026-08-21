"""MiniCPMO45 推理 Worker

每个 Worker 占用一张 GPU，持有一个 UnifiedProcessor 实例，
提供 Chat (HTTP) / Streaming (WebSocket) / Duplex (WebSocket) 三种推理 API。

启动方式：
    cd /user/sunweiyue/lib/swy-dev/minicpmo45_service
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. .venv/base/bin/python worker.py \\
        --port 10031 \\
        --model-path /path/to/base_model \\
        --pt-path /path/to/custom.pt \\
        --ref-audio-path /path/to/ref.wav
"""

import json
import os
import time
import asyncio
import argparse
import logging
import base64
import threading
import subprocess
from typing import Optional, List, Dict, Any

import numpy as np
import uvicorn
import websockets
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from worker_state import WorkerState, WorkerStatus
from runtime.protocol import DEFAULT_WORKER_CAPABILITIES
from runtime.session import BackendRuntimeSession

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("worker")

# ============ 请求/响应模型 ============

class WorkerHealthResponse(BaseModel):
    """健康检查响应"""
    status: str
    worker_status: WorkerStatus
    gpu_id: int
    model_loaded: bool
    current_ticket_id: Optional[str] = None
    total_requests: int = 0
    avg_inference_time_ms: float = 0.0
    kv_cache_length: int = 0  # 当前 LLM KV cache token 总数
    capabilities: List[str] = Field(default_factory=list)


# ============ FastAPI 应用 ============

worker: Optional[Any] = None

# 启动参数（通过 main() 传入）
WORKER_CONFIG: Dict[str, Any] = {}


class RemoteBackendWorker:
    """Worker host used when inference lives in backend_server.py."""

    def __init__(self, *, backend_server_url: str, gpu_id: int = 0) -> None:
        self.backend_server_url = backend_server_url
        self.gpu_id = gpu_id
        self.processor = None
        self.state = WorkerState(status=WorkerStatus.IDLE)

    def metrics(self) -> Dict[str, Any]:
        return {"backend": "backend_server", "backend_server_url": self.backend_server_url}

    def shutdown(self) -> None:
        return None


def _backend_server_url() -> Optional[str]:
    value = WORKER_CONFIG.get("backend_server_url")
    return str(value).rstrip("/") if value else None


def _input_payload(message: Dict[str, Any]) -> Dict[str, Any]:
    value = message.get("input")
    if isinstance(value, dict):
        return value
    raise RuntimeError("input.append must carry an object `input`")


def _init_payload(message: Dict[str, Any]) -> Dict[str, Any]:
    value = message.get("payload")
    if isinstance(value, dict):
        return value
    raise RuntimeError("session.init must carry an object `payload`")


def _event_payload(event: Any) -> Dict[str, Any]:
    payload = dict(getattr(event, "payload", {}) or {})
    raw_event = payload.get("event")
    if isinstance(raw_event, dict):
        return raw_event
    return payload


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：启动时加载模型"""
    global worker
    config = WORKER_CONFIG

    active_model = os.environ.get("ACTIVE_MODEL") or config.get("active_model", "minicpm")
    backend_server_url = _backend_server_url()

    if active_model == "qwen3omni":
        # Auto-start llama-qwen3omni-server
        from config import get_config as get_app_config
        app_cfg = get_app_config()
        qcfg = app_cfg.backend.qwen3omni

        if not backend_server_url:
            backend_server_url = f"http://{qcfg.backend_host}:{qcfg.backend_port}"

        # Start the C++ server process
        llama_server_bin = config.get("llama_server_bin", "/opt/llama.cpp-omni/bin/llama-qwen3omni-server")
        gguf_model = config.get("model_path") or qcfg.model_path

        if gguf_model and not backend_server_url:
            logger.info("Starting llama-qwen3omni-server: %s", gguf_model)
            gpu_id = int(config.get("gpu_id", 0) or 0)
            server_proc = subprocess.Popen(
                [
                    llama_server_bin,
                    "-m", gguf_model,
                    "--host", qcfg.backend_host,
                    "--port", str(qcfg.backend_port),
                    "-ngl", str(qcfg.n_gpu_layers),
                    "-c", str(qcfg.n_ctx),
                    "--n-parallel", "4",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            # Store for cleanup
            config["_qwen3_proc"] = server_proc

            # Wait for server to be ready
            import httpx
            for i in range(600):
                if server_proc.poll() is not None:
                    logger.error("llama-qwen3omni-server exited prematurely")
                    break
                try:
                    async with httpx.AsyncClient() as client:
                        resp = await client.get(f"{backend_server_url}/health", timeout=2.0)
                        if resp.status_code == 200:
                            logger.info("llama-qwen3omni-server ready after ~%ds", i * 2)
                            break
                except Exception:
                    pass
                await asyncio.sleep(2)
        elif not gguf_model:
            logger.warning("Qwen3-Omni model_path not configured, backend may fail")

        worker = RemoteBackendWorker(
            backend_server_url=backend_server_url,
            gpu_id=int(config.get("gpu_id", 0) or 0),
        )
        logger.info("Worker running as Qwen3-Omni host: %s", backend_server_url)

    elif backend_server_url:
        worker = RemoteBackendWorker(
            backend_server_url=backend_server_url,
            gpu_id=int(config.get("gpu_id", 0) or 0),
        )
        logger.info("Worker running as backend-server runtime host: %s", backend_server_url)
    else:
        from core.processors.backend_factory import create_backend

        worker = create_backend(config)

        # 模型加载是同步操作（~15s），在线程中执行避免阻塞
        await asyncio.to_thread(worker.load_model)

    try:
        yield
    finally:
        logger.info("Worker shutting down")
        if worker is not None:
            await asyncio.to_thread(worker.shutdown)
        # Cleanup Qwen3-Omni server process
        qwen3_proc = config.get("_qwen3_proc")
        if qwen3_proc:
            qwen3_proc.terminate()
            try:
                await asyncio.wait_for(asyncio.to_thread(qwen3_proc.wait), timeout=10)
            except asyncio.TimeoutError:
                qwen3_proc.kill()
                await asyncio.to_thread(qwen3_proc.wait)
            logger.info("llama-qwen3omni-server stopped")


app = FastAPI(title="MiniCPMO45 Worker", lifespan=lifespan)


# ========== 健康检查 ==========

@app.get("/health", response_model=WorkerHealthResponse)
async def health():
    """健康检查"""
    if worker is None:
        return WorkerHealthResponse(
            status="initializing",
            worker_status=WorkerStatus.LOADING,
            gpu_id=0,
            model_loaded=False,
            capabilities=[],
        )

    avg_time = 0.0
    if worker.state.total_requests > 0:
        avg_time = worker.state.total_inference_time_ms / worker.state.total_requests

    worker_metrics = worker.metrics()
    kv_len = int(worker_metrics.get("kv_cache_length", 0) or 0)
    remote_backend_url = _backend_server_url()
    model_loaded = bool(remote_backend_url) or worker.processor is not None
    # 多路并发：有 active session 时报告 busy，否则 idle
    reported_status = WorkerStatus.BUSY_CHAT if worker.state.concurrent_sessions > 0 else WorkerStatus.IDLE

    # Capabilities depend on active model (env overrides config.json)
    # Qwen3-Omni supports streaming video/audio via VAD+TurnSense too, so it
    # advertises the duplex capabilities the gateway routes on (omni_duplex /
    # audio_duplex for mode=video / mode=audio in /v1/realtime).
    active_model = os.environ.get("ACTIVE_MODEL") or WORKER_CONFIG.get("active_model", "minicpm")
    if active_model == "qwen3omni":
        caps = ["chat", "streaming", "half_duplex_audio", "audio_duplex", "omni_duplex"]
    else:
        caps = DEFAULT_WORKER_CAPABILITIES

    return WorkerHealthResponse(
        status="healthy" if model_loaded else "error",
        worker_status=reported_status,
        gpu_id=worker.gpu_id,
        model_loaded=model_loaded,
        current_ticket_id=worker.state.current_ticket_id,
        total_requests=worker.state.total_requests,
        avg_inference_time_ms=avg_time,
        kv_cache_length=kv_len,
        capabilities=caps,
    )



async def _handle_remote_backend_runtime_ws(
    ws: WebSocket,
    *,
    mode: str,
    active_status: WorkerStatus,
    idle_status: WorkerStatus,
) -> None:
    """Bridge worker runtime WebSocket to backend_server.py."""

    backend_url = _backend_server_url()
    if backend_url is None:
        await ws.close(code=1013, reason="Backend server URL is not configured")
        return
    if worker is None:
        await ws.close(code=1013, reason="Worker not ready")
        return

    await ws.accept()
    worker.state.inc_sessions()

    async def _send_runtime_event(event: Any) -> Dict[str, Any]:
        payload = _event_payload(event)
        await ws.send_json(payload)
        return payload

    runtime = BackendRuntimeSession(
        backend_base_url=backend_url,
        mode=mode,
    )
    backend_closed = False
    hd_session: Optional[Any] = None  # half-duplex (VAD+TurnSense) state machine, if enabled

    try:
        first = json.loads(await ws.receive_text())
        first_type = str(first.get("type") or "")
        pending_input: Optional[Dict[str, Any]] = None

        if first_type == "session.init":
            init_params = _init_payload(first)
        elif first_type == "input.append":
            init_params = {"mode": mode}
            pending_input = _input_payload(first)
        else:
            raise RuntimeError(f"first message must initialize or push input, got: {first_type}")

        init_params = dict(init_params)
        init_params.setdefault("mode", mode)

        # VAD+TurnSense 判决：压低 <|listen|> 采样概率，让模型在 TurnSense
        # 触发后倾向 Speak（否则模型自主决策常选 listen 不回复）。
        if (init_params.get("config") or {}).get("turn_decision") == "vad_turnsense":
            cfg = dict(init_params.get("config") or {})
            cfg.setdefault("listen_prob_scale", 0.4)
            init_params["config"] = cfg

        # Retry backend init with backoff to handle race condition where
        # the previous session hasn't been cleaned up yet on the C++ backend.
        MAX_INIT_RETRIES = 3
        INIT_RETRY_DELAYS = [0.5, 1.5, 3.0]

        init_event = None
        for attempt in range(MAX_INIT_RETRIES):
            if attempt > 0:
                await runtime.aclose()
                runtime = BackendRuntimeSession(
                    backend_base_url=backend_url,
                    mode=mode,
                )
            try:
                init_event = await runtime.init(init_params)
                break
            except (websockets.ConnectionClosed, ConnectionError, OSError) as e:
                if attempt < MAX_INIT_RETRIES - 1:
                    logger.warning(
                        "Backend init attempt %d/%d failed, retrying in %.1fs: %s",
                        attempt + 1, MAX_INIT_RETRIES,
                        INIT_RETRY_DELAYS[attempt], e,
                    )
                    await asyncio.sleep(INIT_RETRY_DELAYS[attempt])
                    continue
                raise  # all retries exhausted — outer except handler sends backend_error

        await _send_runtime_event(init_event)

        if pending_input is not None:
            await runtime.push(pending_input)

        # ── Half-duplex (VAD+TurnSense) turn-decision setup ──────────────
        # If the frontend requested turn_decision="vad_turnsense" in session.init,
        # wrap the stream with a VAD+TurnSense state machine that decides when to
        # trigger the model reply (reusing full_duplex transport).
        # VAD 参数可从 session.init payload.config.vad 设置：
        #   { vad_model: "fsmn"|"silero", vad_tail_sil: 600, vad_max_len: 60000,
        #     fsmn_model_dir: "..." }
        from runtime.half_duplex import HalfDuplexConfig, HalfDuplexSession, TurnSenseCfg, VadCfg

        hd_cfg_payload = {}
        vad_cfg_payload = {}
        if init_params.get("config") and isinstance(init_params["config"], dict):
            hd_cfg_payload = init_params["config"].get("turn_decision") or {}
            vad_p = init_params["config"].get("vad") or {}
            if isinstance(vad_p, dict):
                vad_cfg_payload = vad_p
        if hd_cfg_payload == "vad_turnsense":
            vad_cfg = VadCfg(
                vad_model=str(vad_cfg_payload.get("vad_model", "fsmn")),
                fsmn_model_dir=str(vad_cfg_payload.get("fsmn_model_dir", "")),
                vad_tail_sil=int(vad_cfg_payload.get("vad_tail_sil", 600)),
                vad_max_len=int(vad_cfg_payload.get("vad_max_len", 60000)),
                vad_chunk_size=int(vad_cfg_payload.get("vad_chunk_size", 1000)),
            )
            hd_session = HalfDuplexSession(
                config=HalfDuplexConfig(vad=vad_cfg, turnsense=TurnSenseCfg()),
                push=lambda payload: runtime.push(payload),
                interrupt=lambda: runtime.interrupt(),
                send_event=lambda payload: ws.send_json(payload),
            )
            logger.info("half-duplex VAD+TurnSense enabled for session %s "
                        "(vad=%s tail_sil=%dms max_len=%dms)",
                        runtime.session_id, vad_cfg.vad_model,
                        vad_cfg.vad_tail_sil, vad_cfg.vad_max_len)

        async def client_to_backend() -> None:
            nonlocal backend_closed
            async for raw in ws.iter_text():
                msg = json.loads(raw)
                msg_type = str(msg.get("type") or "")

                if msg_type == "input.append":
                    input_payload = _input_payload(msg)
                    if hd_session is not None:
                        await hd_session.feed(
                            audio_b64=input_payload.get("audio", ""),
                            video_frames=input_payload.get("video_frames"),
                        )
                    else:
                        await runtime.push(input_payload)
                    continue

                if msg_type == "session.close":
                    close_event = await runtime.unary("close", {"reason": str(msg.get("reason") or "client_closed")})
                    backend_closed = True
                    close_payload = _event_payload(close_event)
                    if close_payload.get("type") != "session.closed":
                        close_payload = {
                            "type": "session.closed",
                            "session_id": runtime.session_id,
                            "reason": msg.get("reason", "client_closed"),
                        }
                    await ws.send_json(close_payload)
                    await ws.close(code=1000, reason="client_closed")
                    return

                raise RuntimeError(f"unsupported runtime message type: {msg_type}")

        async def backend_to_client() -> None:
            nonlocal backend_closed
            while not backend_closed:
                event = await runtime.pull()
                payload = await _send_runtime_event(event)
                if hd_session is not None:
                    await hd_session.on_backend_event(payload)
                if payload.get("type") == "session.closed":
                    backend_closed = True
                    return

        done, pending = await asyncio.wait(
            [
                asyncio.create_task(client_to_backend()),
                asyncio.create_task(backend_to_client()),
            ],
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in pending:
            task.cancel()
        for task in done:
            try:
                task.result()
            except (WebSocketDisconnect, websockets.exceptions.ConnectionClosed):
                backend_closed = True

    except Exception as exc:
        logger.error("Remote backend runtime failed: error=%s", exc, exc_info=True)
        try:
            await ws.send_json({
                "type": "session.closed",
                "session_id": runtime.session_id,
                "reason": "backend_error",
            })
        except Exception:
            pass
    finally:
        if hd_session is not None:
            try:
                hd_session.shutdown()
            except Exception:
                pass
        try:
            if not backend_closed and runtime.session_id:
                await runtime.unary("close", {"reason": "worker_disconnected"})
        except Exception:
            logger.exception("Remote backend runtime cleanup failed")
        worker.state.dec_sessions()
        try:
            await ws.close()
        except Exception:
            pass


@app.websocket("/v1/worker/chat")
async def worker_chat_runtime_ws(ws: WebSocket):
    """Worker-internal turn-based chat runtime protocol (backend-server only)."""
    await _handle_remote_backend_runtime_ws(
        ws,
        mode="turn_based",
        active_status=WorkerStatus.BUSY_CHAT,
        idle_status=WorkerStatus.IDLE,
    )


# ========== Duplex WebSocket ==========

@app.websocket("/v1/worker/duplex")
async def worker_duplex_runtime_ws(ws: WebSocket):
    """Worker-internal duplex runtime protocol (backend-server only).

    This endpoint is meant for gateway-worker communication and uses runtime
    event payloads instead of page/demo-shaped result messages.
    """
    await _handle_remote_backend_runtime_ws(
        ws,
        mode="full_duplex",
        active_status=WorkerStatus.DUPLEX_ACTIVE,
        idle_status=WorkerStatus.IDLE,
    )


# ============ 缓存状态查询 ==========

@app.get("/cache_info")
async def cache_info():
    """查询当前 Worker 的 KV Cache 状态"""
    if worker is None:
        raise HTTPException(status_code=503, detail="Worker not ready")

    return {
        "status": worker.state.status.value,
        "note": "KV cache state is now tracked by Gateway (cached_hash on WorkerConnection)",
    }


@app.post("/clear_cache")
async def clear_cache():
    """手动清除 KV Cache（重置 Streaming 模型 session）"""
    if worker is None:
        raise HTTPException(status_code=503, detail="Worker not ready")

    worker.reset_half_duplex_session()
    return {"success": True, "message": "Cache cleared"}


# ============ 入口 ============

def main():
    from config import get_config
    cfg = get_config()

    parser = argparse.ArgumentParser(description="MiniCPMO45 Worker")
    parser.add_argument("--port", type=int, default=None, help=f"Worker port (default: from config, base={cfg.worker_base_port})")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host")
    parser.add_argument("--model-path", type=str, default=None, help="Base model path")
    parser.add_argument("--pt-path", type=str, default=None, help="Custom weights path (.pt)")
    parser.add_argument("--ref-audio-path", type=str, default=None, help="Default ref audio path")
    parser.add_argument("--gpu-id", type=int, default=None, help="GPU ID (inferred from port if not set)")
    parser.add_argument("--worker-index", type=int, default=0, help="Worker index (0, 1, 2, ...)")
    parser.add_argument("--duplex-pause-timeout", type=float, default=None, help="Duplex pause timeout (s)")
    parser.add_argument("--backend-server-url", type=str, default=None, help="Remote backend_server.py base URL")
    args = parser.parse_args()

    port = args.port or cfg.worker_port(args.worker_index)
    gpu_id = args.gpu_id if args.gpu_id is not None else args.worker_index

    WORKER_CONFIG.update({
        "model_path": args.model_path or cfg.model.model_path,
        "gpu_id": gpu_id,
        "pt_path": args.pt_path or cfg.model.pt_path,
        "ref_audio_path": args.ref_audio_path or cfg.ref_audio_path,
        "duplex_pause_timeout": args.duplex_pause_timeout or cfg.duplex_pause_timeout,
        "backend_server_url": args.backend_server_url,
        "compile": cfg.compile,
        "chat_vocoder": cfg.chat_vocoder,
        "attn_implementation": cfg.attn_implementation,
        "active_model": cfg.backend.active_model,
    })

    logger.info(f"Starting Worker on port {port}, GPU {gpu_id}")
    # Bump WS max payload from uvicorn's 16 MiB default to 128 MiB so that
    # base64-encoded video attachments (commonly 30-60 MiB after inflation)
    # can be received without the connection being torn down with code 1009.
    uvicorn.run(app, host=args.host, port=port, ws_max_size=128 * 1024 * 1024)


if __name__ == "__main__":
    main()
