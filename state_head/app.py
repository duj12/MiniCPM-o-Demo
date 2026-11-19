"""state head 的 HTTP 门面 —— ``POST /v1/state`` / ``GET /health``。

编排侧（``orchestrator/omni/state_client.py``）在每次"该更新视觉状态"的
那一拍同时起两条：Qwen3-Omni 的短文本描述、以及这里的 typed-question
推理。两者**互不等待**，谁先到谁先写状态。

## 请求 / 响应

请求::

    POST /v1/state
    {
      "frames_b64": ["<最新帧>", "<1 秒前的帧>"],   # 裸 base64 或 data URL
      "fps": 1.0,                                  # 帧间隔（给 video 打时间戳）
      "stage": "turn",                             # turn | full
      "tier2": false,                              # 已作废（动作是一题 act.type）
      "session_id": "abc",                         # 只用于同会话互斥
      "slots": ["attn.gaze", "beh.activity"],      # 白名单；[]=问全套，缺省=用 STATE_SLOTS
      "transcript": "",                            # 预留：V1 不参与推断
      "asr_emotion": null,                         # 预留：融合在编排侧做
      "asr_emotion_prob": null
    }

响应::

    {
      "schema_version": "2.1",
      "state": { ...E2 §4.3... },
      "confidence": {"ATTENTION.orientation": 0.71, ...},
      "latency_ms": 312, "model": "OneJev1.6-0.8B",
      "backend": "onejev", "stage": "turn", "n_questions": 15
    }

## 为什么 ``transcript`` / ``asr_emotion`` 收下却不用

EMOTION 的晚融合在**编排侧**做（``orchestrator/omni/state_schema.py``）——
只有那里知道 ASR 段末的时序（``2pass-offline`` 才带情绪，且不能无限沿用）。
本服务只负责 ``visual``：题目里明确禁止参考"他正在说什么"。把这两个字段
收进来是为了将来做"端到端一次性融合"时不必改协议；现在它们只回显在
``echo`` 里，方便抓包核对。
"""
from __future__ import annotations

import asyncio
import base64
import io
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .assemble import assemble, flatten_confidence, project_slots
from .backends import BackendError, SystemOneBackend, build_media, warmup
from .questions import build_questions
from .scheduler import Capacity, SessionGate, parse_urls

logger = logging.getLogger("state_head")
logging.basicConfig(
    level=os.environ.get("STATE_LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s %(message)s")


def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() not in ("0", "false", "no", "off", "")


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _parse_slots(raw: Optional[str]) -> frozenset:
    """``STATE_SLOTS`` → 题集 id 集合。空 / ``all`` / ``*`` ⇒ 空集合（问全套）。

    与编排侧 ``orchestrator/omni/state_schema.parse_slots`` **同一口径**；
    两边都为空时就是老行为（E2 出全部枚举槽位、文本侧走短文本）。
    """
    items = {s.strip() for s in (raw or "").split(",") if s.strip()}
    if not items or items & {"all", "*"}:
        return frozenset()
    return frozenset(items)


class Settings:
    """启动期读一次的环境（全部在 compose / .env 里可调）。"""

    def __init__(self) -> None:
        self.backend_name = os.environ.get("STATE_BACKEND", "onejev")
        self.urls = parse_urls(os.environ.get("STATE_BACKEND_URLS"))
        self.timeout_s = _env_float("STATE_TIMEOUT_S", 3.0)
        self.debias = _env_int("STATE_DEBIAS", 1)
        self.max_concurrency = _env_int("STATE_MAX_CONCURRENCY", 64)
        #: 发前缩图的长边上限（0 = 不缩）。显存与延迟的主旋钮之一：
        #: 640px 比原始 1280×720 少一半以上的视觉 token。
        self.max_px = _env_int("STATE_MAX_MEDIA_PX", 640)
        self.jpeg_quality = _env_int("STATE_JPEG_QUALITY", 85)
        self.person_p = _env_float("STATE_PERSON_VISIBLE_P", 0.5)
        self.action_p = _env_float("STATE_ACTION_P", 0.5)
        self.enum_min_conf = _env_float("STATE_ENUM_MIN_CONF", 0.0)
        self.max_events = _env_int("STATE_ACTION_MAX_EVENTS", 2)
        self.default_tier2 = _env_bool("STATE_TIER2", False)
        self.lighting = _env_bool("STATE_INCLUDE_LIGHTING", True)
        self.warmup = _env_bool("STATE_WARMUP", True)
        self.model_dir = os.environ.get("STATE_MODEL_DIR", "onejev-1.6-0.8b")
        #: 场景/任务提示（模型卡 ``state.task``）。空 = 不发，行为不变。
        self.context = os.environ.get("STATE_CONTEXT", "")
        #: 槽位白名单（``STATE_SLOTS``）：非空则**只问这些**，题数与延迟同步
        #: 下降。空 / ``all`` / ``*`` = 问全套。默认那 4 个零样本实测可靠的
        #: 槽位由编排侧的 ``ORCH_STATE_SLOTS`` 决定，这里两处必须一致 ——
        #: 不一致的症状是「文本以为 E2 会答某一项，E2 却没问」，那一项就空了。
        self.slots = _parse_slots(os.environ.get("STATE_SLOTS", ""))


SETTINGS = Settings()
GATE = SessionGate()
CAPACITY = Capacity(SETTINGS.max_concurrency)
BACKEND: Optional[SystemOneBackend] = None


def _downscale(b64: str, max_px: int, quality: int) -> str:
    """把一帧缩到长边 ``max_px``（等比）。失败原样返回 —— 不能因为缩图挂掉。"""
    if max_px <= 0:
        return b64
    raw = b64.split(",", 1)[-1] if b64.startswith("data:") else b64
    try:
        from PIL import Image

        data = base64.b64decode(raw, validate=False)
        img = Image.open(io.BytesIO(data))
        w, h = img.size
        longest = max(w, h)
        if longest <= max_px:
            return b64
        scale = max_px / float(longest)
        img = img.convert("RGB").resize(
            (max(1, int(w * scale)), max(1, int(h * scale))),
            Image.BILINEAR)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    except Exception as exc:  # noqa: BLE001
        logger.debug("[state_head] 缩图失败（按原图继续）: %s", exc)
        return b64


class StateRequest(BaseModel):
    frames_b64: List[str] = Field(default_factory=list)
    fps: float = 1.0
    stage: str = "turn"
    tier2: Optional[bool] = None
    session_id: str = ""
    #: 场景/任务提示（对应模型卡里的 ``state.task``）。空则用 ``STATE_CONTEXT``。
    context: str = ""
    #: 槽位白名单（题集 id）。编排侧 `StateClient` **每次都发**：``[]`` =
    #: 明确要求问全套；``None`` = 没意见，用服务端 ``STATE_SLOTS``。
    slots: Optional[List[str]] = None
    # 预留字段（见模块文档）：V1 不参与推断，只回显
    transcript: str = ""
    asr_emotion: Optional[str] = None
    asr_emotion_prob: Optional[float] = None


class StateResponse(BaseModel):
    schema_version: str = "2.1"
    state: Dict[str, Any]
    confidence: Dict[str, float] = Field(default_factory=dict)
    latency_ms: int = 0
    model: str = ""
    backend: str = ""
    stage: str = "turn"
    n_questions: int = 0
    echo: Dict[str, Any] = Field(default_factory=dict)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global BACKEND
    logger.info("[state_head] backend=%s urls=%s max_px=%d tier2=%s slots=%s",
                SETTINGS.backend_name, SETTINGS.urls, SETTINGS.max_px,
                SETTINGS.default_tier2,
                ",".join(sorted(SETTINGS.slots)) or "all")
    BACKEND = SystemOneBackend(SETTINGS.urls, timeout_s=SETTINGS.timeout_s,
                               debias=SETTINGS.debias,
                               max_connections=SETTINGS.max_concurrency)
    if SETTINGS.warmup:
        # 后台预热：不阻塞 /health 就绪，否则 compose 的 healthcheck 会
        # 在模型加载的几十秒里连续失败、把 restart 策略点着。
        asyncio.create_task(warmup(BACKEND))
    try:
        yield
    finally:
        if BACKEND is not None:
            await BACKEND.aclose()
        await GATE.close()


app = FastAPI(title="Omni State Head", version="2.1", lifespan=lifespan)


_ALIVE_CACHE: Dict[str, Any] = {"at": 0.0, "ok": False}


@app.get("/health")
async def health() -> JSONResponse:
    if BACKEND is None:
        return JSONResponse({"ok": False, "reason": "backend not ready"},
                            status_code=503)
    # 探活结果缓存 3 秒：docker 的 healthcheck 每 15s 一次，但编排侧的 probe
    # 可能更密；每来一次就同步打一遍 flide 没必要，也会占用连接池。
    now = time.monotonic()
    if now - _ALIVE_CACHE["at"] > 3.0:
        _ALIVE_CACHE["ok"] = await BACKEND.alive()
        _ALIVE_CACHE["at"] = now
    alive = _ALIVE_CACHE["ok"]
    body = {
        # ok 只在 flide 真的能应答时为真 —— 模型没加载时报 false 且回 503，
        # docker 的 `curl -sf` 因此会判 unhealthy，编排侧 probe 也会退回 E0，
        # 而不是每轮白等一个超时。
        "ok": alive,
        "reason": "" if alive else "flide 端点不可达（模型未加载？）",
        "model": BACKEND.model_name or SETTINGS.model_dir,
        "backend": SETTINGS.backend_name,
        "endpoints": SETTINGS.urls,
        "max_px": SETTINGS.max_px,
        # 白名单为空表示"问全套"。部署时它应与编排侧的 `ORCH_STATE_SLOTS`
        # 一致，否则会出现"文本以为 E2 会答某一项、E2 却没问"的空洞。
        "slots": sorted(SETTINGS.slots),
        "inflight": CAPACITY.inflight,
        "peak": CAPACITY.peak,
        "stats": dict(BACKEND.stats, **GATE.stats),
    }
    return JSONResponse(body, status_code=200 if alive else 503)


@app.get("/v1/questions")
async def questions(stage: str = "turn", tier2: bool = None) -> Dict[str, Any]:
    """当前生效的题集（排错用：延迟超预算时先看这里数了几题）。"""
    q = build_questions(stage=stage,
                        tier2=SETTINGS.default_tier2 if tier2 is None else tier2,
                        include_lighting=SETTINGS.lighting,
                        slots=SETTINGS.slots)
    return {"stage": stage, "count": len(q),
            "slots": sorted(SETTINGS.slots), "questions": q}


@app.post("/v1/state", response_model=StateResponse)
async def state(req: StateRequest) -> StateResponse:
    if BACKEND is None:
        raise HTTPException(status_code=503, detail="backend not ready")
    if not req.frames_b64:
        # 没有画面就别让模型编（与编排侧的帧新鲜度闸同一口径）。返回 422
        # 而不是空状态：调用方据此**保留上一份 E2**。
        raise HTTPException(status_code=422, detail="frames_b64 is empty")

    stage = "full" if str(req.stage).lower() == "full" else "turn"
    tier2 = SETTINGS.default_tier2 if req.tier2 is None else bool(req.tier2)
    # 请求给了就听请求（`[]` = 问全套），没给才用服务端默认 —— 编排侧是
    # 唯一的白名单写者，两边的 `STATE_SLOTS` 因此不会漂。
    slots = SETTINGS.slots if req.slots is None else frozenset(
        s.strip() for s in req.slots if s and s.strip())
    if slots & {"all", "*"}:
        slots = frozenset()
    questions = build_questions(stage=stage, tier2=tier2,
                                include_lighting=SETTINGS.lighting,
                                slots=slots)
    frames = [_downscale(f, SETTINGS.max_px, SETTINGS.jpeg_quality)
              for f in req.frames_b64]
    media_state, media = build_media(
        frames, fps=req.fps, context=req.context or SETTINGS.context)

    started = time.monotonic()

    async def _run():
        async with CAPACITY:
            return await BACKEND.systemone(media_state, questions, media)

    try:
        answers = await GATE.run(req.session_id or "", _run)
    except BackendError as exc:
        logger.warning("[state_head] 推理失败: %s", exc)
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except asyncio.TimeoutError as exc:
        raise HTTPException(status_code=504, detail="inference timeout") from exc

    e2 = assemble(
        answers, stage=stage, tier2=tier2,
        has_scene="env.scene" in questions,
        person_p=SETTINGS.person_p, action_p=SETTINGS.action_p,
        enum_min_conf=SETTINGS.enum_min_conf, max_events=SETTINGS.max_events,
    )
    if slots:
        # 只返回**真的问过**的槽位（"人不在画面"那种整份 null 不裁）。
        e2 = project_slots(e2, questions)
    latency_ms = int((time.monotonic() - started) * 1000)
    conf = flatten_confidence(e2)
    logger.info("[VLM state] stage=%s latency=%dms slots=%d conf=%d 键",
                stage, latency_ms, len(questions), len(conf))
    return StateResponse(
        state=e2, confidence=conf, latency_ms=latency_ms,
        model=BACKEND.model_name or SETTINGS.model_dir,
        backend=SETTINGS.backend_name, stage=stage,
        n_questions=len(questions),
        echo={"transcript": req.transcript[:200],
              "asr_emotion": req.asr_emotion,
              "asr_emotion_prob": req.asr_emotion_prob},
    )
