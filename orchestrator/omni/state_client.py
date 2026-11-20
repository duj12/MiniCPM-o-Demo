"""state head 的 HTTP 客户端 —— **永不抛异常，只返回 None**。

## 为什么放在 ``create_task`` 里跑

state head 的目标延迟是 250~450ms，而一轮对话的 ``on_answer`` 窗口只有几十
毫秒。它**绝不能**出现在任何 await 链上：会话侧在触发文本描述的那一拍顺手
起一条协程，成不成功都与回复无关。所以本模块的契约只有一条：

    ``post_state()`` 永远不抛异常；拿不到 = ``None``，调用方跳过这一轮。

## 熔断

state head 挂了会让每一轮都白等一个超时（0.8s × 每秒一次 = 白烧 CPU 与
连接）。连续失败 ``fail_threshold`` 次后熔断 ``cooldown_s`` 秒，期间直接
返回 ``None``、一个请求都不发。这期间行为与 ``ORCH_STATE_HEAD=0`` 完全一致。
半开：冷却结束后放一个请求出去，成功就恢复。
"""
from __future__ import annotations

import base64
import io
import logging
import time
from typing import Any, Dict, List, Optional, Sequence

import httpx

logger = logging.getLogger(__name__)

#: 编排侧缩图只影响带宽（容器侧 `STATE_MAX_MEDIA_PX` 才是显存/延迟的主旋钮）。
#: Pillow 是可选依赖：装了就顺手缩，没装就原样发。
_PIL_OK: Optional[bool] = None
_PIL_WARNED = False


def _resize_jpeg(b64: str, max_px: int, quality: int = 85) -> str:
    """长边缩到 ``max_px``；任何异常都原样返回（不能让缩图失败拖垮链路）。"""
    if max_px <= 0 or not b64:
        return b64
    global _PIL_OK, _PIL_WARNED
    if _PIL_OK is None:
        try:
            import PIL  # noqa: F401
            _PIL_OK = True
        except Exception:  # noqa: BLE001
            _PIL_OK = False
    if not _PIL_OK:
        if not _PIL_WARNED:
            _PIL_WARNED = True
            logger.info("[VLM state] 未安装 Pillow，跳过编排侧缩图"
                        "（容器侧 STATE_MAX_MEDIA_PX 仍然生效）")
        return b64
    try:
        from PIL import Image

        raw = b64.split(",", 1)[-1] if b64.startswith("data:") else b64
        img = Image.open(io.BytesIO(base64.b64decode(raw, validate=False)))
        w, h = img.size
        longest = max(w, h)
        if longest <= max_px:
            return b64
        scale = max_px / float(longest)
        img = img.convert("RGB").resize(
            (max(1, int(w * scale)), max(1, int(h * scale))), Image.BILINEAR)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        return base64.b64encode(buf.getvalue()).decode()
    except Exception as exc:  # noqa: BLE001
        logger.debug("[VLM state] 缩图失败（原样发送）: %s", exc)
        return b64


class StateClient:
    """``POST /v1/state`` 的瘦客户端。"""

    def __init__(self, url: str, timeout_s: float = 0.8, n_frames: int = 2,
                 max_px: int = 640, fail_threshold: int = 5,
                 cooldown_s: float = 30.0, slots: str = "") -> None:
        self.url = (url or "").rstrip("/")
        self.timeout_s = float(timeout_s)
        self.n_frames = max(1, int(n_frames))
        self.max_px = int(max_px)
        self.fail_threshold = max(1, int(fail_threshold))
        self.cooldown_s = float(cooldown_s)
        #: 槽位白名单（`ORCH_STATE_SLOTS`）—— **每次请求都带给服务端**，
        #: 这样"文本省哪几项"与"E2 答哪几项"只有一个写者，不会因为容器
        #: 的 `STATE_SLOTS` 没跟着改而出现"没人回答的槽位"。
        #: 空列表 = 问全套（显式发 `[]`，与服务端 env 默认值区分开）。
        self.slots: List[str] = [s.strip() for s in str(slots or "").split(",")
                                 if s.strip()]
        if set(self.slots) & {"all", "*"}:
            self.slots = []
        self._client: Optional[httpx.AsyncClient] = None
        self._fails = 0
        self._open_until = 0.0
        #: 统计（日志/健康检查用）
        self.stats = {"ok": 0, "fail": 0, "timeout": 0, "skipped": 0,
                      "latency_ms_sum": 0}

    # ------------------------------------------------------------------ #

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout_s))
        return self._client

    def tripped(self) -> bool:
        return time.monotonic() < self._open_until

    async def aclose(self) -> None:
        if self._client is not None:
            try:
                await self._client.aclose()
            except Exception:  # noqa: BLE001
                pass
            self._client = None

    # ------------------------------------------------------------------ #

    async def post_state(self, frames_b64: Sequence[str], stage: str = "turn",
                         session_id: str = "", transcript: str = "",
                         asr_emotion: Optional[str] = None,
                         asr_emotion_prob: Optional[float] = None,
                         tier2: Optional[bool] = None,
                         ) -> Optional[Dict[str, Any]]:
        """跑一次 state head。**任何失败都返回 None**，绝不抛。"""
        if not frames_b64:
            return None
        if self.tripped():
            self.stats["skipped"] += 1
            return None

        frames = [_resize_jpeg(f, self.max_px)
                  for f in list(frames_b64)[-self.n_frames:]]
        body: Dict[str, Any] = {
            "frames_b64": frames,
            "stage": stage,
            "session_id": session_id,
            "transcript": transcript or "",
            "asr_emotion": asr_emotion,
            "asr_emotion_prob": asr_emotion_prob,
            # **总是**发（哪怕是空列表）：服务端据此区分"调用方要求全套"
            # 与"调用方没意见，用我的 STATE_SLOTS 默认值"。
            "slots": list(self.slots),
        }
        if tier2 is not None:
            body["tier2"] = bool(tier2)

        t0 = time.monotonic()
        try:
            resp = await self._http().post(f"{self.url}/v1/state", json=body)
            if resp.status_code != 200:
                self._fail(f"HTTP {resp.status_code}: {resp.text[:160]}")
                return None
            data = resp.json()
        except httpx.TimeoutException:
            self.stats["timeout"] += 1
            self._fail(f"超时 {self.timeout_s}s")
            return None
        except Exception as exc:  # noqa: BLE001
            self._fail(repr(exc))
            return None

        state = data.get("state") if isinstance(data, dict) else None
        if not isinstance(state, dict) or not state:
            self._fail(f"响应里没有 state: {str(data)[:160]}")
            return None

        self._ok()
        dt_ms = int((time.monotonic() - t0) * 1000)
        self.stats["latency_ms_sum"] += dt_ms
        return {
            "state": state,
            "confidence": data.get("confidence") or {},
            "latency_ms": int(data.get("latency_ms") or dt_ms),
            "model": str(data.get("model") or ""),
            "n_questions": int(data.get("n_questions") or 0),
        }

    # ------------------------------------------------------------------ #

    def _ok(self) -> None:
        self.stats["ok"] += 1
        if self._fails:
            logger.info("[VLM state] 恢复正常（此前连续失败 %d 次）", self._fails)
        self._fails = 0
        self._open_until = 0.0

    def _fail(self, why: str) -> None:
        self.stats["fail"] += 1
        self._fails += 1
        if self._fails == 1:
            # 只在第一次与即将熔断时说话：每轮都打会把日志淹没。
            logger.warning("[VLM state] 请求失败（后续静默计数）: %s", why)
        if self._fails >= self.fail_threshold:
            self._open_until = time.monotonic() + self.cooldown_s
            logger.warning("[VLM state] 连续失败 %d 次，熔断 %.0fs —— "
                           "期间退回纯文本描述", self._fails, self.cooldown_s)
            self._fails = 0
