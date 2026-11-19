"""推理后端 —— **只做转发，不做排队**。

## 为什么只有一个 HTTP 后端

Flide 的 ``/v1/systemone`` 是 Jev 家族的统一 API：OneJev 直接跑，其它
决策模型只要 Flide 能加载就能换 —— 换模型只是换 ``--model``（以及可能的
``--format``），**代码一行不动**。所以本模块把"后端"抽象成"若干个
``/v1/systemone`` 端点"，而不是"OneJev 一套、OmniJev 一套"。

> ⚠️ 2026-11-19 核实：``tinnel123/OmniJev-2B`` 是 **Qwen3.5-2B 上的
> PEFT/LoRA 适配器**（model card 的 library_name=peft），没有任何推理
> 代码或 serving 说明，PyPI 上的 ``mso`` 是无关包 —— 计划里
> ``mso.infer.MSO1(ckpt, base)`` 这条路**不存在**。要用它就得先确认 Flide
> 能否以 Qwen3.5 家族 + 适配器的方式加载；在那之前 ``STATE_BACKEND``
> 只有 ``onejev`` 是可信的。

## 并发：不在这一层排队

Flide 自己会**跨请求合批**（``--batch-window-ms`` / ``--planners`` /
``--frontends``）。如果在客户端加一个 ``Semaphore`` 或用一把锁按顺序发，
就把它的合批彻底废掉、把并发变成排队。所以：

* 一个后端对应 **N 个端点**（``STATE_BACKEND_URLS``），按请求**轮转**
  （round-robin）；
* 每次调用都是一条独立的协程，共用同一个 ``httpx.AsyncClient``
  （连接池有上限，但那是连接数上限，不是推理闸）；
* 端点失败时**换下一个**重试一次 —— 单实例崩了不会整条链断。
"""
from __future__ import annotations

import asyncio
import itertools
import logging
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)


class BackendError(RuntimeError):
    """所有端点都没能给出可用答案。"""


def normalize_frames(frames_b64: List[str]) -> List[str]:
    """把帧统一成 ``data:image/jpeg;base64,…`` 形式。

    编排侧发的是**裸 base64**（与 ``omni/client.py`` 的 ``_pending_frame``
    同源），而 Flide 的 ``media.data`` 两种都收。统一加前缀是为了让日志与
    抓包一眼看出是图片。
    """
    out: List[str] = []
    for f in frames_b64 or []:
        if not f:
            continue
        s = str(f)
        out.append(s if s.startswith("data:") else f"data:image/jpeg;base64,{s}")
    return out


def build_media(frames_b64: List[str], fps: float = 1.0,
                context: str = "") -> tuple:
    """帧列表 → ``(state 片段, media 列表)``。

    两帧及以上用 **video**（Flide 会按 ``fps`` 给每帧打时间戳，motion /
    duration / 短时动作都靠它）；只有一帧退回 **image**。

    ``state`` 必须用 ``<video:1>`` / ``<image:1>`` 占位符指过来 —— 否则
    图片根本进不了模型的上下文（docs: "the state marks where each one
    belongs"）。
    """
    frames = normalize_frames(frames_b64)
    if not frames:
        raise BackendError("no frames")
    # ``context`` 是给模型的场景/任务提示（模型卡示例里的 ``state.task``）。
    # 不写就只是一段没头没尾的画面，零样本下动作与情绪类问句更容易判成 neutral。
    extra = {"context": context} if context else {}
    if len(frames) >= 2:
        return ({"clip": "<video:1>", **extra},
                [{"type": "video", "fps": float(fps or 1.0), "frames": frames}])
    return ({"frame": "<image:1>", **extra},
            [{"type": "image", "data": frames[0]}])


class SystemOneBackend:
    """若干 ``/v1/systemone`` 端点上的并发转发器。"""

    def __init__(self, urls: List[str], timeout_s: float = 3.0,
                 debias: int = 1, max_connections: int = 64) -> None:
        urls = [u.rstrip("/") for u in (urls or []) if u]
        if not urls:
            raise BackendError("STATE_BACKEND_URLS 为空 —— 没有可用的推理端点")
        self.urls = urls
        self.timeout_s = float(timeout_s)
        self.debias = max(1, int(debias or 1))
        self._rr = itertools.count()
        # ⚠️ 连接池上限只是"别把 fd 打爆"，不是推理闸：Flide 那边该批的
        #    还是会批，该并发的还是会并发。
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(self.timeout_s),
            limits=httpx.Limits(max_connections=max_connections,
                                max_keepalive_connections=max_connections),
        )
        self.model_name = ""
        self.stats = {"requests": 0, "errors": 0, "failovers": 0}

    async def aclose(self) -> None:
        await self._client.aclose()

    async def alive(self, timeout_s: float = 1.5) -> bool:
        """任一端点可达即算活（``GET /stats``）。

        ``/health`` 靠它把"门面起来了但模型没加载"如实报出来 —— 否则编排侧
        会以为能用，每轮都白等一个超时。
        """
        for url in self.urls:
            try:
                resp = await self._client.get(f"{url}/stats",
                                              timeout=httpx.Timeout(timeout_s))
                if resp.status_code < 500:
                    return True
            except Exception:  # noqa: BLE001
                continue
        return False

    def _endpoint(self, skip: set) -> str:
        n = len(self.urls)
        for _ in range(n):
            idx = next(self._rr) % n
            if idx not in skip:
                return self.urls[idx]
        return self.urls[0]

    async def systemone(self, state: Any, questions: Dict[str, Any],
                        media: Optional[List[Dict[str, Any]]] = None,
                        ) -> Dict[str, Any]:
        """POST 一次，返回 ``answers`` 字典。**失败会换端点重试一次**。"""
        body: Dict[str, Any] = {"state": state, "questions": questions}
        if media:
            body["media"] = media
        params = {"debias": self.debias} if self.debias > 1 else None

        tried: set = set()
        last_exc: Optional[BaseException] = None
        for attempt in range(len(self.urls)):
            url = self._endpoint(tried)
            tried.add(self.urls.index(url))
            target = f"{url}/v1/systemone"
            self.stats["requests"] += 1
            try:
                resp = await self._client.post(target, json=body, params=params)
                if resp.status_code == 422:
                    # 选项数超过模型槽位 / 请求超长 —— 重试没有意义，但要
                    # 把服务端的原话带到日志（它会说"多长"）。
                    raise BackendError(f"422 from {target}: {resp.text[:400]}")
                resp.raise_for_status()
                data = resp.json()
            except BackendError:
                raise
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                self.stats["errors"] += 1
                if len(self.urls) > 1:
                    self.stats["failovers"] += 1
                    logger.warning("[state_head] %s 失败（%s），换下一个端点",
                                   target, exc)
                continue
            answers = data.get("answers")
            if not isinstance(answers, dict):
                last_exc = BackendError(f"{target} 没有 answers: {str(data)[:200]}")
                continue
            if data.get("model"):
                self.model_name = str(data["model"])
            return answers
        raise BackendError(f"所有端点都失败（{len(self.urls)} 个）: {last_exc}")


async def warmup(backend: SystemOneBackend, timeout_s: float = 60.0) -> bool:
    """首帧预热 —— 让 CUDA graph / 权重在第一个真请求之前就位。

    只是**尽力而为**：失败不算错误（模型目录缺失时``/health``会报出来），
    但如果成功，第一次真实请求就不会把 2~3 秒的加载时间算进 P95。
    """
    try:
        # 用 Pillow 现造一张 32×32 的图，而不是内嵌一段 base64 —— 内嵌的
        # 常量一旦手抄错，预热会静默失败，而失败又"不影响启动"，没人会发现。
        import io

        from PIL import Image

        buf = io.BytesIO()
        Image.new("RGB", (32, 32), (128, 128, 128)).save(buf, format="JPEG")
        import base64 as _b64

        dummy = "data:image/jpeg;base64," + _b64.b64encode(buf.getvalue()).decode()
    except Exception as exc:  # noqa: BLE001
        logger.warning("[state_head] 预热构造测试帧失败: %s", exc)
        return False
    try:
        await asyncio.wait_for(backend.systemone(
            state={"frame": "<image:1>"},
            questions={"ok": {"type": "noul",
                              "instructions": "the image is not empty"}},
            media=[{"type": "image", "data": dummy}],
        ), timeout=timeout_s)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("[state_head] 预热未成功（不影响启动）: %s", exc)
        return False
