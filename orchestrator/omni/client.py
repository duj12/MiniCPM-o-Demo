"""OmniLLM 客户端 —— 包装 MiniCPM-o-Demo 的 ``StreamingChatClient``。

**直接复用 ``StreamingChatClient``，不自己写协议** —— 它已经处理好了
minicpm/qwen3omni 两种后端的话轮语义差异，尤其是最容易写错的那条：

> **qwen3omni 下 ``listen`` delta 是本轮内的分块 prefill 回执，绝不能
> 据此收尾**；只有 ``response.done`` 才是完整回复结束。按 ``listen``
> 收尾会把回复截断在开头。

音频格式：``send_input`` 的 audio 是 **float32 raw base64**（``b64()``
内部 ``.astype(np.float32).tobytes()``），16kHz。

⚠️ 生效后端**必须读 ``session.created.active_model``**，不能读 config.json ——
实测容器里 ``ACTIVE_MODEL=qwen3omni`` 环境变量覆盖了配置文件里的
``"active_model": "minicpm"``。
"""
from __future__ import annotations

import asyncio
import base64
import logging
import sys
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# 复用 streaming_chat_demo.py 的 StreamingChatClient
_DEMO_ROOT = Path(__file__).resolve().parents[2]
if str(_DEMO_ROOT) not in sys.path:
    sys.path.insert(0, str(_DEMO_ROOT))


def _b64_float32(x: np.ndarray) -> str:
    return base64.b64encode(
        np.ascontiguousarray(x, dtype=np.float32).tobytes()
    ).decode()


class OmniClient:
    """一路 OmniLLM 会话。

    把 ``StreamingChatClient`` 的原始事件流翻译成 downstream 事件。
    """

    def __init__(self, url: str, system_prompt: str = "",
                 on_event: Optional[Callable[[dict], None]] = None,
                 connect_timeout: float = 30.0,
                 verify_ssl: bool = False,
                 turn_trigger: str = "turnsense") -> None:
        self.url = url
        self.system_prompt = system_prompt
        self.on_event = on_event
        self.connect_timeout = connect_timeout
        # gateway 默认自签证书；公网部署应置 True 并配正规 CA
        self.verify_ssl = verify_ssl
        # 回复触发方式：
        #   "turnsense" —— 服务端 VAD+TurnSense 判决（默认，模型被服务端
        #                  自动触发；客户端只 force_listen 累积）
        #   "asr"       —— **由我方按 ASR 文本触发**。服务端设
        #                  turn_decision="model" 穿透（不注入自触发），
        #                  客户端持续 force_listen 累积视听上下文，
        #                  收到 ASR final 时补一个 force_listen=False 触发
        self.turn_trigger = turn_trigger

        self.client = None            # StreamingChatClient
        self.backend: Optional[str] = None
        self.session_id: Optional[str] = None
        self.closed = False
        #: 连接**意外断开**（不是我们主动 close）。与 `closed` 区分开：
        #: `closed` = 会话收尾主动关的；`dead` = 对端断了、需要重连或降级。
        self.dead = False
        #: 连续发送失败计数（用于"第一次告警 + 判定断线"，避免刷屏）
        self._send_errors = 0

        # 待发送的视频帧（1fps）。用最新的替换旧的 —— 旧画面没有价值。
        self._pending_frame: Optional[str] = None
        # 音频缓冲：凑够 1s 再发（StreamingChatClient 的节奏）
        self._audio_buf: List[np.ndarray] = []
        self._audio_len = 0
        self._chunk_samples = 16000

        # 统计
        self.frames_sent = 0
        self.audio_sent_s = 0.0
        self.text_deltas = 0
        self.listen_deltas = 0
        self.done_count = 0
        self.triggers = 0        # 我方主动触发的回复数（turn_trigger="asr"）

    # ------------------------------------------------------------------ #

    async def connect(self) -> None:
        from streaming_chat_demo import StreamingChatClient  # type: ignore

        # gateway 用自签证书（config.json 里 gateway 跑 https/wss），
        # 需要跳过校验。公网部署应改为带 CA 的正规校验。
        ssl_ctx = None
        if self.url.startswith("wss://"):
            import ssl as _ssl
            ssl_ctx = _ssl.create_default_context()
            if not self.verify_ssl:
                ssl_ctx.check_hostname = False
                ssl_ctx.verify_mode = _ssl.CERT_NONE

        self.client = StreamingChatClient(self.url, ssl_ctx=ssl_ctx, echo=False)
        await self.client.connect()
        # turn_trigger="asr" 时显式请求 "model"（穿透）—— 服务端就不会包
        # HalfDuplexSession，也就不会自注入触发；触发权完全归我方。
        init_kw = {}
        if self.turn_trigger == "asr":
            init_kw["turn_decision"] = "model"
        ev = await self.client.init(
            mode="full_duplex",
            system_prompt=self.system_prompt or "你是一个实时视频对话助手。",
            **init_kw,
        )
        self.session_id = self.client.session_id
        self.backend = self.client.backend
        logger.info("OmniLLM 已连接: session=%s backend=%s",
                    self.session_id, self.backend)

    def offer_frame(self, jpeg: bytes) -> None:
        """登记一帧待发的视频（1fps）。只保留最新的一帧。"""
        self._pending_frame = base64.b64encode(jpeg).decode()

    async def push_audio(self, seg: np.ndarray) -> None:
        """送一段 AEC 清洗后的音频。内部按 1s 聚合。"""
        if self.closed or self.dead or self.client is None:
            return
        self._audio_buf.append(np.asarray(seg, dtype=np.float32).reshape(-1))
        self._audio_len += self._audio_buf[-1].size
        if self._audio_len < self._chunk_samples:
            return
        audio = np.concatenate(self._audio_buf)[: self._chunk_samples]
        self._audio_buf = []
        self._audio_len = 0
        await self._send(audio)

    async def flush_audio(self) -> None:
        """把不足 1s 的残余也发出去（收尾时用）。"""
        if not self._audio_buf:
            return
        audio = np.concatenate(self._audio_buf)
        self._audio_buf = []
        self._audio_len = 0
        await self._send(audio)

    async def _send(self, audio: np.ndarray) -> None:
        frames = None
        if self._pending_frame is not None:
            frames = [self._pending_frame]
            self._pending_frame = None
            self.frames_sent += 1
        # ⚠️ 所有音频推送一律 force_listen（只 prefill 进 KV，"边听边看"），
        # 由触发源决定何时生成：
        #   · turn_trigger="asr"      → 我方在 ASR final 时补触发
        #   · turn_trigger="turnsense"→ 服务端 VAD+TurnSense 自注入触发
        # 两种模式下客户端推送都不应自己触发回复（否则与触发源打架）。
        try:
            await self.client.send_input(
                audio_b64=_b64_float32(audio), video_frames=frames,
                force_listen=True,
            )
            self.audio_sent_s += audio.size / 16000.0
            self._send_errors = 0
        except Exception as exc:  # noqa: BLE001
            # ⚠️ 只在**第一次**失败时告警，之后静默 —— 否则连接一断就
            #    每秒刷一条，把日志冲垮，反而看不出"什么时候断的"
            #    （真机日志实测：一次会话里刷了 43 条同样的 warning）。
            self._send_errors += 1
            if self._send_errors == 1:
                logger.warning("Omni send_input 失败（后续同类失败不再重复告警）: %s", exc)
            # 触发源那条路要能立刻知道"送不出去"，别再假装成功
            if self._send_errors >= 3 and not self.dead:
                self.dead = True
                logger.warning("Omni 连续 %d 次发送失败 —— 标记连接已断",
                               self._send_errors)

    async def trigger_reply(self, text: str = "") -> bool:
        """主动触发一次回复（``turn_trigger="asr"`` 时用）。

        机制：不带 ``force_listen`` 的 push 会让后端 **decode**（生成），
        而不带它则只 prefill。见 ``runtime/half_duplex.py:639``
        「force_listen intentionally omitted → decode」。

        ``text`` 非空时同时注入文本（给模型一个明确的话轮边界提示）。
        返回是否成功送出。
        """
        if self.closed or self.dead or self.client is None:
            # ⚠️ 连接已断就**别再发了** —— 发出去必然失败，只会刷日志，
            #    而且给调用方一个"已触发"的假象（真正的症状是"识别到了
            #    但永远等不到回复"）。返回 False 让编排服务知道没送出去。
            if self.dead:
                logger.warning("OmniLLM 连接已断，触发被跳过（不回复）")
            return False
        try:
            await self.client.send_input(text=text, force_listen=False)
            self.triggers += 1
            logger.info("ASR 触发回复 #%d%s", self.triggers,
                        f"（附文本 {len(text)} 字）" if text else "")
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("触发回复失败: %s", exc)
            return False

    async def send_text(self, text: str, force_listen: bool = False) -> None:
        """直接注入文本输入（下游 SendToOmni 用）。"""
        if self.closed or self.dead or self.client is None:
            return
        await self.client.send_input(text=text, force_listen=force_listen)

    # ------------------------------------------------------------------ #

    async def recv_loop(self) -> None:
        """接收事件并转成 downstream 事件。

        直接复用 ``StreamingChatClient.handle_event`` 的判定逻辑，只在其
        基础上旁路出 downstream 事件 —— 避免重新实现那套易错的语义。
        """
        if self.client is None:
            return
        orig = self.client.handle_event

        def hooked(ev: dict) -> bool:
            self._emit(ev)
            return orig(ev)

        self.client.handle_event = hooked  # type: ignore[assignment]
        try:
            await self.client.receive_loop()
        finally:
            # ⚠️ **连接断了必须记下来**。早先这里什么都不做：`self.closed`
            #    仍是 False，`send_input` 也不检查连接 —— 于是往一条死连接上
            #    每秒发一次音频，异常只打个 warning，**永远不恢复**。
            #    真机现象：长会话跑几十分钟后，ASR 照常识别、但 OmniLLM
            #    再也不回复（日志里一秒一条 "send_input 失败"）。
            #    这里置位后，`send_input`/`trigger_reply` 会立刻短路，
            #    由编排服务决定是重连还是收尾。
            if not self.closed:
                self.dead = True
                logger.warning(
                    "OmniLLM 接收循环已退出（连接断开）—— "
                    "后续 send_input/触发将直接短路，不会静默堆积")

    def _emit(self, ev: dict) -> None:
        if self.on_event is None:
            return
        t = ev.get("type")
        if t == "turn.turnsense":
            self.on_event(ev)
        elif t == "response.output.delta":
            kind = ev.get("kind")
            if kind == "text":
                self.text_deltas += 1
            elif kind == "listen":
                self.listen_deltas += 1
            self.on_event(ev)
        elif t == "response.done":
            self.done_count += 1
            self.on_event(ev)
        elif t in ("session.closed", "session.error"):
            self.on_event(ev)

    # ------------------------------------------------------------------ #

    async def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        if self.client is not None:
            try:
                await self.client.close(reason="orchestrator_stop")
            except Exception as exc:  # noqa: BLE001
                logger.warning("OmniLLM 关闭异常: %s", exc)
        logger.info(
            "OmniLLM 已关闭: backend=%s 音频=%.1fs 帧=%d text=%d listen=%d done=%d",
            self.backend, self.audio_sent_s, self.frames_sent,
            self.text_deltas, self.listen_deltas, self.done_count,
        )
