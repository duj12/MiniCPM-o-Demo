"""ASR 服务客户端（Fun-ASR 协议，WebSocket + subprotocol binary）。

**协议形态已实测确认**（见 `orchestrator/tests/README.md`）：

  - 首帧 text JSON 配置 → 之后流式发 **PCM int16 / 16kHz / 单声道**
  - 收 JSON text 帧：``2pass-online``（部分）/ ``2pass-offline``（最终）/
    ``turnsense``（条件触发）
  - 结束：发 text ``{"is_speaking": false}``

**实测字段契约**（真实中文语音 6.02s）：

  - ``timestamp``: ``[[519,692,"呃",0.847], ...]`` —— 字级**毫秒** + 每字置信度
  - ``vad_segments``: ``[[290, 5980]]`` —— 毫秒
  - ``confidence``: ``{"avg":0.95652,"token":{"chars":[...],"scores":[...]}}``
  - ``turnsense``: ``probabilities`` 是 **3 元素数组**（不是 dict），
    且**条件触发、不是每段都发**（真实语音可能 0 次）

**部分结果延迟实测 950–1450ms** —— 高于实时决策的理想值，因此 barge-in
不能依赖 ASR（走本地 VAD + 人脸唇动）。
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)

SR = 16000
DEFAULT_CHUNK_MS = 100
WS_MAX_SIZE = 64 * 1024 * 1024


@dataclass
class AsrConfig:
    """ASR 会话配置（对应首帧 JSON）。默认值与 funasr_wss_client.py 对齐。"""

    mode: str = "2pass"                  # offline | online | 2pass
    chunk_size: str = "5,10,5"
    chunk_interval: int = 10
    wav_format: str = "pcm"
    itn: bool = True
    vad_tail_sil: int = 600              # ms
    vad_max_len: int = 20000             # ms
    vad_energy: int = -100
    svs_lang: str = "auto"
    spk_diar: bool = False
    enable_turnsense: bool = True
    turnsense_incomplete_wait_ms: int = 1000
    enable_timestamp: bool = True
    confidence_threshold: float = 0.8
    online_confidence_threshold: float = 0.6

    def to_json(self, wav_name: str) -> str:
        return json.dumps({
            "mode": self.mode,
            "chunk_size": [int(x) for x in self.chunk_size.split(",")],
            "chunk_interval": self.chunk_interval,
            "audio_fs": SR,
            "wav_name": wav_name,
            "wav_format": self.wav_format,
            "is_speaking": True,
            "itn": self.itn,
            "vad_tail_sil": self.vad_tail_sil,
            "vad_max_len": self.vad_max_len,
            "vad_energy": self.vad_energy,
            "svs_lang": self.svs_lang,
            "spk_diar": self.spk_diar,
            "enable_turnsense": self.enable_turnsense,
            "turnsense_incomplete_wait_ms": self.turnsense_incomplete_wait_ms,
            "enable_timestamp": self.enable_timestamp,
            "confidence_threshold": self.confidence_threshold,
            "online_confidence_threshold": self.online_confidence_threshold,
        }, ensure_ascii=False)


class AsrClient:
    """一路 ASR 会话。**不是线程安全的**，请在单个 asyncio loop 里用。

    用法::

        asr = AsrClient(url)
        await asr.connect()
        async with asyncio.TaskGroup() as tg:
            tg.create_task(asr.recv_loop(on_message))
            ...   # asr.push(pcm_float32) / await asr.finish()
    """

    def __init__(self, url: str, config: Optional[AsrConfig] = None,
                 wav_name: str = "orchestrator") -> None:
        self.url = url
        self.config = config or AsrConfig()
        self.wav_name = wav_name

        self.ws = None
        self.closed = False
        self.error: Optional[str] = None

        # 统计
        self.chunks_sent = 0
        self.samples_sent = 0
        self.partials = 0
        self.finals = 0
        self.turnsense_count = 0
        # 时间戳换算：ASR 的毫秒时间戳是**相对流起点**的，
        # 这里记录流起点在会话采样轴上的位置。
        self.stream_t0: int = 0

    # ------------------------------------------------------------------ #

    async def connect(self, timeout: float = 20.0) -> None:
        import websockets
        self.ws = await websockets.connect(
            self.url, subprotocols=["binary"], ping_interval=None,
            max_size=WS_MAX_SIZE,
        )
        await self.ws.send(self.config.to_json(self.wav_name))
        logger.info("ASR 已连接: %s（等待首个结果确认配置被接受）", self.url)

    async def push(self, audio: np.ndarray) -> None:
        """送一段音频。``audio`` 为 1-D float32 ∈ [-1,1]，16kHz。

        内部转 PCM int16（ASR 服务要求）。
        """
        if self.ws is None:
            raise RuntimeError("connect() 未调用")
        x = np.asarray(audio, dtype=np.float32).reshape(-1)
        pcm16 = np.clip(x * 32767.0, -32768, 32767).astype(np.int16)
        await self.ws.send(pcm16.tobytes())
        self.chunks_sent += 1
        self.samples_sent += pcm16.size

    async def finish(self) -> None:
        """通知服务端音频结束（触发最终结果与 is_final）。"""
        if self.ws is None:
            return
        await self.ws.send(json.dumps({"is_speaking": False}))

    # ------------------------------------------------------------------ #

    def ms_to_sample(self, ms: float) -> int:
        """ASR 的毫秒时间戳 → 会话采样轴索引。

        ⚠️ ASR 的时间戳原点是其**流起点**。调用方需在会话开始时用
        ``stream_t0`` 标注该起点在会话轴上的位置。
        """
        return self.stream_t0 + int(round(ms * SR / 1000.0))

    async def recv_loop(self, on_message: Callable[[dict], None]) -> None:
        """接收循环。``on_message(parsed_dict)`` 对每条 JSON 消息回调。

        识别到 ``is_final`` 或连接关闭时返回。
        """
        import websockets
        if self.ws is None:
            raise RuntimeError("connect() 未调用")
        try:
            while not self.closed:
                raw = await self.ws.recv()
                if isinstance(raw, bytes):
                    logger.warning("ASR 收到未预期的 binary 帧（%d B），忽略", len(raw))
                    continue
                msg = json.loads(raw)
                mode = str(msg.get("mode") or "")

                if mode == "turnsense":
                    self.turnsense_count += 1
                elif mode == "2pass-online":
                    self.partials += 1
                elif mode.startswith("2pass-offline"):
                    self.finals += 1

                on_message(msg)

                if msg.get("is_final"):
                    logger.info(
                        "ASR 流结束: partials=%d finals=%d turnsense=%d",
                        self.partials, self.finals, self.turnsense_count,
                    )
                    return
        except websockets.ConnectionClosed as exc:
            if not self.closed:
                self.error = f"连接关闭: {exc}"
                logger.warning("ASR %s", self.error)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            self.error = f"{type(exc).__name__}: {exc}"
            logger.error("ASR 接收异常: %s", self.error)

    # ------------------------------------------------------------------ #

    async def drain(self, timeout: float = 15.0) -> bool:
        """发结束信号并等最终结果回来。

        ⚠️ 发完 ``is_speaking: false`` 后**不能立刻 close** —— 服务端需要
        时间去跑完最终识别（2pass-offline）并把结果发回来。立刻关连接会
        丢掉最后一句，表现为「有部分结果但没有最终结果」。

        所以拆成两步：``drain()`` 在会话收尾时调用（可以慢慢等），
        ``close()`` 只负责断开（必须快，不能被 cancel 拖住）。

        返回是否收到 ``is_final``。
        """
        if self.ws is None:
            return False
        try:
            await asyncio.wait_for(self.finish(), timeout=5.0)
        except Exception as exc:  # noqa: BLE001
            logger.debug("ASR 发送 is_speaking=false 失败（忽略）: %s", exc)
            return False
        # 轮询等 is_final（recv_loop 在收到时会把 finals 加一）
        t0 = time.monotonic()
        target = self.finals + 1
        while time.monotonic() - t0 < timeout:
            if self.finals >= target:
                logger.info("ASR 已收到最终结果（%.1fs）", time.monotonic() - t0)
                return True
            await asyncio.sleep(0.1)
        logger.warning("ASR 等待最终结果超时（%.1fs），可能有尾句丢失", timeout)
        return False

    async def close(self, send_end: bool = True) -> None:
        """关闭连接。**必须快** —— 不在这里等待/睡眠。

        收尾（发 is_speaking=false + 等最终结果）请先调 ``drain()``。
        """
        if self.ws is None:
            return
        self.closed = True
        try:
            await self.ws.close()
        except Exception:  # noqa: BLE001
            pass
        logger.info(
            "ASR 已关闭: sent=%d(%.1fs) partials=%d finals=%d turnsense=%d",
            self.chunks_sent, self.samples_sent / SR,
            self.partials, self.finals, self.turnsense_count,
        )


# ====================================================================== #
#  消息解析辅助（把 ASR 的原始 JSON 转成 downstream 事件）
# ====================================================================== #

def parse_confidence(msg: dict) -> Optional[float]:
    """从 ASR 消息里取平均置信度。实测 2pass-online 可能没有该字段。"""
    conf = msg.get("confidence")
    if isinstance(conf, dict):
        avg = conf.get("avg")
        if isinstance(avg, (int, float)):
            return float(avg)
    return None


def parse_final(msg: dict) -> dict:
    """把 2pass-offline 消息的字段规范化（时间戳一律转成毫秒 int）。"""
    ts = msg.get("timestamp") or []
    tokens, token_times = [], []
    if isinstance(ts, list):
        for item in ts:
            if isinstance(item, (list, tuple)) and len(item) >= 3:
                try:
                    s, e, ch = int(item[0]), int(item[1]), str(item[2])
                except (TypeError, ValueError):
                    continue
                prob = float(item[3]) if len(item) > 3 else 0.0
                tokens.append(ch)
                token_times.append((s, e, ch, prob))
    return {
        "text": msg.get("text", ""),
        "confidence": parse_confidence(msg),
        "tokens": tokens,
        "token_times": token_times,
        "start_ms": int(msg.get("start_time", 0) or 0),
        "end_ms": int(msg.get("end_time", 0) or 0),
        "vad_segments": msg.get("vad_segments") or [],
        "is_final": bool(msg.get("is_final")),
    }


def parse_turnsense(msg: dict) -> dict:
    """规范化 turnsense 消息。

    ⚠️ ``probabilities`` 实测是 **3 元素数组**，不是 dict。
    顺序对应 complete / incomplete / invalid（由 label 反推校验）。
    """
    probs = msg.get("probabilities")
    if not isinstance(probs, list):
        probs = []
    return {
        "label": str(msg.get("label") or ""),
        "probabilities": [float(p) for p in probs if isinstance(p, (int, float))],
        "segment_start_ms": int(msg.get("segment_start", 0) or 0),
        "segment_end_ms": int(msg.get("segment_end", 0) or 0),
        "speech_duration_s": float(msg.get("speech_duration", 0.0) or 0.0),
    }
