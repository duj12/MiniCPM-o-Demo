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

    # ---- 感知（SOP 03/12/13/14/15）：语种 / 情绪 / 音频事件 / SNR /
    #      突发噪声 / 交叠语音。字段并入服务端下发的 ``state``，本层只透传。
    #      打开后 ``state`` 从 5 个基础字段变成 14 个；关掉则字节级回到原样。
    #      代价在**服务端**：每段多一次 SenseVoice 标签 + 交叠 ONNX 推理
    #      （与识别并行、独立线程池，不拖慢 ASR）。默认开，可用环境变量
    #      ``ORCH_ASR_PERCEPTION=0`` 全局关掉（见 ``config.Settings``）。
    #      ⚠️ ``svs_lang`` 固定成 ``zh`` 时模型 lid 输入被钉死，语种恒为
    #      ``zh``、``language_switch`` 永远 false —— 要真语种检测必须保持 ``auto``。
    enable_perception: bool = True
    perception_language: bool = True
    perception_emotion: bool = True
    perception_event: bool = True
    perception_noise: bool = True
    perception_overlap: bool = True
    perception_snr: bool = True

    #: 允许客户端在 ``session.start`` 里覆盖的字段 → 类型转换。
    #:
    #: ⚠️ **白名单**，不是"客户端传什么就改什么" —— ASR 首帧里还有
    #: ``mode`` / ``chunk_size`` / ``wav_format`` 这类改了会直接让识别
    #: 跑不起来的字段，不该开放给前端随手改。
    #: 这里只放**调参性质**的：影响切句敏感度与过滤强度。
    CLIENT_OVERRIDABLE = {
        "vad_tail_sil": int,
        "turnsense_incomplete_wait_ms": int,
        "confidence_threshold": float,
        "online_confidence_threshold": float,
    }

    def apply_overrides(self, overrides: Optional[dict]) -> list:
        """按白名单应用客户端覆盖，返回实际生效的 ``[(字段, 值)]``。

        值不合法（转不成目标类型 / 为 None）时**跳过该字段**并保留默认，
        绝不因为前端传了个垃圾就让整个会话起不来。
        """
        applied = []
        for key, cast in self.CLIENT_OVERRIDABLE.items():
            if not overrides or key not in overrides:
                continue
            raw = overrides[key]
            if raw is None or raw == "":
                continue
            try:
                val = cast(raw)
            except (TypeError, ValueError):
                logger.warning("ASR 参数 %s=%r 非法，用默认值 %r",
                               key, raw, getattr(self, key))
                continue
            setattr(self, key, val)
            applied.append((key, val))
        return applied

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
            "enable_perception": self.enable_perception,
            "perception_language": self.perception_language,
            "perception_emotion": self.perception_emotion,
            "perception_event": self.perception_event,
            "perception_noise": self.perception_noise,
            "perception_overlap": self.perception_overlap,
            "perception_snr": self.perception_snr,
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

        # 服务端 state 的透传缓存（说/抢/信/完 + transcript）
        self._tracker = AsrStateTracker(self.config)

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

    @property
    def state(self) -> AsrState:
        """最近一条 ASR 消息带来的**服务端**状态快照。

        ``recv_loop`` 在每条消息上都会刷新它，所以 ``on_message`` 回调里读到的
        就是**与该消息同步**的状态（值完全来自服务端，客户端不做推导）。
        """
        return self._tracker.state

    def expire_if_idle(self) -> None:
        """把「一轮已结束」的清零推进一步（**需周期性调用**）。

        由 ``session.run_tick`` 每 50ms 调一次 —— 服务端在一句话说完之后
        不再发消息，不在 tick 里清，IC 会永远停在最后一条 HIGH 上。
        """
        self._tracker.expire_if_idle()

    def reset_turn_state(self) -> None:
        """话轮结束：清空快照。

        ⚠️ **生产不调用** —— 改用 ``expire_if_idle()``。
        保留给测试构造干净起点用。
        """
        self._tracker.reset_turn()

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

                # 先归纳状态再回调 —— 这样 on_message 里读 ``asr.state``
                # 拿到的必然是与本条消息同步的快照。
                self._tracker.update(msg)
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
#  状态快照（**由 ASR 服务端计算**，编排层只负责透传）
# ====================================================================== #

HIGH = "HIGH"
MEDIUM = "MEDIUM"
LOW = "LOW"
NONE = "NONE"

_TIERS = (HIGH, MEDIUM, LOW, NONE)


def _opt_str(v) -> Optional[str]:
    """感知字段的宽容取值：空 / null / 缺失统一归 ``None``（**不是空串**）。

    消费方的判据只能是"有没有值"，所以"没有"必须只有一个表示。
    """
    if v is None:
        return None
    s = str(v).strip()
    return s or None


def _opt_float(v) -> Optional[float]:
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _opt_bool(v) -> Optional[bool]:
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("true", "1", "yes"):
        return True
    if s in ("false", "0", "no"):
        return False
    return None


def _tier(v) -> str:
    """把服务端下发的档位归一成四值之一；未知/缺失 → ``NONE``。"""
    s = str(v or "").upper()
    return s if s in _TIERS else NONE


@dataclass(frozen=True)
class AsrState:
    """一次 ASR 消息带来的状态快照（**值由服务端计算**）。

    字段与取值约定见两份 ``docs/websocket-api.md`` §4（服务端权威），
    以及 `orchestrator/docs/asr-state.md`（编排层的消费约定）。这里只是
    服务端 ``state`` 的结构化拷贝：编排层**不再**自己推导档位。

    一轮结束后（下一拍 tick）五个字段一起清空，没有保持窗口。
    """

    user_speaking_confidence: str = NONE
    barge_in_confidence: str = NONE
    transcript: str = ""
    asr_confidence: str = NONE
    turn_complete_confidence: str = NONE

    # ---- 感知字段（2026-10-29 起透传，服务端 ``enable_perception=true``）----
    #
    # ⚠️ 这些值**只在 ``2pass-offline`` 那一帧里有**（online 帧里服务端发
    #    ``null``），所以它们描述的是「**刚说完的这一段**」，不是"此刻"。
    #    段结束后随 `说`/`抢` 一起在两拍里清掉（见 `expire_if_idle`）。
    #
    # 全部 ``None`` = 服务端没开感知 / 还没出结果。消费方**必须容忍** None：
    # 线上两种配置（5 键 / 14 键）并存，判据只能是"这个键有没有值"。
    language: Optional[str] = None          # 语种码，如 'zh'
    language_prob: Optional[float] = None
    language_switch: Optional[bool] = None  # 本段内发生过语种切换？
    emotion: Optional[str] = None           # 语音情绪标签，如 'neutral'
    emotion_prob: Optional[float] = None
    speech_event: Optional[str] = None      # 如 'speech' / 'laugh' …
    noise: Optional[dict] = None            # {'floor_dbfs','burst':{...},…}
    overlap: Optional[dict] = None          # {'detected','prob','regions',…}
    snr_db: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "user_speaking_confidence": self.user_speaking_confidence,
            "barge_in_confidence": self.barge_in_confidence,
            "transcript": self.transcript,
            "asr_confidence": self.asr_confidence,
            "turn_complete_confidence": self.turn_complete_confidence,
            # 感知字段原样带出（IC 只读它认识的键，多出来的键无害；
            # UI / 调试 / 后续消费方靠它们拿到服务端的感知结果）。
            "language": self.language,
            "language_prob": self.language_prob,
            "language_switch": self.language_switch,
            "emotion": self.emotion,
            "emotion_prob": self.emotion_prob,
            "speech_event": self.speech_event,
            "noise": self.noise,
            "overlap": self.overlap,
            "snr_db": self.snr_db,
        }


class AsrStateTracker:
    """服务端 ``state`` 的**透传缓存**（不在本地归纳）。

    五个状态量（说 / 抢 / 信 / 完 + ``transcript``）全部由 ASR 服务端算好，
    放在**每条消息**的 ``state`` 里下发（取值约定见两份
    ``docs/websocket-api.md`` §4）。客户端只做两件事：

    1. 记住最后一条消息的 ``state``，原样转给 IC / UI；
    2. 一轮结束后的**下一拍**把快照清成全 ``NONE``。

    第 2 步为什么留在客户端：服务端只在"有消息"时带 ``state``，一句说完
    （``2pass-offline``）之后它**不会再主动发任何消息**。不清的话 IC 会
    永远停在最后一条快照上（实测：一轮结束后 ``说``/``抢`` 一直 HIGH）。

    **没有时间保持窗口**（原 ``ORCH_ASR_TURN_HOLD_MS`` 的 1s 保留已删除），
    但清零要分**两拍**，因为 IC 的 ANSWER 判定需要中间那一拍：

      - offline 那一条：服务端原值（``说``/``抢``=HIGH，``完``=HIGH，带转写）
      - 下一拍（tick）：``说``/``抢`` → ``NONE``，``完``/``信``/``transcript``
        **保留** —— 这就是 IC 的 ``fresh_turn → ANSWER`` 依赖的"安静下来但
        本轮已拍板"快照（见 `tests/test_omni_describe_stage.py` 的说明）
      - 再下一拍：全部清空

    两拍都是 tick 驱动（各 50ms），不是计时器。

    唯一例外是 ``turnsense=incomplete``：段关了但话没说完，后半句马上就来，
    这两拍都不发生 —— 否则 IC 会以为一轮已经结束。
    """

    def __init__(self, config: Optional[AsrConfig] = None) -> None:
        self.config = config or AsrConfig()
        self._state = AsrState()
        #: 收到段末 ⇒ 等 tick 走"安静下来"那一拍
        self._turn_ended = False
        #: 已经走过"安静下来"那一拍 ⇒ 下一次 tick 全部清空
        self._settled = False

    # ------------------------------------------------------------------ #

    @property
    def state(self) -> AsrState:
        """当前快照 —— 服务端原值（清零那两步除外）。"""
        return self._state

    def reset_turn(self) -> None:
        """清空本轮（**仅供测试/显式重置**；生产走 ``expire_if_idle``）。"""
        self._state = AsrState()
        self._turn_ended = False
        self._settled = False

    # ------------------------------------------------------------------ #

    def update(self, msg: dict) -> AsrState:
        """吃一条原始 ASR JSON，采纳其中的 ``state``。

        收尾帧（``is_final`` 且空文本）是"流结束"的信号，直接归零 ——
        服务端那一拍里的值（``说=MEDIUM`` 之类）只是老客户端的兼容残留，
        不代表当前状态，不能透传给 IC。
        """
        mode = str(msg.get("mode") or "")
        text = str(msg.get("text") or "")

        if msg.get("is_final") and not text.strip():
            self._state = AsrState()
            self._turn_ended = False
            self._settled = False
            return self._state

        st = msg.get("state")
        if isinstance(st, dict) and st:
            self._state = AsrState(
                user_speaking_confidence=_tier(st.get("user_speaking_confidence")),
                barge_in_confidence=_tier(st.get("barge_in_confidence")),
                transcript=str(st.get("transcript") or ""),
                asr_confidence=_tier(st.get("asr_confidence")),
                turn_complete_confidence=_tier(
                    st.get("turn_complete_confidence")),
                language=_opt_str(st.get("language")),
                language_prob=_opt_float(st.get("language_prob")),
                language_switch=_opt_bool(st.get("language_switch")),
                emotion=_opt_str(st.get("emotion")),
                emotion_prob=_opt_float(st.get("emotion_prob")),
                speech_event=_opt_str(st.get("speech_event")),
                noise=st.get("noise") if isinstance(st.get("noise"), dict) else None,
                overlap=(st.get("overlap")
                         if isinstance(st.get("overlap"), dict) else None),
                snr_db=_opt_float(st.get("snr_db")),
            )

        # 段末判定：只有「服务端关掉了一个 VAD 段」才需要走清零两步。
        ts = extract_turnsense(msg)
        if ts is not None:
            self._turn_ended = is_turnsense_complete(ts)
        elif mode == "2pass-offline":
            self._turn_ended = True
        elif mode == "2pass-online":
            # 新的一拍流式帧 ⇒ 上一轮已经过去，取消待清理
            self._turn_ended = False
        self._settled = False
        return self._state

    def expire_if_idle(self) -> None:
        """周期性推进（tick 每 50ms 调一次）：一轮结束后分两拍清零。

        第一拍只把 ``说``/``抢`` 归零、保留本轮结论（IC 的 ANSWER 靠它），
        第二拍全部清空。**没有时间保持窗口** —— 两拍都是 tick 驱动。
        """
        if not self._turn_ended:
            return
        if not self._settled:
            self._state = AsrState(
                user_speaking_confidence=NONE,
                barge_in_confidence=NONE,
                transcript=self._state.transcript,
                asr_confidence=self._state.asr_confidence,
                turn_complete_confidence=self._state.turn_complete_confidence,
                # 感知字段是**本轮结论**（描述刚说完的那一段），与 信/完
                # 同档保留一拍；下一拍一起清空。
                language=self._state.language,
                language_prob=self._state.language_prob,
                language_switch=self._state.language_switch,
                emotion=self._state.emotion,
                emotion_prob=self._state.emotion_prob,
                speech_event=self._state.speech_event,
                noise=self._state.noise,
                overlap=self._state.overlap,
                snr_db=self._state.snr_db,
            )
            self._settled = True
        else:
            self._state = AsrState()
            self._turn_ended = False
            self._settled = False


def parse_confidence(msg: dict) -> Optional[float]:
    """从 ASR 消息里取平均置信度（``confidence`` **对象**形态）。

    ⚠️ 两个服务端字段不一致，别拿这一个当唯一来源：

      - ``Fun-ASR-deploy``（Python）：online **和** offline 都发 ``confidence``
        对象 ``{"avg":…, "token":{…}}``
      - ``asr-2pass``（C++）：只有 offline 发该对象；online 走标量
        ``online_confidence``（见 :func:`parse_online_confidence`）
    """
    conf = msg.get("confidence")
    if isinstance(conf, dict):
        avg = conf.get("avg")
        if isinstance(avg, (int, float)):
            return float(avg)
    return None


def parse_online_confidence(msg: dict) -> Optional[float]:
    """取**流式**帧的置信度，兼容两个服务端的字段差异。

    实测：

      - ``asr-2pass``（C++）→ ``online_confidence`` **标量**
        （如 ``0.7588891983032227``）
      - ``Fun-ASR-deploy``（Python）→ ``confidence`` **对象**，取其中的 ``avg``
        （如 ``{"avg": 0.7385, "token": {...}}``）

    ⚠️ 同一个 offline 帧里两个字段**可以同时存在**（31323 实测），所以这里是
    「先标量、后对象」的有序回退，不是二选一。
    """
    scalar = msg.get("online_confidence")
    if isinstance(scalar, (int, float)):
        return float(scalar)
    return parse_confidence(msg)


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


#: turnsense 三分类的顺序（``probabilities`` 归一成数组后按此排列）
TURNSENSE_LABELS = ("complete", "incomplete", "invalid")


def _normalize_probabilities(probs) -> list:
    """把 turnsense 的 ``probabilities`` 归一成 ``[P(complete), P(incomplete), P(invalid)]``。

    两个服务端类型不同，实测：

      - ``asr-2pass``（C++）→ **数组** ``[0.2126, 0.5752, 0.2121]``
      - ``Fun-ASR-deploy``（Python）→ **dict**
        ``{"complete": 0.213, "incomplete": 0.575, "invalid": 0.212}``
    """
    if isinstance(probs, dict):
        out = []
        for name in TURNSENSE_LABELS:
            v = probs.get(name)
            out.append(float(v) if isinstance(v, (int, float)) else 0.0)
        return out
    if isinstance(probs, list):
        return [float(p) for p in probs if isinstance(p, (int, float))]
    return []


def extract_turnsense(msg: dict) -> Optional[dict]:
    """从任意 ASR 消息里挖出 turnsense 结果，没有则返回 ``None``。

    ⚠️ 实测有**两种到达形态**，都必须认：

      1. **独立消息** —— ``mode == "turnsense"``，自带 ``segment_start`` /
         ``segment_end``。VAD 切分出**完整语音段**时走这条。
      2. **嵌在 offline 里** —— ``msg["turnsense"]``，**没有** segment 起止。
         流结束（``is_speaking=false``）时的收尾帧走这条。

    早先只认形态 1，导致真实语音下 turnsense 全部丢失（``mode`` 分支成了死代码）。
    """
    if str(msg.get("mode") or "") == "turnsense":
        ts = msg
    else:
        ts = msg.get("turnsense")
        if not isinstance(ts, dict):
            return None

    # 嵌入形态没有 segment 起止 —— 用宿主 offline 消息的 start_time/end_time 补
    seg_start = ts.get("segment_start", msg.get("start_time", 0))
    seg_end = ts.get("segment_end", msg.get("end_time", 0))

    pred = ts.get("prediction_id")
    return {
        "label": str(ts.get("label") or ""),
        "prediction_id": int(pred) if isinstance(pred, (int, float)) else None,
        "probabilities": _normalize_probabilities(ts.get("probabilities")),
        "segment_start_ms": int(seg_start or 0),
        "segment_end_ms": int(seg_end or 0),
        "speech_duration_s": float(ts.get("speech_duration", 0.0) or 0.0),
    }


def parse_turnsense(msg: dict) -> dict:
    """规范化**独立** turnsense 消息（``mode == "turnsense"``）。

    ⚠️ ``probabilities`` 实测是 **3 元素数组**（C++ 服务端），不是 dict；
    Python 服务端给的是 dict —— 两者都由 :func:`_normalize_probabilities` 归一。
    顺序对应 complete / incomplete / invalid（由 label 反推校验）。

    需要同时兼容「嵌在 offline 里」的形态时请用 :func:`extract_turnsense`。
    """
    ts = extract_turnsense(msg)
    if ts is None:
        return {
            "label": "",
            "prediction_id": None,
            "probabilities": [],
            "segment_start_ms": 0,
            "segment_end_ms": 0,
            "speech_duration_s": 0.0,
        }
    return ts


def is_turnsense_complete(ts: Optional[dict]) -> bool:
    """turnsense 是否判为「语义完整」。

    ``label`` 优先；缺失时回退到 ``prediction_id == 0``（C++ 的数值编码）。
    """
    if not ts:
        return False
    if ts.get("label"):
        return ts["label"] == "complete"
    return ts.get("prediction_id") == 0
