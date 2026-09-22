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
import os
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

        # 状态归纳（user_speaking / barge_in / asr_conf / turn_complete）
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
        """最近一条 ASR 消息归纳出的状态快照。

        ``recv_loop`` 在每条消息上都会推进它，所以 ``on_message`` 回调里读到的
        就是**与该消息同步**的状态。
        """
        return self._tracker.state

    def expire_if_idle(self) -> None:
        """把「本轮结论」的过期检查推进一步（**需周期性调用**）。

        由 ``session.run_tick`` 每 50ms 调一次 —— 静音时没有新 ASR 消息，
        只靠「读快照时顺手清」是清不掉的（见 ``AsrStateTracker.expire_if_idle``）。
        """
        self._tracker.expire_if_idle()

    def reset_turn_state(self) -> None:
        """话轮结束：重置本轮累计。

        ⚠️ **生产不调用** —— 改用 ``expire_if_idle()``（段一关 + 超时）。
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
#  状态归纳（把逐帧的 ASR 消息归纳成 5 个离散状态量）
# ====================================================================== #

HIGH = "HIGH"
MEDIUM = "MEDIUM"
LOW = "LOW"
NONE = "NONE"

#: 一轮结束后（段关闭），**本轮结论**（``完`` / ``信`` / ``transcript``）
#: 再保留多久才清（毫秒）。默认 1000，可用 ``ORCH_ASR_TURN_HOLD_MS`` 覆盖。
#:
#: ⚠️ 这个常量**以前**叫 ``SPEAKING_HOLD_MS``、语义是「段关闭后让
#:    ``user_speaking_confidence`` 维持 ``LOW`` 防闪烁」。那个语义**已废弃**：
#:    ``说`` / ``抢`` 描述的是「**本轮内**用户有没有出声 / 抢话」，一轮结束
#:    这个答案就作废了 —— 继续报 HIGH/LOW 是错的。现在它们**段一关立即 NONE**。
#:
#:    保留窗口只留给「**本轮结论**」那三个字段：``完``（本轮说完没有）、
#:    ``信``（本轮置信度）、``transcript``（本轮说了什么）—— 这些是**已经
#:    发生的事实**，轮结束后还要让人看到（UI 上得能读刚说的那句）。
TURN_HOLD_MS = int(os.environ.get("ORCH_ASR_TURN_HOLD_MS", "1000"))

#: ``barge_in_confidence`` 的**段内**档位映射：第 N 个「带文本的流式帧」对应哪档。
#: 注意 ``HIGH`` 还有另一条达到路径 —— 拿到**带转写的最终结果**（见 ``_barge_in``），
#: 那比「吐了 3 个 chunk」更确定（确实说出了内容）。
_BARGE_TIERS = (LOW, MEDIUM, HIGH)

#: ``barge_in_confidence`` 的档位阈值：第 N 个「带文本的流式帧」对应哪一档
_BARGE_TIERS = (LOW, MEDIUM, HIGH)


@dataclass(frozen=True)
class AsrState:
    """一次 ASR 消息归纳出的状态快照。

    **完整取值表见 `orchestrator/docs/asr-state.md`**（权威）。下面是精简版：

    ── 本轮**内**的量（段一关立即 ``NONE``，无保持窗口）──

    ``user_speaking_confidence`` —— 此刻在不在出声
        ``HIGH``   段开着且有转写文本
        ``MEDIUM`` 段开着但还没文本（盲窗：出声了、ASR 还没吐字）
        ``NONE``   段已关 = 本轮结束

    ``barge_in_confidence`` —— 此刻在不在抢话
        ``HIGH``   拿到**带转写的最终结果**，或本段已吐 ≥3 个带文本流式帧
        ``MEDIUM`` 本段 2 个带文本流式帧
        ``LOW``    本段 1 个
        ``NONE``   本段还没有，或段已关

    ── 本轮**结论**（轮结束后保留 ``TURN_HOLD_MS``，供 UI 读刚才那句）──

    ``transcript`` —— 本轮说了什么（online 增量拼接 / offline 覆盖）
    ``asr_confidence`` —— 本轮转写置信度
        ``HIGH`` ≥0.8 · ``MEDIUM`` ≥0.6 · ``LOW`` <0.6 · ``NONE`` 本轮还没有信号
        ⚠️ 空文本帧若**不带**分数，**保持**上一次的值（不擦成 NONE）
    ``turn_complete_confidence`` —— 本轮说完的把握
        ``HIGH``   已拍板（offline 或 turnsense=complete）
        ``MEDIUM`` 发生过 VAD 切分但未拍板
        ``LOW``    有转写、正在说
        ``NONE``   无转写

    ⚠️ 这个「两类」的划分是**语义要求**，不是实现细节：
    「此刻在不在」是瞬时事实，轮结束即作废；「本轮是什么」是已发生的事实，
    轮结束后还要让人看到。早先两者用同一个保持窗口，导致「一轮结束了还在报
    用户在说话」。
    """

    user_speaking_confidence: str = NONE
    barge_in_confidence: str = NONE
    transcript: str = ""
    asr_confidence: str = NONE
    turn_complete_confidence: str = NONE

    def to_dict(self) -> dict:
        return {
            "user_speaking_confidence": self.user_speaking_confidence,
            "barge_in_confidence": self.barge_in_confidence,
            "transcript": self.transcript,
            "asr_confidence": self.asr_confidence,
            "turn_complete_confidence": self.turn_complete_confidence,
        }


class AsrStateTracker:
    """把逐帧 ASR 消息归纳成 :class:`AsrState`。

    **只跟服务端消息走** —— 不做 100ms 合成 tick。实测流式帧约 **600ms 一帧**，
    所以状态的更新粒度就是 600ms。

    核心是让 ``user_speaking_confidence`` 变成**带保持的状态机**而不是逐帧判断。
    逐帧判断有三个真实缺陷：首个流式结果要 ~1450ms 才到（盲窗）、每次 VAD 切分
    都会把状态打回原形（用户还在说下半句）、帧与帧之间完全没有信息。这里改用
    **ASR 服务自己那套 VAD 的段状态** —— 服务端本来就在跑 VAD，它的判断已经
    免费写在消息里了，不需要客户端再跑一个：

      - 流式帧带 ``start_time`` ⇒ 服务端 VAD 有**开着的**语音段
      - ``2pass-offline`` 到达  ⇒ 服务端 VAD **刚关闭**一个段
      - ``turnsense=incomplete`` ⇒ 段关了但语义没说完，用户很可能接着说
    """

    def __init__(self, config: Optional[AsrConfig] = None) -> None:
        self.config = config or AsrConfig()
        # 服务端 VAD 的段状态
        self._seg_open = False
        # 当前段的累积转写（online 增量拼接 / offline 整段替换）
        self._transcript = ""
        # 本轮抢话档位：累计「带新文本的流式帧」数（段内爬档用）。
        # ⚠️ **不再于 offline 时归零** —— 见 `_barge_final` 与 `_on_offline`。
        self._barge_frames = 0
        #: **本段已拿到带转写的最终结果** —— 抢话的最硬证据（「用户确实
        #: 说出了内容」）。达到即 ``抢`` = ``HIGH``，比"吐了几个 chunk"确定得多。
        #: 段一开始就复位；空文本的最终结果**不置位**（那只是 VAD 收尾帧，
        #: 不代表说出了内容 —— 与「空帧不擦转写」同一口径）。
        self._barge_final = False
        #: **刚拍板、等着被消费的那一拍**（见 ``_turn_active``）。
        #: ``_on_offline`` 置位 → 该拍的 ``说``/``抢`` 仍报 HIGH
        #: → 由 ``expire_if_idle()``（tick）或新消息到达时清掉。
        #:
        #: ⚠️ **读快照（``.state``）不会消费它** —— recv_loop 对每条消息读
        #: 两次快照，只有后一次（``on_message`` 里那次）是发给 IC/UI 的；
        #: 若读快照就消费，第一次读用掉缓冲、发给 IC 的已是 NONE。
        self._pending_close = False
        #: **本段**是否出过声（有过带文本的流式帧）。
        #: ⚠️ 早先它还兼管 `_user_speaking()` 在段关闭后 hold 窗口里的 LOW
        #:    判定；那个 hold 语义已废弃（一轮结束立即 NONE），现在它只剩
        #:    诊断用途。
        self._spoke_this_segment = False
        # 段是否已关闭？（下一段第一条流式帧到达时清空转写）
        self._segment_closed = False
        # **本段已拍板** —— 收到 offline 结果，或 turnsense 判 complete。
        # 这是「本轮已说完」最硬的证据，下一段开始时清掉。
        self._segment_done = False
        # 本轮内是否出现过 VAD 段切分（收到过 offline）。按整轮累计，
        # 用来把切分之后的流式帧标成 MEDIUM。
        self._vad_split_seen = False
        # 段关闭后维持 LOW 的截止时刻（取不到时钟时为 None）
        self._hold_until_ms: Optional[float] = None
        # 最近一次的 asr_confidence 档位
        self._asr_conf = NONE

    # ------------------------------------------------------------------ #

    @property
    def state(self) -> AsrState:
        """当前状态快照。

        ⚠️ **读的时候会顺手清**：一句说完（收到 offline）并过了 hold 窗口
        之后，四个状态量全部归 ``NONE``、``transcript`` 清空。

        为什么需要这一步：状态机是**纯事件驱动**的 —— 所有字段都只在
        "收到下一条消息"时被覆盖。可一句话说完就**不再有消息**了，于是
        最后那组值会**永远挂着**（实测：静音 3.5s 后 `抢`/`信`/`完` 还是
        旧值，只有 `说` 会掉到 NONE）。表现就是"早就不说话了，状态栏还写
        着置信度 HIGH"。

        为什么放在**读快照**时而不是加定时器：清空的本质是"这轮结束了"，
        而没人读的时候清不清都无所谓 —— 省掉一个后台定时任务。

        ⚠️ **不会影响"offline 后立刻来新一句"**：新消息一进来就自己在
        `_on_online` / `_on_offline` 里把该覆盖的覆盖掉（新 transcript、
        barge 重新计数），这条清理路径只在**没有新消息**时才起作用。

        ⚠️⚠️ **这里只清「本轮结论」的超时，绝不消费 ``_pending_close``**。
        原因：``recv_loop`` 对**每条消息读两次快照** ——
        ``self._tracker.update(msg)``（内部 ``return self.state``）一次，
        紧接的 ``on_message(msg)`` 里再读一次，而**后者的 ``st`` 才是发给
        IC / UI 的那份**。若在这里消费缓冲，第一次读就把它用掉了，
        真正的下发读到的已经是 ``NONE`` —— 「拿到最终结果 = 抢 HIGH」
        永远到不了 IC（实测踩过，用户一眼看出来）。
        ``_pending_close`` 只由 ``expire_if_idle()``（tick）或新消息到达消费。
        """
        self._expire_turn_results()
        return AsrState(
            user_speaking_confidence=self._user_speaking(),
            barge_in_confidence=self._barge_in(),
            transcript=self._transcript,
            asr_confidence=self._asr_conf,
            turn_complete_confidence=self._turn_complete_conf(),
        )

    def expire_if_idle(self) -> None:
        """周期性推进：消费一拍缓冲 + 过期「本轮结论」。**由 tick 调用**。

        ⚠️ **必须由外部周期性调用**（``session.run_tick`` 每 50ms 一次）。
        不能只靠读快照时顺手清 —— 那条路只在**收到新 ASR 消息**时才发生，
        而「说完一句就静音」的场景下恰恰没有新消息（这正是问题 A）。
        """
        # 消费「刚拍板那一拍」的缓冲 —— 让 `说`/`抢` 从 HIGH 转 NONE。
        # offline 那一拍已让 IC 收到过 HIGH，这里是"下一拍清"。
        if self._pending_close and not self._seg_open:
            self._pending_close = False
        self._expire_turn_results()

    def _expire_turn_results(self) -> None:
        """过了 ``TURN_HOLD_MS`` ⇒ 清掉**本轮结论**（`完`/`信`/`transcript`）。

        条件收紧到「**不在说话**」：``_seg_open`` 为真说明新一轮已经开始了，
        此时绝不能清 —— 那会打断刚爬起来的本轮状态。

        ⚠️ 只管「本轮结论」那三个字段。``说``/``抢`` 是「本轮内」的量，
        段一关就已返回 ``NONE``，不依赖这个方法。
        """
        if self._seg_open:
            return
        if self._hold_until_ms is None:
            return
        now = self._now_ms()
        if now is None or now < self._hold_until_ms:
            return
        self._clear_all()

    def _clear_all(self) -> None:
        """清掉**本轮结论**（转写 / 置信度 / 是否说完）+ 全部段内状态。

        ``说`` / ``抢`` 不需要在这里"清" ——
        它们是读时现算的，段一关（``_seg_open=False``）就返回 ``NONE`` 了。
        """
        self._transcript = ""
        self._asr_conf = NONE
        self._segment_done = False
        self._vad_split_seen = False
        self._barge_frames = 0
        self._barge_final = False
        self._pending_close = False
        self._spoke_this_segment = False
        self._segment_closed = False
        self._hold_until_ms = None

    def reset_turn(self) -> None:
        """话轮结束：清空本轮累计（等价于立即过期）。

        ⚠️ **生产代码不调用它** —— 「一轮结束」现在由
        ``_seg_open`` 转 False + ``expire_if_idle()`` 的超时路径覆盖。
        保留它是给测试用的（``test_asr_state.py`` 用它构造干净起点）。
        """
        self._clear_all()

    # ------------------------------------------------------------------ #

    def update(self, msg: dict) -> AsrState:
        """吃一条原始 ASR JSON，推进状态机并返回新快照。"""
        mode = str(msg.get("mode") or "")

        # turnsense 可能有**独立消息**和**嵌在 offline 里**两种形态，
        # 这里统一按「本段收到过判决」处理。注意它可能先于 offline 到达
        # （runtime 里 incomplete 会 defer，offline 要等超时）——那时段其实
        # 已经关闭了，所以这里就把段标记为关闭。
        ts = extract_turnsense(msg)
        if ts is not None:
            self._seg_open = False
            self._mark_closed()
            # ⚠️ 语义判完整 **且本段确实有转写** 才拍板。
            #    没有转写 = 这段语音没被识别出内容（噪声/被过滤），
            #    谈不上"说完了" —— 拍板会让 IC 判出一个**空转写的轮**。
            #    与 `_on_offline` 里「空文本不拍板」同一口径。
            if is_turnsense_complete(ts) and self._transcript:
                self._segment_done = True

        if mode == "turnsense":
            # 独立消息：状态已由上面更新，没有转写要处理
            return self.state

        if mode == "2pass-online":
            self._on_online(msg)
        else:
            # 2pass-offline（含匿名收尾帧）
            self._on_offline(msg)
        return self.state

    # ------------------------------------------------------------------ #

    def _on_online(self, msg: dict) -> None:
        delta = msg.get("text") or ""
        # 上一段已关闭 ⇒ 这是新段的第一帧
        if self._segment_closed:
            self._transcript = ""
            self._segment_closed = False
            self._spoke_this_segment = False   # 新的一段，重新计
            # 抢话的两个量**每段重置** —— 这样既不跨句累积（帧数从 0 爬），
            # 又不会因为 offline 不再归零而丢掉"本段已确认说出内容"。
            self._barge_frames = 0
            self._barge_final = False
            # 新段开始：上一轮的"一拍缓冲"就此作废（它只在 offline 那一拍有效）
            self._pending_close = False
            # 新的一段还没拍板。注意 `_vad_split_seen` **不清** —— 它按整轮
            # 累计，正是它把「切分之后的流式帧」标成 MEDIUM。
            self._segment_done = False

        self._seg_open = True
        if delta:
            # ⚠️ 实测流式 ``text`` 是**增量片段**（"今天的"/"说出去看"），
            # 不是累积文本，必须自己拼。
            self._transcript += delta
            self._barge_frames += 1
            self._spoke_this_segment = True
            self._asr_conf = self._grade(parse_online_confidence(msg))
        else:
            # 空文本帧：要么是低置信度被服务端过滤掉，要么是纯噪声段。
            #
            # ⚠️ **没有置信度时保留上一次的值，不回退 NONE。**
            #    实测（31366 / C++）：**有文本的帧一定带 `online_confidence`，
            #    空文本帧一定不带**。早先这里"拿不到分就置 NONE"，于是每个
            #    空帧都把上一帧刚给的分擦掉 —— `信` 在 HIGH/NONE 之间逐帧
            #    抖动（实测日志：0.854 → None → 0.795 → None …）。
            #    空帧只是"这一帧没有新分数"，不代表这一轮没置信度。
            conf = parse_online_confidence(msg)
            if conf is None:
                pass                      # 保留上一次的 `_asr_conf`
            elif conf < self.config.online_confidence_threshold:
                self._asr_conf = LOW      # 有分且偏低 ⇒ 确实没识别出来
            else:
                # 有分但不低：文本却被过滤了 —— 说明服务端因其它原因丢弃
                # （噪声段/无效段），按分数正常分档，不要凭空降级。
                self._asr_conf = self._grade(conf)

    def _on_offline(self, msg: dict) -> None:
        text = (msg.get("text") or "").strip()
        if text:
            # 最终结果**覆盖**本段累积的流式文本
            self._transcript = text
        # 无论成功失败，离线结果到达就意味着服务端那个段已经关闭
        self._seg_open = False
        self._segment_closed = True
        # ⚠️ 但**这一拍** `说`/`抢` 还要报 HIGH（`_pending_close`）——
        #    offline 同时是「本轮结束」和「确认说出内容」，两者在同一条消息上。
        #    直接关段会让「抢=HIGH」永远观测不到（快照与关段同拍完成）。
        #    留一拍缓冲，让 IC 必然收到一次；下一拍由 expire_if_idle 清掉。
        self._pending_close = True
        # ---- 是否"拍板"（turn_complete → HIGH）----
        #
        # ⚠️⚠️ **只有带转写的 offline 才算拍板**。
        #
        # 文本被服务端过滤掉（`text` 为空）时，这一轮**根本不存在** ——
        # 那是噪声段/无效段（实测带的 `turnsense: invalid`）。若照样置
        # `_segment_done=True`，状态会变成：
        #
        #     空转写 + asr_confidence=LOW + turn_complete=HIGH
        #
        # 而 IC 的 Policy 正是拿这个组合判「没听清」：
        #
        #     if fresh_turn and speech.asr_confidence == LOW:
        #         return UTTER(sop="01", text="抱歉，我没听清，请您再说一遍。")
        #
        # → 用户什么也没说（或只是噪声），却被回一句「请再说一遍」。
        # 实测 D16 就是这么复现的。
        #
        # 所以空文本时**不拍板**，`turn_complete` 落到 LOW（有转写才 HIGH）。
        if text:
            self._segment_done = True
        # 本轮出现过 VAD 切分（收到过 offline）。**按整轮累计、不清零** ——
        # 它服务于 `turn_complete`：有它才把"切分之后那段"的流式帧标成
        # MEDIUM（而不是 LOW）。
        # ⚠️ 与 barge 的"每句重新爬"是**两套语义**，别混：barge 在下面归零，
        #    这个不归零。早先改 barge 时误删过这一行，导致切分后的段掉成 LOW。
        self._vad_split_seen = True
        self._mark_closed()
        # 离线也走同一套三档（用离线自带的 confidence.avg）—— 与流式口径
        # 一致，用户不用记"流式看这个线、离线看那个线"。
        #
        # ⚠️ **没带分数时保留上一次的值，不擦成 NONE** —— 与流式空帧同一口径
        #    （见 `_on_online` 里的长注释）。收尾帧（`is_final=true`、
        #    `text` 为空）就是典型：它只是"这段结束了"，不代表这一轮没置信度。
        #    早先无条件 `_grade(...)`，`_grade(None)` 返回 NONE，于是收尾帧
        #    把刚拿到的分擦掉。
        _off_conf = parse_confidence(msg)
        if _off_conf is not None:
            self._asr_conf = self._grade(_off_conf)

        # ---- 抢话：**有转写 = 确实说出了内容 → 置位（不再归零）** ----
        #
        # ⚠️ 这里以前是 `self._barge_frames = 0`（归零）—— 那是错的：
        #    短句往往 1~2 个 chunk 就出 offline，归零让「刚确认说出内容」
        #    这一刻 `抢` 反而**砸到 NONE**，正好反了。
        #
        # 但**跨句累积**那个问题是真的（早先不清零 → 句3 刚开始就是 HIGH，
        # 实测复现）。两者兼顾的办法是分开两个量：
        #   · `_barge_frames` 段**内**爬档，段一开就重置（见 `_on_online`）
        #   · `_barge_final`   本段拿到带转写的最终结果 → `抢` 直接 HIGH
        # 这样既不跨句累积（帧数每段重置），也不会在拿到结果时反而降档。
        #
        # ⚠️ 空文本的 offline **不置位** —— 那只是 VAD 收尾帧，不代表
        #    说出了内容（与「空帧不擦转写」同一口径）。
        if text:
            self._barge_final = True
        # `_spoke_this_segment` **不清** —— 段内用，下段第一帧才重置。
        # （它现在只作诊断用：`说` 的 hold 语义已废弃，一轮结束即 NONE）

    # ------------------------------------------------------------------ #

    def _mark_closed(self) -> None:
        self._hold_until_ms = self._now_ms() + TURN_HOLD_MS

    def _now_ms(self) -> Optional[float]:
        try:
            return time.monotonic() * 1000.0
        except Exception:  # noqa: BLE001
            return None

    def _grade(self, conf: Optional[float]) -> str:
        """浮点置信度 → **三档**（``HIGH`` / ``MEDIUM`` / ``LOW``），无信号 ``NONE``。

        阈值用配置里现有的两个，不新增参数：

            conf >= confidence_threshold(0.8)          → HIGH
            >= online_confidence_threshold(0.6)        → MEDIUM
            <  online_confidence_threshold             → LOW

        ⚠️ 早先只有两档（HIGH/LOW，且流式拿 0.6 当 HIGH 线）—— 于是
        **0.6~0.8 的临时结果也被显示成 HIGH**，说话时满屏"高置信"的低质量
        结果，看不出哪些能信。加一档 MEDIUM 正好落在这个区间。
        """
        if conf is None:
            return NONE
        if conf >= self.config.confidence_threshold:
            return HIGH
        if conf >= self.config.online_confidence_threshold:
            return MEDIUM
        return LOW

    def _user_speaking(self) -> str:
        """用户是否在出声 —— **本轮内**的量。

        ``HIGH``   段开着（含"刚拍板那一拍"）且有转写文本
        ``MEDIUM`` 段开着但还没有文本（盲窗：出声了、ASR 还没吐字）
        ``NONE``   段已关且过了缓冲那一拍

        ⚠️ **没有"保持窗口"**（早先的 ``SPEAKING_HOLD_MS`` 语义已废弃 ——
        那等于「一轮结束了还在报用户在说话」）。但有**一拍缓冲**，
        见 ``_turn_active``：offline 拍板那一拍仍报 HIGH，让 IC 一定收到
        "用户刚才在说话 + 确实说出了内容"这个组合；下一拍起才 NONE。
        """
        if self._turn_active():
            return HIGH if self._transcript else MEDIUM
        return NONE

    def _barge_in(self) -> str:
        """抢话把握 —— **本轮内**的量。

        本轮进行中时，两种达到 ``HIGH`` 的路径：

          · **拿到带转写的最终结果**（``_barge_final``）—— 最确定，
            「用户确实说出了内容」
          · 本段已吐 **≥3 个**带文本的流式帧

        其余按帧数爬档：1 帧 ``LOW`` → 2 帧 ``MEDIUM`` → 0 帧 ``NONE``。

        ⚠️ 早先这里**只**看帧数，且 ``offline`` 一到就把计数归零 —— 于是
        短句（1~2 帧就出最终结果）刚确认说出内容，``抢`` 反而**砸到 NONE**，
        正好反了。现在保留最终结果那条路径，且**用一拍缓冲**保证 IC 收得到。
        """
        if not self._turn_active():
            return NONE          # 一轮结束（且过了缓冲那一拍）
        if self._barge_final:
            return HIGH          # 已确认说出内容
        if self._barge_frames <= 0:
            return NONE
        idx = min(self._barge_frames, len(_BARGE_TIERS)) - 1
        return _BARGE_TIERS[idx]

    def _turn_active(self) -> bool:
        """本轮是否"算还在进行"—— 供 ``说``/``抢`` 判定。

        两种情况都算：

          · **段开着** —— 正常进行中
          · **刚拍板但还没被读过**（``_pending_close``）—— offline 到达的
            **那一拍**。``session.on_message`` 收到消息后会立刻读一次
            ``state`` 投给 IC，所以这一拍读到的 ``说``/``抢`` 就是
            「拍板时刻」的值（``HIGH``）；下一次读（下一拍 tick）才转 NONE。

        ⚠️ **为什么需要这一拍缓冲**：用户要求「拿到最终结果 = 抢话 HIGH」，
        但 offline **同时**是「本轮结束」—— 两者在同一条消息上。
        若在 offline 里直接关段，那个 HIGH 就**永远观测不到**
        （快照和关段在同一拍完成）。留一拍，IC 必然收到一次。
        """
        return self._seg_open or self._pending_close

    def _turn_complete_conf(self) -> str:
        """本轮是否已经说完。

        ``HIGH``   本段**已拍板**（带转写的 ``2pass-offline``，或带转写的
                   turnsense=complete）
        ``MEDIUM`` 本轮发生过 VAD 切分，且**当前有转写**（续接段）
        ``LOW``    **没有任何转写** —— 这段语音没被识别出内容（噪声段、
                   或文本被置信度过滤掉了）
        ``NONE``   （同上，``LOW`` 与 ``NONE`` 目前同义；保留 NONE 供
                   "完全没收到过任何消息"的初始态）

        ⚠️ **「没有转写」优先于「切过段」**：早先判据是
        ``if _vad_split_seen: return MEDIUM`` 在前，没有先看转写 —— 于是
        「本轮切过段，但这一段是空的」会报 ``MEDIUM``。
        而 IC 的 Policy 对空转写有明确分支（``asr_confidence==LOW`` 时
        走「没听清，请您再说一遍」），拿一个 MEDIUM/HIGH 的 ``turn`` 配上
        空转写，语义上是在说"用户说完了一句空话"—— 那不存在。

        **拍板（HIGH）必须是"带转写"的**：文本被过滤掉（``text`` 为空）
        时这一段根本没被识别出内容，谈不上"说完了"。这一条是 D16 那个
        「用户没说有效内容、却被回『请再说一遍』」的根因。
        """
        if not self._transcript:
            # 没有转写 ⇒ 没有一轮可言。哪怕切过段、哪怕收到过 offline。
            return LOW if self._segment_done or self._vad_split_seen else NONE
        if self._segment_done:
            return HIGH
        if self._vad_split_seen:
            # 本轮已经切过段，当前这段是续接的 —— 比首段更有把握一些
            return MEDIUM
        return LOW


# ====================================================================== #
#  消息解析辅助（把 ASR 的原始 JSON 转成 downstream 事件）
# ====================================================================== #

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
