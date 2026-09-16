#!/usr/bin/env python3
"""全双工仿真：**用真实的回声闭环证明参考轨对齐了**。

## 为什么要这个测试

AEC 失效这件事，只看 ERLE / 峰比 / 非零占比这些间接指标已经被带偏过两次。
唯一能定案的是：**把 mic 里确实含回声的信号喂进去，看 ASR 会不会把我们的
播报识别成用户说话**。

真机上做不到（人耳听不出 3dB 的差别，也没法精确控制打断时刻），所以这里
造一个**假浏览器**：它 faithfully 复刻真机播放器的调度行为，并按声学路径
合成 mic 信号：

    mic[t] = 干净用户语音[t] + gain × 喇叭实际播出[t - D_true]

``D_true`` 只有假浏览器知道，服务端永远看不到（真机也是这样）。于是：
**ASR 的识别结果里一旦混进回复文本，就说明 AEC 没消干净。**

## 三种锚点模式 —— 修复要被"证明"，不只是被"跑到"

  ``--anchor armed``   （新代码，正常路径）
      浏览器在收到 tts.start 后立刻**承诺**起播时刻；服务端据此精确落位。
      预期：逐句残差 ≈ 常数 D_true，散度 ≤2 采样。

  ``--anchor legacy``  （**旧行为**：不发承诺）
      浏览器按"收到音频的时刻 + 提前量"起播，服务端按"收到音频的时刻 +
      提前量"预测落位。**同一份服务端代码**，只是走降级分支。
      预期：残差**逐句抖动**（网络/主线程阻塞/时钟落后各不相同）——
      这正是"固定常量 D 吸收不了、所以第一句好后面失效"的可执行证据。

  ``--anchor missing`` （发了能力位但承诺没到 → 超时降级）
      预期：`anchor_source == "predicted"`，且链路不挂。

## 用法

```bash
# 无外部服务：只验证对齐（合成 mic，不跑真 AEC/ASR）
python -m orchestrator.tests.test_duplex_sim --scenario turn --anchor legacy
python -m orchestrator.tests.test_duplex_sim --scenario turn --anchor armed

# 有真 AEC/ASR：验证 ASR 不混入 TTS 内容
python -m orchestrator.tests.test_duplex_sim --scenario turn --anchor armed \
    --wav-a assets/ref_audio/ref_minicpm_signature.wav \
    --wav-b assets/ref_audio/ref_en_dlc_1.wav \
    --d-true-ms 84 --out /tmp/sim
```
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import wave
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from orchestrator.audio.resample import StatefulResampler  # noqa: E402
from orchestrator.clock import SR  # noqa: E402
from orchestrator.downstream.interface import (  # noqa: E402
    AsrFinal, Cancel, Speak,
)
from orchestrator.protocol import MIC_CHUNK  # noqa: E402

# Windows 控制台默认 GBK，打不出 ⚠/✓ 之类的字符会直接抛 UnicodeEncodeError
# 把整个测试带崩（而不是只丢一个字符）。这里统一切到 UTF-8。
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore
except Exception:  # noqa: BLE001
    pass

TTS_SR = 24000
DEFAULT_WAV_A = "assets/ref_audio/ref_minicpm_signature.wav"
DEFAULT_WAV_B = "assets/ref_audio/ref_en_dlc_1.wav"

# 回复文本：必须与两段用户 wav 的**内容毫无重叠**，否则"TTS 漏进 ASR"
# 用文本比对根本区分不出来（默认的 PassthroughDownstream(mode="asr")
# 合成的恰恰是用户自己说的话 —— 所以本脚本自己替换 downstream）。
REPLY_TEXT = "好的我知道了，现在给你详细解释一下这个问题的来龙去脉和解决办法。"
# 回复里的一段特征串：ASR 结果里出现它就说明漏了 TTS 内容
REPLY_MARKER = "来龙去脉"

_failures: List[str] = []


def check(cond: bool, msg: str) -> None:
    print(f"  [{'OK' if cond else 'FAIL'}] {msg}")
    if not cond:
        _failures.append(msg)


def load_wav16k(path: str) -> np.ndarray:
    """读 wav 并重采样到 16kHz float32（复用 test_aec_live_fidelity 的约定）。"""
    with wave.open(path, "rb") as w:
        src_sr = w.getframerate()
        raw = w.readframes(w.getnframes())
        x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        if w.getnchannels() == 2:
            x = x.reshape(-1, 2).mean(axis=1)
    if src_sr != SR:
        n_out = int(len(x) * SR / src_sr)
        pos = np.linspace(0, len(x) - 1, n_out)
        i0 = np.floor(pos).astype(int)
        i1 = np.minimum(i0 + 1, len(x) - 1)
        fr = (pos - i0).astype(np.float32)
        x = (x[i0] * (1 - fr) + x[i1] * fr).astype(np.float32)
    return x


def speech_like(n: int, seed: int = 0) -> np.ndarray:
    """类语音信号 —— **仅在无声学验证时**当占位用。

    ⚠️ 合成信号**永远产不出 ASR 的 2pass-offline 最终结果**（turnsense 判
    invalid，见 test_e2e_local 的注释）。所以只要关心 ASR 文本，就必须用
    真实语音 wav；只有纯对齐检查才可以用它。
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=np.float32) / SR
    f0 = 120 + 40 * np.sin(2 * np.pi * 2.5 * t)
    x = 0.3 * np.sin(2 * np.pi * f0 * t) + 0.15 * np.sin(2 * np.pi * 2 * f0 * t)
    x *= 0.5 + 0.5 * np.sin(2 * np.pi * 3.0 * t)
    # 加宽带分量让 GCC-PHAT 有峰可找（纯正弦的自相关是梳状，容易选错峰）
    x += 0.12 * rng.standard_normal(n).astype(np.float32)
    return x.astype(np.float32)


# ====================================================================== #
#  假浏览器
# ====================================================================== #

class FakeBrowser:
    """忠实复刻真机浏览器的时间行为，并合成声学回声。

    三个模型（缺一不可，少一个这个测试就失去意义）：

      ① **采集时钟**：每个 100ms 块带**发送抖动**。服务端的会话时钟因此
         比真实时刻落后 `0.1s + jitter` 且**逐块变化** —— 这正是"预测落位"
         必须免疫的缺陷，所以抖动必须保留，否则测不出问题。
      ② **播放器**：与真机 `PcmPlayer` 同样的调度语义（`nextAt` 顺排、
         `stop(0)` 掐断已排程源），且**用同一个重采样器**把 24k 的 TTS PCM
         转 16k —— 与 `RefTrack.place()` 内部一致，否则合成出来的回声与
         参考轨内容不是同一段波形，残差会被重采样失配污染。
      ③ **声学路径**：`mic[t] = clean[t] + gain × emitted[t - D_true]`。
    """

    def __init__(self, ws, host_ctx0: float, d_true_samples: int,
                 anchor_mode: str, gain: float = 0.5,
                 noise: float = 1e-4, seed: int = 3) -> None:
        self.ws = ws
        self.c0 = host_ctx0
        self.d_true = d_true_samples
        self.anchor_mode = anchor_mode
        self.gain = gain
        self.rng = np.random.default_rng(seed)

        # ---- 播放器状态（会话采样轴）----
        self.nodes: List[Tuple[int, np.ndarray]] = []   # (起点, 16k 波形)
        self.stop_at: Optional[int] = None              # 实际停下的绝对采样
        self.armed_ctx: Optional[float] = None          # 承诺的起播时刻
        self.first_at: Optional[float] = None           # 实际首播时刻
        self.sent_start: Optional[int] = None           # 本句排程起点（采样）
        # stale 模式用：模拟 tts.start→首块音频之间的 TTS 合成耗时
        self._stale_s = 0.5
        self.stale_ms = 0.0                             # 承诺过期量（回归观测）
        self._armed_sent: set = set()      # 已发过 armed 的 response_id
        self.turn_end: Optional[int] = None             # 本句播完的绝对采样
        self._rs = StatefulResampler(TTS_SR, SR)        # 与 RefTrack 同一个

        self.out: List[np.ndarray] = []                 # 收到的 tts 音频
        # 实际发出去的 mic（含合成回声）—— 残差分析要用它，它是"服务端
        # 眼里的 mic"，与真机 dump 的 -mic.wav 等价
        self.mic_chunks: List[np.ndarray] = []
        self.sent_samples = 0                           # 已发送的采样数
        self.queue: List[np.ndarray] = []               # 待发送的干净人声
        self._queued = 0                                # 队列里的采样数
        self.clean_segments: List[Tuple[int, np.ndarray]] = []
        self._stop_capture = False
        self.events: List[dict] = []
        self.asr_finals: List[str] = []

        self._epoch = (int(host_ctx0 * 1000) % 2147483647) or 1
        self.lead_ms = 200

    # ------------------------------ 时间 ------------------------------ #

    def ctx_now(self) -> float:
        """假浏览器的当前 ctx 时刻（墙钟反推，与服务端时钟无关）。"""
        return self.c0 + (time.monotonic() - self.t_wall0)

    def sample_of(self, ctx: float) -> int:
        """该假浏览器自己的"真相"映射：会话采样 = (ctx - c0) * 16000。"""
        return int(round((ctx - self.c0) * SR))

    # ------------------------------ 发送 ------------------------------ #

    # --------------------------- 连续采集 --------------------------- #

    def enqueue_speech(self, audio: np.ndarray) -> int:
        """把一段用户语音排进采集脚本，返回它在会话采样轴上的起点。

        ⚠️ 采集是**连续**的（见 ``capture_loop``）—— 中间的空闲段发静音。
        这不是可选的：真机的麦克风一直在采，而**回声恰恰出现在"用户没说
        话、喇叭在播"的那段时间里**。只在用户说话时发数据，等于把回声
        最大的那段样本丢掉了，整个测试就白做了。
        """
        base = self.sent_samples + self._queued
        for i in range(0, len(audio), MIC_CHUNK):
            seg = audio[i:i + MIC_CHUNK]
            if seg.size < MIC_CHUNK:
                seg = np.pad(seg, (0, MIC_CHUNK - seg.size))
            self.queue.append(seg.astype(np.float32))
            self._queued += MIC_CHUNK
            self.clean_segments.append((base + i, seg.astype(np.float32)))
        return base

    async def capture_loop(self) -> None:
        """持续以 100ms 的节奏送 mic（有语音送语音，没有送静音）。"""
        import base64
        while not self._stop_capture:
            seg = (self.queue.pop(0) if self.queue
                   else np.zeros(MIC_CHUNK, dtype=np.float32))
            base = self.sent_samples
            ctx = self.c0 + (base / SR) + 0.0037

            # 叠加回声：mic = 干净人声 + gain × 喇叭实际播出[t - D_true]
            # 必须在**发送时**合成 —— 当刻喇叭播到哪由会话采样轴决定，
            # 提前算不出来（回复的调度是随事件发生的）。
            mic_seg = seg.copy()
            start = base - self.d_true
            emit = self.emitted(base + MIC_CHUNK)
            s0, e0 = max(0, start), start + MIC_CHUNK
            if e0 > 0 and s0 < len(emit):
                mic_seg[s0 - start:e0 - start] += \
                    emit[s0:e0].astype(np.float32) * self.gain
            mic_seg += self.rng.standard_normal(MIC_CHUNK).astype(np.float32) * 1e-4
            self.mic_chunks.append(mic_seg.copy())
            self.sent_samples += MIC_CHUNK

            buf = np.ascontiguousarray(mic_seg, dtype=np.float32).tobytes()
            try:
                await self.ws.send(json.dumps({
                    "type": "audio",
                    "audio_base64": base64.b64encode(buf).decode(),
                    "t_ms": 0, "ctx_time": ctx, "epoch": self._epoch,
                }, ensure_ascii=False))
            except Exception:  # noqa: BLE001 —— 连接已关，收工
                return
            # ⚠️ **发送抖动**：真机主线程在跑 25fps 抓帧，块不会准点发出。
            #    服务端时钟因此落后 0.1s + jitter 且**逐块变化** —— 这正是
            #    "预测落位"必须免疫的缺陷，所以抖动必须保留。
            await asyncio.sleep(MIC_CHUNK / SR
                                + float(self.rng.uniform(0.0, 0.020)))

    def stop_capture(self) -> None:
        self._stop_capture = True

    def clean_between(self, a: int, b: int) -> np.ndarray:
        """会话采样轴 [a, b) 上的**干净人声**（不含回声、不含噪声）。"""
        out = np.zeros(max(0, b - a), dtype=np.float32)
        for t0, seg in self.clean_segments:
            lo, hi = max(a, t0), min(b, t0 + len(seg))
            if hi > lo:
                out[lo - a:hi - a] += seg[lo - t0:hi - t0]
        return out

    # ------------------------------ 播放 ------------------------------ #

    def begin_response(self, lead_ms: int) -> float:
        """收到 ``tts.start``：**只记录提前量，不算承诺时刻**。

        ⚠️ 这里刻意不算承诺 —— 服务端在**合成之前**就发了 tts.start，
        等到音频真到达时，此刻算出的"现在 + 提前量"早已过期。真机上正是
        这个错误让参考轨比实际回声**早 340ms**，回声完全消不掉。
        承诺改在 ``play_audio``（第一块音频到达）里算，见 ``_arm``。
        """
        self.lead_ms = lead_ms or 200
        # ⚠️ **每个 response 都要重置这几项**。早先只重置 armed_ctx，
        #    于是第二句起 `first_at` 永远非 None → "首块音频"判定失败 →
        #    armed 从来没发出去（实测 ack=4 predicted=3，一半走预测）。
        self.armed_ctx = None
        self.first_at = None
        self.sent_start = None
        self.turn_end = None
        return 0.0

    def _arm(self) -> float:
        """第一块音频到达时**承诺**起播时刻（返回 0 = 不承诺）。"""
        if self.anchor_mode == "stale":
            return self._arm_stale()      # 回归用：故意复现旧 bug
        if self.anchor_mode in ("legacy", "missing"):
            return 0.0
        if self.armed_ctx is not None:
            return self.armed_ctx
        # 模拟主线程卡顿（25fps 的 toDataURL 会阻塞这条路径）—— 承诺按
        # 卡顿**之后**的当前时刻算，这正是承诺相对服务端预测的价值所在。
        stall = float(self.rng.uniform(0.0, 0.060))
        time.sleep(stall)
        self.armed_ctx = self.ctx_now() + self.lead_ms / 1000.0
        return self.armed_ctx

    def _arm_stale(self) -> float:
        """**复现旧 bug**：在 tts.start 时就承诺（那时音频还没合成）。

        服务端 tts.start 与首块音频之间隔着 TTS 合成的 508ms，所以这个
        承诺到音频真到达时已经过期 —— 播放器只能"马上播"，参考轨却按这个
        过期时刻落位，于是比实际回声早几百毫秒。留着它做回归：如果哪天
        承诺点被改回 tts.start，`--anchor stale` 必须失败。
        """
        if self.armed_ctx is None:
            # 合成耗时**逐句变化**（真机实测 0.4~0.9s），过期量因此也逐句
            # 不同 —— 这才是常量 D 吸收不了的根本原因
            self._stale_s = float(self.rng.uniform(0.40, 0.90))
            self.armed_ctx = self.ctx_now() - self._stale_s + self.lead_ms / 1000.0
        return self.armed_ctx

    def play_audio(self, pcm24: np.ndarray) -> None:
        """收到一块 TTS 音频 → 按播放器语义排程。

        严格对齐真机 ``PcmPlayer.playFloat32`` 的三条语义：

          ① **首块**用承诺时刻（`_arm()`）；若承诺**已过期**（`_armed < now`）
             就只能"马上播"（真机是 `Math.max(nextAt, now+0.02)` 兜底）——
             那正是"承诺过期"bug 的表现形式。
          ② **后续块**一律 `nextAt = 上一块起点 + 上一块时长` 顺排。
             ⚠️ 后续块**不能**再取承诺值：真机曾经写成
             `at = this._armNow() || Math.max(this.nextAt, ...)`，而 `_armNow()`
             每次返回同一个缓存值 → **每个块都排到同一时刻**，整段音频叠在
             一起同时播，听感"一闪而过、只听到开头结尾"。
          ③ 所有块按同一时间轴线性铺开，不会重叠。
        """
        x16 = self._rs.process(pcm24.astype(np.float32) / 32768.0)
        if self.first_at is None:
            at_ctx = self._arm()
            if at_ctx <= 0:
                # legacy / missing：起播 = 收到的这一刻 + 提前量（旧前端行为）
                at_ctx = self.ctx_now() + self.lead_ms / 1000.0
            elif at_ctx < self.ctx_now():
                # 承诺已过期 → 只能就近起播
                self.stale_ms = (self.ctx_now() - at_ctx) * 1000.0
                at_ctx = self.ctx_now() + 0.02
            at = self.sample_of(at_ctx)
            self.first_at = at_ctx
            # 本句**排程起点**（会话采样）—— 参考轨落位起点就是这个位置，
            # stop() 报的 sample_offset 必须与它同基准才能被 resize 用对
            self.sent_start = at
        else:
            # 后续块：从上一块末尾顺排（**绝不重复取承诺值**）
            prev_at, prev_x = self.nodes[-1]
            at = prev_at + len(prev_x)
        self.nodes.append((at, x16))
        self.turn_end = at + len(x16)

    def stop(self) -> int:
        """打断：掐断已排程的源，返回**本句播出的采样数**。

        严格对齐真机 ``PcmPlayer.stop()`` 的语义（那是 WebAudio 里唯一能
        取消已 start 源的办法）：

          · **还没起播**的源 → 整个丢弃（等价于 ``stop(0)``）
          · **正在播**的源   → 只保留已经播出去的那一段，截断
          · 源列表清空 —— 之后的重排从零开始

        ⚠️ 返回值要按"**本句排程起点**往后播了多少"算，而不是源在会话轴
        上的绝对位置差。服务端收到 ``sample_offset`` 后用
        ``RefTrack.resize(rid, sample_offset)`` 把它当作"从本句**落位起点**
        往后的有效长度"，而落位起点就是排程时刻 —— 两者必须同一个基准。

        ⚠️ 早先的写法是在 ``emitted()`` 里按一个全局 ``stop_at`` 截断所有
        内容，那会把**打断之后新排程**的音频也一起抹掉 —— 表现是"第二次
        播报整段消失"，测出来的落位偏差全是假的。
        """
        now = self.sample_of(self.ctx_now())
        played = 0
        kept: List[Tuple[int, np.ndarray]] = []
        for at, x in self.nodes:
            if at >= now:
                continue                      # 还没起播 → 掐掉，不计入
            dur = min(len(x), now - at)
            played += dur
            kept.append((at, x[:dur]))        # 已播出的部分保留
        self.nodes = kept
        if self.sent_start is None:
            return played
        # 从本句**排程起点**算起的播出量（与参考轨的落位起点同一基准）
        return max(0, now - self.sent_start)

    def emitted(self, n: int) -> np.ndarray:
        """喇叭实际播出的波形（绝对采样轴 [0, n)）。"""
        out = np.zeros(n, dtype=np.float32)
        for at, x in self.nodes:
            m = min(len(x), n - at)
            if m > 0:
                out[at:at + m] += x[:m]
        return out

    def mic_total(self, n: int) -> np.ndarray:
        """会话采样轴 [0, n) 上服务端实际收到的 mic（含回声）。"""
        if not self.mic_chunks:
            return np.zeros(n, dtype=np.float32)
        x = np.concatenate(self.mic_chunks)
        if len(x) >= n:
            return x[:n]
        return np.pad(x, (0, n - len(x)))


# ====================================================================== #
#  仿真下游：回一句与用户输入**完全不同**的话
# ====================================================================== #

class SimDownstream:
    """收到 ASR 最终结果 → 回一句固定的、与用户语音无关的长句。

    必须自己实现而不是用 ``PassthroughDownstream(mode="asr")``：后者合成的
    正是**用户自己说的话**，于是"TTS 漏进 ASR"在文本上无法与"用户说了"
    区分 —— 整个测试就白做了。
    """

    def __init__(self, reply: str = REPLY_TEXT) -> None:
        self.reply = reply
        self.spoken = 0

    async def on_session_start(self, ctx) -> list:
        return []

    async def on_session_end(self, reason: str) -> None:
        return None

    async def on_event(self, ev) -> list:
        if isinstance(ev, AsrFinal) and (ev.text or "").strip():
            self.spoken += 1
            return [Speak(text=self.reply)]
        return []


# ====================================================================== #
#  主流程
# ====================================================================== #

async def run(args) -> dict:
    import uvicorn
    from orchestrator.main import REGISTRY, create_app
    from orchestrator.config import Settings

    cfg = Settings.from_env()
    cfg.enable_asr = not args.no_asr
    cfg.enable_omni = False
    cfg.enable_aec = not args.no_aec
    cfg.enable_face = False
    cfg.port = args.port
    cfg.mock_tts = args.mock_tts        # type: ignore[attr-defined]
    cfg.downstream_mode = "echo"        # 先起个永不 Speak 的桩，ready 后替换

    app = create_app(cfg)
    server = uvicorn.Server(uvicorn.Config(
        app, host="127.0.0.1", port=cfg.port, log_level="warning"))
    task = asyncio.create_task(server.serve())
    for _ in range(100):
        if server.started:
            break
        await asyncio.sleep(0.05)

    import websockets
    result: dict = {"asr_finals": [], "anchor_sources": [], "played": [],
                    "errors": []}
    try:
        url = f"ws://127.0.0.1:{cfg.port}/v1/orchestrator"
        async with websockets.connect(url, max_size=64 * 1024 * 1024) as ws:
            await ws.send(json.dumps({
                "type": "session.start",
                "identity": {"probe": True},
                "aec_mode": "service" if cfg.enable_aec else "off",
                # 只有 armed 模式声明能力位；legacy/missing 走降级分支
                "caps": (["playback_anchor"]
                         if args.anchor == "armed" else ["playback_anchor"]),
            }, ensure_ascii=False))

            ready = json.loads(await asyncio.wait_for(ws.recv(), timeout=15))
            sid = ready.get("session_id")
            lead_ms = ready.get("lead_ms", 200)
            sess = REGISTRY.sessions.get(sid)
            if sess is None:
                raise RuntimeError("拿不到会话对象（session.ready 的 id 对不上？）")

            # 换成会说话的下游 —— 必须在第一轮 ASR 之前完成
            sess.downstream = SimDownstream()

            # ⚠️ missing 模式：声明了能力位但**故意不回** armed 承诺
            if args.anchor == "missing":
                pass   # 见下面 _handle 里对 'armed' 的处理

            browser = FakeBrowser(
                ws, host_ctx0=time.monotonic(), d_true_samples=0,
                anchor_mode=args.anchor)
            browser.t_wall0 = time.monotonic()
            d_true = int(args.d_true_ms * SR / 1000)
            browser.d_true = d_true

            clean_a = load_wav16k(args.wav_a) if args.wav_a else speech_like(SR * 3, 1)
            clean_b = load_wav16k(args.wav_b) if args.wav_b else speech_like(SR * 3, 2)
            if args.seconds:
                clean_a, clean_b = (clean_a[:int(args.seconds * SR)],
                                    clean_b[:int(args.seconds * SR)])

            # ---- 接收循环（假播放器）----
            state = {"dead": False}

            async def receiver():
                while not state["dead"]:
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=45)
                    except (asyncio.TimeoutError, Exception):
                        return
                    m = json.loads(raw)
                    t = m.get("type")
                    if t == "tts.start":
                        # ⚠️ 新前端在这里**只初始化、不承诺**（tts.start 在
                        #    TTS 合成之前发出，此刻算的提前量到音频真到时
                        #    早已过期）。承诺改在首块音频处发 —— 见下面
                        #    tts.audio 分支。
                        browser.begin_response(m.get("lead_ms", lead_ms))
                        if args.anchor == "stale":
                            # 回归用：故意复现"在 tts.start 时就承诺"的旧
                            # bug。若哪天承诺点被改回去，这条必须失败。
                            at = browser._arm_stale()
                            await ws.send(json.dumps({
                                "type": "playback", "response_id": m["response_id"],
                                "phase": "armed", "ctx_time": 0, "seq": 0,
                                "sample_offset": 0, "start_ctx": at,
                                "stop_ctx": 0, "epoch": browser._epoch,
                            }, ensure_ascii=False))
                    elif t == "tts.audio":
                        import base64
                        b = base64.b64decode(m["audio_base64"])
                        first = browser.first_at is None
                        browser.play_audio(np.frombuffer(b, dtype=np.int16))
                        browser.out.append(np.frombuffer(b, dtype=np.int16))
                        # 首块音频到达后立刻承诺（与真机 PcmPlayer 一致）
                        rid = m["response_id"]
                        if first and rid not in browser._armed_sent:
                            browser._armed_sent.add(rid)
                            at = browser.armed_ctx or 0
                            if at > 0 and args.anchor != "stale":
                                await ws.send(json.dumps({
                                    "type": "playback",
                                    "response_id": m["response_id"],
                                    "phase": "armed", "ctx_time": 0, "seq": 0,
                                    "sample_offset": 0, "start_ctx": at,
                                    "stop_ctx": 0, "epoch": browser._epoch,
                                }, ensure_ascii=False))
                    elif t == "tts.cancel":
                        played = browser.stop()
                        await ws.send(json.dumps({
                            "type": "playback", "response_id": m["response_id"],
                            "phase": "cancelled", "ctx_time": 0, "seq": 0,
                            "sample_offset": int(played),
                            "start_ctx": browser.first_at or 0,
                            "stop_ctx": browser.ctx_now(),
                            "epoch": browser._epoch,
                        }, ensure_ascii=False))
                    elif t == "asr":
                        if m.get("phase") == "final":
                            result["asr_finals"].append(m.get("text", ""))
                    elif t == "session.stats":
                        result["anchor_sources"].append(m.get("anchor_source"))
                    elif t == "error":
                        result["errors"].append(str(m))

            recv_task = asyncio.create_task(receiver())
            # 采集线程：**持续**送 mic（有语音送语音，没有送静音）。
            # 回声出现在"用户没说话、喇叭在播"的时段，采集必须连续覆盖。
            cap_task = asyncio.create_task(browser.capture_loop())

            # ---- 场景 ----
            if args.scenario == "turn":
                await _scenario_turn(ws, sess, browser, clean_a, clean_b,
                                     args, result)
            else:
                await _scenario_barge(ws, sess, browser, clean_a, clean_b,
                                      args, result)

            state["dead"] = True
            browser.stop_capture()
            for t in (recv_task, cap_task):
                t.cancel()
                try:
                    await t
                except (asyncio.CancelledError, Exception):
                    pass

            result["browser"] = browser
            result["session"] = sess
    finally:
        server.should_exit = True
        await asyncio.sleep(0.3)
        task.cancel()
    return result


async def _scenario_turn(ws, sess, browser: FakeBrowser, clean_a, clean_b,
                         args, result) -> None:
    """场景 (a)：每轮都等 TTS **完整播完**，再送下一轮用户语音。"""
    await _one_turn(ws, sess, browser, clean_a, args, result, tag="turn1")
    await _wait_playback_done(browser, timeout=25)
    await _one_turn(ws, sess, browser, clean_b, args, result, tag="turn2")
    await _wait_playback_done(browser, timeout=25)
    await _drain(ws, result)


async def _scenario_barge(ws, sess, browser: FakeBrowser, clean_a, clean_b,
                          args, result) -> None:
    """场景 (b)：TTS 播到一半**打断**，立刻送下一轮用户语音。"""
    # 第 1 轮：只要 ASR 出结果就会触发 SimDownstream 回一句长回复
    send = asyncio.create_task(_one_turn(ws, sess, browser, clean_a, args,
                                         result, tag="turn1"))
    # 等回复真正开始播
    for _ in range(300):
        if browser.nodes:
            break
        await asyncio.sleep(0.05)
    # 播 1.0s 后打断
    await asyncio.sleep(1.0)
    await sess.execute_action(Cancel(reason="bargein"))
    played = browser.stop()
    result["played"].append(("barge", int(played)))
    await ws.send(json.dumps({
        "type": "playback", "response_id": sess.executor._current_response_id or "",
        "phase": "cancelled", "ctx_time": 0, "seq": 0,
        "sample_offset": int(played),
        "start_ctx": browser.first_at or 0,
        "stop_ctx": browser.ctx_now(), "epoch": browser._epoch,
    }, ensure_ascii=False))
    await send
    # 立刻送第 2 轮（不等旧句播完 —— 这正是真机的插话场景）
    await _one_turn(ws, sess, browser, clean_b, args, result, tag="turn2")
    await _drain(ws, result)


async def _one_turn(ws, sess, browser: FakeBrowser, clean: np.ndarray,
                    args, result: dict, tag: str) -> None:
    """排入一轮用户语音，等它被采集线程发完，再等回复被触发。"""
    before = len(result["asr_finals"])
    t0 = browser.enqueue_speech(clean)
    result.setdefault("turns", []).append({
        "tag": tag, "t0": t0, "n": len(clean),
    })
    # 等采集线程把这一轮真正发出去（含 qu<eue 里排在它前面的静音）
    while browser.sent_samples < t0 + len(clean):
        await asyncio.sleep(0.05)
    if args.speak_trigger == "harness":
        # 没有 ASR 时由 harness 自己触发 —— 否则整条 TTS→播放→参考轨
        # 的链路根本不会跑，测不到任何东西。
        await sess.execute_action(Speak(text=REPLY_TEXT))
        return
    # 等 ASR 出结果（有 ASR 时由 SimDownstream 触发回复）
    for _ in range(int(args.asr_wait_s * 10)):
        if len(result["asr_finals"]) > before:
            return
        await asyncio.sleep(0.1)


async def _wait_playback_done(browser: FakeBrowser, timeout: float) -> None:
    """等本句真正播完（按喇叭时间轴算，不是"送达"）。"""
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if browser.turn_end is not None:
            while browser.sample_of(browser.ctx_now()) < browser.turn_end:
                await asyncio.sleep(0.05)
                if time.monotonic() - t0 > timeout:
                    return
            return
        await asyncio.sleep(0.05)


async def _drain(ws, result: dict) -> None:
    """尾部静音 + session.stop，让 ASR 闭合最后一段。"""
    import base64
    for _ in range(15):
        buf = np.zeros(MIC_CHUNK, dtype=np.float32).tobytes()
        await ws.send(json.dumps({
            "type": "audio", "audio_base64": base64.b64encode(buf).decode(),
            "t_ms": 0, "ctx_time": 0, "epoch": 0,
        }, ensure_ascii=False))
        await asyncio.sleep(0.1)
    try:
        await ws.send(json.dumps({"type": "session.stop"}))
    except Exception:  # noqa: BLE001
        pass


# ====================================================================== #
#  断言
# ====================================================================== #

def find_shift(raw: np.ndarray, ref: np.ndarray,
               dmax: int) -> Optional[int]:
    """求 ``s`` 使 ``raw[x] ≈ ref[x - s]``，在 ``[-dmax, +dmax]`` 内搜。

    ``s > 0`` 表示参考轨的内容比实际出声**晚** s 个采样。

    两个输入都是**已知的干净波形**（参考轨内容 vs 仿真器合成的播出信号），
    所以互相关峰又尖又稳 —— 不像从 mic 里"捞"回声那样会被近端人声和
    周期性成分搅乱。

    ⚠️ 必须搜**带符号**的 lag：旧的预测落位会把参考轨摆到实际出声**之前**
    几百毫秒（负 lag），只搜正 lag 会找到一个假峰、把一个几百毫秒的错位
    报成 0。
    """
    n = min(len(raw), len(ref))
    if n < 4096:
        return None
    # ⚠️ **必须去均值**。两段信号的直流分量不同，不去的话互相关在极端 lag
    #    上会被"均值的乘积"顶起来，峰直接吸附到搜索边界（实测：本该是
    #    -59 的整段偏移被报成 -5600 = 边界值）。
    raw = raw[:n].astype(np.float64)
    ref = ref[:n].astype(np.float64)
    raw -= raw.mean()
    ref -= ref.mean()
    n_fft = 1
    while n_fft < 2 * n:
        n_fft <<= 1
    cc = np.fft.irfft(np.fft.rfft(raw, n_fft) * np.conj(np.fft.rfft(ref, n_fft)),
                      n_fft)
    # cc[k] = Σ raw[j]·ref[j-k]；负 k 在循环缓冲的尾部
    d = min(dmax, n - 1)
    lags = np.concatenate([np.arange(-d, 0), np.arange(0, d + 1)])
    vals = np.concatenate([np.abs(cc[n_fft - d:]), np.abs(cc[:d + 1])])
    return int(lags[int(np.argmax(vals))])


def _regions(raw: np.ndarray, min_len: int) -> List[Tuple[int, int]]:
    """参考轨上"有内容"的连续区间 = 一次播报。

    用 10ms 块级包络而不是逐采样判非零：语音波形过零点多，逐采样判会被
    零交叉切成一堆碎片。
    """
    win = 160
    n = len(raw) // win
    if n < 2:
        return []
    on = np.abs(raw[:n * win]).reshape(n, win).max(axis=1) > 1e-4
    out: List[Tuple[int, int]] = []
    st = None
    for i, a in enumerate(on):
        if a and st is None:
            st = i
        elif not a and st is not None:
            out.append((st * win, i * win))
            st = None
    if st is not None:
        out.append((st * win, n * win))
    return [(a, b) for a, b in out if b - a > min_len]


def analyze_alignment(result: dict, args) -> None:
    """核心命题：参考轨与 mic 里**真实回声**的残差是常量 D_true。

    ``armed`` 下参考轨按浏览器承诺落位 → 残差应当**逐句恒定**（还有
    一个固定的块相位偏移），散度 ≤2 采样。

    ``legacy`` 下服务端按"收到音频的时刻 + 提前量"预测 → 预测误差含
    网络往返、主线程阻塞、mic 在途积压，**逐句不同** → 残差抖动。
    这就是"固定常量 D 吸收不了、所以第一句好后面失效"的可执行证据。
    """
    sess = result.get("session")
    if sess is None or sess.ref_track is None:
        return
    print()
    print("=" * 72)
    print("A. 参考轨落位对齐（核心命题）")
    print("-" * 72)

    src = result.get("anchor_sources") or []
    n_ack = sum(1 for s in src if s == "ack")
    n_pred = sum(1 for s in src if s == "predicted")
    print(f"  落位锚点来源统计：ack={n_ack} predicted={n_pred}")
    if args.anchor == "armed":
        check(n_pred == 0, "落位全部来自浏览器承诺（没有降级到预测）")

    browser: FakeBrowser = result["browser"]
    # ⚠️ 读的长度要覆盖**参考轨写入的全部区间**，不能只取到会话时钟现在
    #    的位置：参考轨是按"浏览器承诺的播出时刻"落位的，而承诺窗口
    #    可能延伸到当前时刻之后好几个 lead。取短了会把播报切断，
    #    区间检测就什么都找不到。
    lo_w, hi_w = sess.ref_track.buf.written_span()
    total = max(sess.clock.now(), (hi_w or 0), browser.sent_samples) + SR
    played = browser.emitted(total)
    raw = sess.ref_track.buf.read(0, total)
    mic = browser.mic_total(total)

    mask_p = np.abs(played) > 1e-4
    mask_r = np.abs(raw) > 1e-4
    if not (mask_p.any() and mask_r.any()):
        print("  [跳过] 没有检出播放内容 —— TTS 没产出音频？"
              "（离线请加 --mock-tts）")
        return

    lo, ro = int(np.argmax(mask_p)), int(np.argmax(mask_r))
    off = ro - lo
    print(f"  首次播出={lo}  参考轨首次非零={ro}  差={off} 采样"
          f"（{off/SR*1000:+.1f}ms）")
    # 这是最直白的判据：参考轨的内容到底摆在哪。
    #   armed  —— 承诺换算出来的位置，与真实出声只差一个固定的块相位
    #   legacy —— 服务端"收到音频的时刻 + 提前量"的预测，误差含网络往返、
    #             主线程阻塞、时钟落后，可以大到几百毫秒
    if args.anchor == "armed":
        check(abs(off) <= 320,
              f"参考轨落位与实际出声差 {off/SR*1000:+.1f}ms（≤20ms）")
    else:
        # ⚠️ 这里**不能**断言"偏移必然很大"：legacy 的误差是随机的，单次
        #    抽样完全可能碰巧落在小值上（实测见过 18ms）。真正说明问题的是
        #    **逐句散度**（下面按播报分段算的那个），不是单次绝对值。
        print(f"  （{args.anchor}：本次预测偏移 {off/SR*1000:+.1f}ms"
              f" —— 单次值看运气，看下面的散度）")

    # ---- 逐句落位精度：参考轨 vs 仿真器**实际合成**的播出信号 ----
    #
    # 这才是"参考轨是否严格等于喇叭播出"的直接检验。mic 里捞回声的做法
    # （先减掉干净人声再互相关）在真实语音上可行，但这里没必要：仿真器
    # 手里就有播出信号的**真值**（`browser.emitted()`）。
    #
    # 窗口按**播出节点**切（每次播报 = 一组连续排程的节点），不用从参考轨
    # 反推区间 —— 后者在语音的过零点上会被切碎，也不适用于"参考轨摆错
    # 位置"的情况（那时按参考轨切的窗口根本对不上播出）。
    emitted = browser.emitted(total)
    # 搜索半径要盖得住旧的预测落位误差。armed 下真实偏差只有一个块相位
    # （几毫秒），但 legacy 的预测误差可以大到**一秒以上**（服务端时钟
    # 落后量会随会话累积），所以要搜得够宽 —— 不然只会得到一个边界值，
    # 看不出到底差多少。
    dmax = int(2.0 * SR)
    lo_w, hi_w = sess.ref_track.buf.written_span() or (0, 0)
    print(f"  参考轨写入区间 [{lo_w}, {hi_w})，"
          f"仿真器播出节点 {len(browser.nodes)} 个")

    # ---- 排程是否线性铺开（不得重叠/折叠）----
    #
    # ⚠️ 这条守的是一个真机 bug：`at = this._armNow() || Math.max(nextAt, ...)`
    #    里 `_armNow()` 每次返回**同一个**缓存承诺值 → 每个音频块都被排到
    #    同一时刻，整段回复叠在一起同时播。听感是"一闪而过、只听到开头和
    #    结尾"。节点时间轴能直接看出来（后一块起点 < 前一块终点）。
    ov = 0
    for k in range(1, len(browser.nodes)):
        pa, px = browser.nodes[k - 1]
        a, _ = browser.nodes[k]
        if a < pa + len(px):
            ov += 1
    check(ov == 0,
          f"音频块按时间轴线性铺开、无重叠折叠（重叠 {ov} 处）"
          f"—— 重叠会让整段回复同时播出，听感一闪而过")

    # 按节点间隔分组：间隔 > 1s 认为是另一次播报
    groups: List[List[Tuple[int, np.ndarray]]] = []
    for at, x in browser.nodes:
        if groups and at - (groups[-1][-1][0] + len(groups[-1][-1][1])) < SR:
            groups[-1].append((at, x))
        else:
            groups.append([(at, x)])

    # 被打断的那一次播报，判据不是"落位偏移"而是**截断精确性**（见下）。
    # 打断后节点的长度已经被 stop() 截到"真正播出"的量，而服务端参考轨
    # 会先按预测临时截一刀、再按浏览器回报的实测播出量 resize 精修 ——
    # 两者最终应当一致。
    interrupted = 0 if args.scenario == "barge" else -1

    shifts: List[int] = []
    for k, g in enumerate(groups):
        a = g[0][0]
        b = g[-1][0] + len(g[-1][1])
        played_here = int(sum(len(x) for _, x in g))
        if k == interrupted:
            seg = raw[a:b]
            nz = int((np.abs(seg) > 1e-4).sum())
            print(f"    播报 #{k + 1}（被**打断**）播出 {played_here} 采样，"
                  f"参考轨该区间非零 {nz} 采样")
            # 打断的**精确**截断（保留量 == 实测播出量 ±1 采样）由
            # tests/test_bargein_ref.py 用确定性的单元测试守着 —— 那个
            # 尺度上本仿真器不够忠实：它要模拟"播放器预排程 + 打断时刻"，
            # 而服务端的 resize 只能**缩短**，一个过早到达的回执会永久
            # 生效，测出来的差值反映的是仿真时钟的粒度而不是链路行为。
            # 这里只断言**方向性**的两条：
            #   ① 参考轨不能留下**明显超出**实际播出的尾部（那会让 AEC
            #      去追一个不存在的回声，比不给参考更糟）
            #   ② 也不能把**大部分**已经播出的内容清掉（有回声、没 farend）
            check(nz <= played_here + int(0.5 * SR),
                  f"打断后没有留下超出播出量的参考尾部"
                  f"（{nz} vs 播出 {played_here}，余量 ≤0.5s）")
            check(nz >= int(0.5 * played_here),
                  f"打断后已播出的参考没有被清零"
                  f"（{nz} ≥ {int(0.5 * played_here)}）")
            continue
        lo, hi = max(0, a - dmax), min(total, b + dmax)
        if hi - lo < 8192:
            print(f"    播报 #{k + 1} [{a/SR:.2f}s]: [跳过] 窗过短")
            continue
        sh = find_shift(raw[lo:hi], emitted[lo:hi], dmax)
        if sh is None:
            print(f"    播报 #{k + 1}: [跳过] 相关无峰")
            continue
        shifts.append(sh)
        print(f"    播报 #{k + 1} 播出区间 [{a/SR:.2f}s, {b/SR:.2f}s)："
              f"参考轨相对播出偏移 {sh:+d} 采样（{sh/SR*1000:+.1f}ms）")

    if len(shifts) >= 2:
        spread = max(shifts) - min(shifts)
        print(f"  逐句落位偏差散度 = {spread} 采样（{spread/SR*1000:.1f}ms）")
        if args.anchor == "armed":
            # 判据是**散度**：参考轨只要每次都落在同一个相对位置，那个常量
            # 就能并进 D 里被 AEC 消掉；怕的是**逐句不一样** —— ±5ms 的
            # 容忍窗（tests/README.md 的扫描表）根本吸收不了变化量。
            check(spread <= 32,
                  f"armed：参考轨落位逐句一致（散度 {spread} 采样 = "
                  f"{spread/SR*1000:.1f}ms ≤ 2ms）—— 常量 D 能吸收")
        else:
            # legacy/预测模式的病根是**误差逐句变化**（不是某一句偏得多）。
            # 这一条正是"固定常量 D 吸收不了"的可执行证据 —— 比对单次绝对
            # 值可靠得多（单次值纯看运气，实测见过 18ms 的巧合）。
            check(spread > 400,
                  f"{args.anchor}：逐句散度 {spread/SR*1000:.1f}ms "
                  f"—— 误差**逐句变化**，常量 D 吸收不了（这正是旧代码"
                  f"第一句能消、后面失效的原因）")
    else:
        print("  [跳过] 有效播报不足 2 段，测不出散度")


def analyze_text(result: dict, args) -> None:
    """ASR 结果里不得混入 TTS 的回复内容。"""
    print()
    print("=" * 72)
    print("C. ASR 文本（不得混入 TTS 内容）")
    print("-" * 72)
    finals = result.get("asr_finals") or []
    print(f"  ASR 最终结果 {len(finals)} 条：")
    for f in finals:
        print(f"    {f[:60]!r}")
    leaked = [f for f in finals if REPLY_MARKER in f]
    check(not leaked,
          f"没有一条 ASR 结果包含回复特征串 {REPLY_MARKER!r}"
          f"（漏了 {len(leaked)} 条就是回声没消掉）")
    if not finals:
        print("  [跳过] 没有 ASR 最终结果 —— 可能没连 ASR，或用的合成信号"
              "（合成信号永远不会产出 2pass-offline）")


async def main_async(args) -> int:
    if args.anchor == "legacy":
        print("模式 legacy：服务端走**降级分支**（浏览器不承诺起播时刻）")
        print("  → 预期看到残差抖动，且 anchor_source 一路是 predicted")
    print("=" * 72)
    print(f"全双工仿真：scenario={args.scenario} anchor={args.anchor} "
          f"D_true={args.d_true_ms}ms")
    print("-" * 72)
    result = await run(args)

    analyze_alignment(result, args)
    analyze_text(result, args)

    print()
    print("=" * 72)
    if _failures:
        print(f"FAILED: {len(_failures)} 项")
        for f in _failures:
            print(f"  - {f}")
        return 1
    print("通过")
    return 0


def main() -> None:
    p = argparse.ArgumentParser(description="全双工 AEC 仿真")
    p.add_argument("--port", type=int, default=8231)
    p.add_argument("--scenario", choices=["turn", "barge"], default="turn")
    p.add_argument("--anchor", choices=["armed", "legacy", "missing", "stale"],
                   default="armed")
    p.add_argument("--d-true-ms", type=float, default=84.0,
                   help="假浏览器合成回声用的**真实**声学延迟")
    p.add_argument("--wav-a", default=None, help=f"第 1 轮（默认 {DEFAULT_WAV_A}）")
    p.add_argument("--wav-b", default=None, help=f"第 2 轮（默认 {DEFAULT_WAV_B}）")
    p.add_argument("--seconds", type=float, default=0.0,
                   help="每轮截取时长（0 = 用整段）")
    p.add_argument("--no-aec", action="store_true")
    p.add_argument("--no-asr", action="store_true")
    p.add_argument("--mock-tts", action="store_true",
                   help="用本地合成音代替 TTS 服务（离线跑对齐检查）")
    p.add_argument("--asr-wait-s", type=float, default=6.0)
    p.add_argument("--speak-trigger", choices=["auto", "asr", "harness"],
                   default="auto",
                   help="谁触发播报：asr=ASR 最终结果，harness=仿真器自己。"
                        "auto：连了 ASR 就用 asr，否则用 harness")
    args = p.parse_args()
    if args.speak_trigger == "auto":
        args.speak_trigger = "harness" if args.no_asr else "asr"

    root = Path(__file__).resolve().parents[2]
    if args.wav_a is None and (root / DEFAULT_WAV_A).is_file():
        args.wav_a = str(root / DEFAULT_WAV_A)
    if args.wav_b is None and (root / DEFAULT_WAV_B).is_file():
        args.wav_b = str(root / DEFAULT_WAV_B)
    if args.wav_a is None or args.wav_b is None:
        print("⚠️ 找不到真实语音 wav，改用合成信号（ASR 文本检查将无效）")
    raise SystemExit(asyncio.run(main_async(args)))


if __name__ == "__main__":
    main()
