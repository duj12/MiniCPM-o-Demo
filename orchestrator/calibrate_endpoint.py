"""独立的延迟校准通道 —— **不需要建立完整会话**。

## 为什么单独做一条通道

校准只需要三件事：① 给浏览器一段已知音频去播 ② 收麦克风 ③ 算互相关。
它不需要 ASR / OmniLLM / TTS / 人脸 —— 走完整会话既慢又要求用户先
"开始会话"，交互上说不通。

## 流程

    浏览器连 /v1/calibrate
      → 服务端回一段校准音频（**走与 TTS 相同的播放路径**）
      → 浏览器播放它，同时把麦克风 PCM 发回来
      → 服务端互相关求延迟 → 回结果
      → 断开

整程约 3 秒，用户只需授权麦克风 + 点一次按钮。

## 校准信号

默认用一段**真实语音**（比啁啾更接近实际使用场景，且用户听起来自然）；
互相关时用它的波形即可 —— 语音虽准周期，但取 1.5s 整段做宽带相关，
主峰仍足够突出。若置信度低可换用啁啾（``--chirp``）。
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np

from .calibrate import CAL_SR, CalibrationSession, make_chirp

logger = logging.getLogger(__name__)

SR = 16000
# 默认校准音频（仓库自带的测试语音）
DEFAULT_CAL_WAV = "assets/ref_audio/ref_minicpm_signature.wav"


def load_calibration_audio(wav_path: Optional[str] = None,
                           duration_s: float = 1.5,
                           prefer_chirp: bool = False) -> np.ndarray:
    """返回 24kHz int16 的校准音频。

    优先用真实语音（更贴近实际使用；用户听起来是一句正常的话），
    读不到则退回啁啾。
    """
    if not prefer_chirp and wav_path:
        p = Path(wav_path)
        if not p.is_absolute():
            p = Path(__file__).resolve().parents[1] / p
        if p.is_file():
            try:
                import wave
                with wave.open(str(p), "rb") as w:
                    src_sr = w.getframerate()
                    raw = w.readframes(int(w.getnframes()))
                    x = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
                    if w.getnchannels() == 2:
                        x = x.reshape(-1, 2).mean(axis=1)
                # 重采样到 24kHz（TTS 的输出采样率，走同一条播放路径）
                if src_sr != CAL_SR:
                    n_out = int(len(x) * CAL_SR / src_sr)
                    pos = np.linspace(0, len(x) - 1, n_out)
                    i0 = np.floor(pos).astype(int)
                    i1 = np.minimum(i0 + 1, len(x) - 1)
                    fr = (pos - i0).astype(np.float32)
                    x = x[i0] * (1 - fr) + x[i1] * fr
                x = x[: int(CAL_SR * duration_s)]
                # 淡入淡出，避免爆音
                ramp = int(0.01 * CAL_SR)
                if ramp and 2 * ramp < len(x):
                    x[:ramp] *= np.linspace(0, 1, ramp)
                    x[-ramp:] *= np.linspace(1, 0, ramp)
                logger.info("校准音频：%s（%.2fs @24k）", p.name, len(x) / CAL_SR)
                return x.astype(np.int16)
            except Exception as exc:  # noqa: BLE001
                logger.warning("校准音频读取失败（%s），改用啁啾", exc)
    logger.info("校准音频：宽带啁啾（%.2fs @24k）", duration_s)
    return make_chirp(duration_s=duration_s)


class CalibrationHandler:
    """一条校准连接的处理逻辑。"""

    def __init__(self, wav_path: Optional[str] = None,
                 playback_delay_ms: float = 200.0,
                 prefer_chirp: bool = False) -> None:
        self.wav_path = wav_path
        self.playback_delay_ms = playback_delay_ms
        self.prefer_chirp = prefer_chirp
        self.audio = load_calibration_audio(wav_path, prefer_chirp=prefer_chirp)
        # 校准用的互相关参考：把播放音频降到 16k
        x = self.audio.astype(np.float32) / 32768.0
        self.ref16 = self._resample(x, CAL_SR, SR)
        self._session: Optional[CalibrationSession] = None

    @staticmethod
    def _resample(x: np.ndarray, src: int, dst: int) -> np.ndarray:
        if src == dst:
            return x
        n_out = int(len(x) * dst / src)
        pos = np.linspace(0, len(x) - 1, n_out)
        i0 = np.floor(pos).astype(int)
        i1 = np.minimum(i0 + 1, len(x) - 1)
        fr = (pos - i0).astype(np.float32)
        return (x[i0] * (1 - fr) + x[i1] * fr).astype(np.float32)

    # ------------------------------------------------------------------ #

    def start(self) -> dict:
        """返回给浏览器的开始指令（含要播放的音频）。"""
        self._session = CalibrationSession(sr=SR)
        # 用我们自己的参考波形覆盖（可能是语音而非啁啾）
        self._session.chirp16 = self.ref16
        self._session.start()
        logger.info("校准开始：播放 %.2fs，等麦克风回采",
                    len(self.audio) / CAL_SR)
        return {
            "type": "calibrate.play",
            "sample_rate": CAL_SR,
            "audio_base64": base64.b64encode(self.audio.tobytes()).decode(),
            "duration_s": len(self.audio) / CAL_SR,
            "expect_mic_s": round(
                len(self.audio) / CAL_SR + 1.5, 2),   # 含最大搜索延迟
        }

    def feed(self, pcm_b64: str) -> None:
        """收浏览器回传的麦克风 PCM（16k float32 base64）。"""
        if self._session is None:
            return
        raw = base64.b64decode(pcm_b64)
        x = np.frombuffer(raw, dtype=np.float32)
        self._session.feed(x)

    def ready(self) -> bool:
        return self._session is not None and self._session.ready()

    def finish(self) -> dict:
        """结算。返回 ``{ok, delay_ms, ...}``。"""
        if self._session is None:
            return {"ok": False, "error": "校准未开始"}
        res = self._session.finish(playback_delay_ms=self.playback_delay_ms)
        self._session = None
        if res is None:
            return {"ok": False, "error": "麦克风数据不足"}
        ok = bool(res.get("ok"))
        out = {
            "ok": ok,
            "delay_ms": round(float(res["delay_ms"]), 1),
            "delay_samples": int(res["delay_samples"]),
            "peak_ratio": res.get("peak_ratio", 0),
        }
        if not ok:
            out["error"] = ("置信度低 —— 请确认手机外放（不是耳机）、"
                            "环境较安静、音量适中后重试")
        logger.info("校准结果: %s", out)
        return out


async def handle_calibrate_ws(ws, cfg) -> None:
    """``/v1/calibrate`` 的完整处理。

    协议：
      客户端 → ``{"type":"calibrate.start"}``
      服务端 → ``{"type":"calibrate.play", audio_base64, sample_rate, ...}``
      客户端 → ``{"type":"calibrate.mic", audio_base64}`` × N
      服务端 → ``{"type":"calibrate.done", ok, delay_ms, ...}``
    """
    from orchestrator.delay_store import DelayStore

    store = DelayStore(cfg.delay_store_path, cfg.aec_default_delay_ms)
    handler: Optional[CalibrationHandler] = None
    client_key = "default"
    mic_pkts = 0

    try:
        # 首条必须是 calibrate.start —— 超时给出明确提示（而不是静默断开，
        # 那样前端只看到"连上了但没反应"，很难排查）
        try:
            first = await asyncio.wait_for(ws.receive_text(), timeout=10)
        except asyncio.TimeoutError:
            logger.warning("校准连接 10s 未收到 calibrate.start —— 断开")
            await ws.send_text(json.dumps(
                {"ok": False, "error": "未收到 calibrate.start"}, ensure_ascii=False))
            return
        m0 = json.loads(first)
        if m0.get("type") != "calibrate.start":
            await ws.send_text(json.dumps(
                {"ok": False,
                 "error": f"首条消息必须是 calibrate.start，收到 {m0.get('type')!r}"},
                ensure_ascii=False))
            return
        client_key = str(m0.get("client_key") or "default")
        handler = CalibrationHandler(
            wav_path=DEFAULT_CAL_WAV,
            playback_delay_ms=cfg.playback_delay_ms,
            prefer_chirp=bool(m0.get("chirp")),
        )
        await ws.send_text(json.dumps(handler.start()))

        while True:
            try:
                raw = await asyncio.wait_for(ws.receive_text(), timeout=15)
            except asyncio.TimeoutError:
                # 浏览器停发了但还没 ready —— **不要干等**。若已有可用数据
                # 就试着结算（可能仍能算出），否则给出明确诊断。
                if handler is not None:
                    got_s = mic_pkts * 0.1
                    logger.warning("校准超时：收到 %.1fs 麦克风数据（需要 %.1fs）",
                                   got_s, handler._session.min_needed_s()
                                   if handler._session else 0)
                    res = handler.finish() if mic_pkts > 0 else {
                        "ok": False,
                        "error": f"只收到 {got_s:.1f}s 麦克风数据，"
                                 "浏览器可能未回传音频",
                    }
                    await ws.send_text(json.dumps(res, ensure_ascii=False))
                return
            msg = json.loads(raw)
            t = msg.get("type")

            if t == "calibrate.mic":
                if handler is not None:
                    handler.feed(msg.get("audio_base64", ""))
                    mic_pkts += 1
                    if mic_pkts % 10 == 0:
                        logger.info("校准：已收 %d 个麦克风包（约 %.1fs）",
                                    mic_pkts, mic_pkts * 0.1)
                    if handler.ready():
                        res = handler.finish()
                        if res.get("ok"):
                            saved = store.put(client_key, res["delay_ms"],
                                              n_samples=10)
                            res["saved"] = bool(saved)
                            res["source"] = "calibrated"
                            if not saved:
                                res["ok"] = False
                                res["error"] = (
                                    f"测得 {res['delay_ms']:.0f} ms 超出合理"
                                    "区间（10~700ms），多半是相关峰选错。"
                                    "请确认外放、环境安静后重试。")
                        await ws.send_text(json.dumps(res, ensure_ascii=False))
                        handler = None

            elif t == "calibrate.abort":
                handler = None
                await ws.send_text(json.dumps({"ok": False,
                                               "error": "已取消"}))

            elif t == "bye":
                break

    except Exception as exc:  # noqa: BLE001
        logger.info("校准连接结束: %s", exc)
