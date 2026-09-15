"""独立的延迟校准通道 —— **不需要建立完整会话**。

## 为什么单独做一条通道

校准只需要三件事：① 给浏览器一段已知音频去播 ② 收麦克风 ③ 算互相关。
它不需要 ASR / OmniLLM / TTS / 人脸 —— 走完整会话既慢又要求用户先
"开始会话"，交互上说不通。

## 流程

    浏览器连 /v1/calibrate
      → 服务端回一段**双段啁啾**（走与 TTS 相同的播放路径）
      → 浏览器播放它，**同时**持续回传 16kHz 麦克风
      → 浏览器在起播时报 calibrate.anchor（真实起播位置）
      → 服务端对每段独立做互相关 + 自洽性校验 → 回结果
      → 断开

## 两个曾经的坑（现在都由协议保证，不再靠约定）

  ① **采样率**：前端拿的是设备默认率（通常 48k）的 AudioContext，而
     服务端按 16k 解析 —— 3× 时间尺度错误。现在协议**必须**带
     ``sample_rate``，前端负责重采样到 16k 回传，不一致直接报错。

  ② **播放起点**：早先用配置常量 ``playback_delay_ms`` 当播放起点，而
     校准路径根本没有那个提前量，且前端只在起播后才回传麦克风 ——
     净延迟恒为负。现在由前端上报 ``calibrate.anchor``，拿不到就拒绝
     给结论。
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
from typing import Optional

import numpy as np

from .calibrate import CAL_MIC_SR, CAL_SR, CalibrationSession

logger = logging.getLogger(__name__)

# 前端回传的麦克风采样率必须是这个值（协议校验）
EXPECTED_MIC_SR = CAL_MIC_SR


class CalibrationHandler:
    """一条校准连接的处理逻辑。"""

    def __init__(self) -> None:
        self._session: Optional[CalibrationSession] = None
        self.error: Optional[str] = None

    def start(self) -> dict:
        """返回给浏览器的开始指令（含要播放的音频）。"""
        self._session = CalibrationSession(sr=CAL_MIC_SR)
        signal, starts = self._session.start()
        logger.info("校准开始：播放 %.2fs 双段啁啾，等麦克风回采",
                    len(signal) / CAL_SR)
        return {
            "type": "calibrate.play",
            "sample_rate": CAL_SR,
            "audio_base64": base64.b64encode(signal.tobytes()).decode(),
            "duration_s": len(signal) / CAL_SR,
            "segments": len(starts),
            "expect_mic_s": round(
                len(signal) / CAL_SR + 1.5, 2),   # 含最大搜索延迟
        }

    def set_anchor(self, n_samples: int) -> None:
        if self._session is not None:
            self._session.set_play_anchor(n_samples)

    def feed(self, pcm_b64: str, sample_rate: int) -> None:
        """收浏览器回传的麦克风 PCM。

        ⚠️ ``sample_rate`` 必须显式给出且等于 16k —— 早先协议里根本没有
        这个字段，48k 的数据被当 16k 解析，时间尺度整体错 3 倍。
        """
        if self._session is None:
            return
        if int(sample_rate) != EXPECTED_MIC_SR:
            self.error = (
                f"麦克风回传采样率 {sample_rate} ≠ {EXPECTED_MIC_SR} —— "
                "前端必须重采样到 16kHz 再回传（否则延迟测量会整体错倍）"
            )
            raise ValueError(self.error)
        raw = base64.b64decode(pcm_b64)
        x = np.frombuffer(raw, dtype=np.float32)
        self._session.feed(x)

    def ready(self) -> bool:
        return self._session is not None and self._session.ready()

    def min_needed_s(self) -> float:
        return self._session.min_needed_s() if self._session else 0.0

    def finish(self) -> dict:
        """结算。返回 ``{ok, delay_ms, ...}``。"""
        if self._session is None:
            return {"ok": False, "error": "校准未开始"}
        res = self._session.finish()
        self._session = None
        if res is None:
            return {"ok": False, "error": "麦克风数据不足"}
        if not res.get("ok"):
            out = {"ok": False}
            out["error"] = res.get("reason") or res.get("error") \
                or "校准失败，请重试"
            for k in ("delay_ms", "peak_ratio", "segments", "spread_ms",
                      "mic_rms", "sig_rms", "noise_rms", "snr_db"):
                if res.get(k) is not None:
                    out[k] = res[k]
            logger.info("校准结果: %s", out)
            return out
        out = {
            "ok": True,
            "delay_ms": round(float(res["delay_ms"]), 1),
            "delay_samples": int(res["delay_samples"]),
            "peak_ratio": res.get("peak_ratio", 0),
            "segments": res.get("segments", 0),
            "spread_ms": res.get("spread_ms", 0),
            "mic_rms": res.get("mic_rms", 0),
        }
        logger.info("校准结果: %s", out)
        return out


async def handle_calibrate_ws(ws, cfg) -> None:
    """``/v1/calibrate`` 的完整处理。

    协议：
      客户端 → ``{"type":"calibrate.start", "client_key": ...}``
      服务端 → ``{"type":"calibrate.play", audio_base64, sample_rate, ...}``
      客户端 → ``{"type":"calibrate.anchor", "play_offset_samples": N}``
      客户端 → ``{"type":"calibrate.mic", audio_base64, "sample_rate":16000}`` × N
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
                {"ok": False, "error": "未收到 calibrate.start"},
                ensure_ascii=False))
            return
        m0 = json.loads(first)
        if m0.get("type") != "calibrate.start":
            await ws.send_text(json.dumps(
                {"ok": False,
                 "error": f"首条消息必须是 calibrate.start，收到 {m0.get('type')!r}"},
                ensure_ascii=False))
            return
        client_key = str(m0.get("client_key") or "default")
        handler = CalibrationHandler()
        await ws.send_text(json.dumps(handler.start()))

        while True:
            try:
                raw = await asyncio.wait_for(ws.receive_text(), timeout=20)
            except asyncio.TimeoutError:
                # 浏览器停发了但还没 ready —— **不要干等**。若已有可用数据
                # 就试着结算（可能仍能算出），否则给出明确诊断。
                if handler is not None:
                    got_s = mic_pkts * 0.1
                    logger.warning("校准超时：收到 %.1fs 麦克风数据（需要 %.1fs）",
                                   got_s, handler.min_needed_s())
                    res = handler.finish() if mic_pkts > 0 else {
                        "ok": False,
                        "error": f"只收到 {got_s:.1f}s 麦克风数据，"
                                 "浏览器可能未回传音频",
                    }
                    await ws.send_text(json.dumps(res, ensure_ascii=False))
                return
            msg = json.loads(raw)
            t = msg.get("type")

            if t == "calibrate.anchor":
                if handler is not None:
                    handler.set_anchor(int(msg.get("play_offset_samples") or 0))

            elif t == "calibrate.mic":
                if handler is not None:
                    try:
                        handler.feed(msg.get("audio_base64", ""),
                                     msg.get("sample_rate", 0))
                    except ValueError as exc:
                        # 协议违规（采样率不对）：发了错误就**断开**。
                        # 只发错误继续循环的话，前端会一直回传、一直报错，
                        # 直到自己超时 —— 表现为"卡住 30 秒"。
                        await ws.send_text(json.dumps(
                            {"ok": False, "error": str(exc)},
                            ensure_ascii=False))
                        try:
                            await ws.close()
                        except Exception:  # noqa: BLE001
                            pass
                        return
                    mic_pkts += 1
                    if mic_pkts % 20 == 0:
                        logger.info("校准：已收 %d 个麦克风包（约 %.1fs）",
                                    mic_pkts, mic_pkts * 0.1)
                    if handler.ready():
                        res = handler.finish()
                        if res.get("ok"):
                            saved = store.put(client_key, res["delay_ms"],
                                              n_samples=res.get("segments", 1) * 10)
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
