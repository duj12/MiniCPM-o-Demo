"""Half-duplex streaming: VAD + TurnSense turn-decision state machine.

This implements the `turn_decision="vad_turnsense"` strategy on top of the
existing full_duplex transport.  The worker streams every incoming
audio/video chunk to the C++ backend as a `force_listen=true` prefill
(accumulating into KV cache — "边听边看"), and only triggers a model reply
(decode) when:

  1. StreamingVAD detects a speech-pause (silence threshold exceeded), AND
  2. TurnSense judges the accumulated utterance semantically complete.

During the model reply (GENERATING state) the worker keeps feeding VAD to
detect barge-in: if the user speaks over the reply, it calls the backend's
HTTP `/interrupt` route to cut generation and return to listening.

The VAD / TurnSense / accumulation / watchdog logic is ported from
Fun-ASR-deploy/funasr_wss_server.py (lines 190-300, 1164-1213) and reuses:
  - StreamingVAD            (MiniCPM-o-Demo/vad/vad.py)
  - TurnSenseModule         (Fun-ASR-deploy/turnsense_module.py)

Audio is forwarded to the backend as base64 float32 PCM (16kHz mono) — the
same wire format full_duplex already consumes.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from vad.vad import StreamingVAD, StreamingVadOptions

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class TurnSenseCfg:
    """TurnSense model + decision knobs."""

    enabled: bool = True
    model_path: str = ""                      # explicit onnx path; "" = auto-detect
    cmvn_path: str = ""                       # explicit am.mvn path
    audio_seconds: int = 8
    clip_mode: str = "tail"
    frontend_conf: Dict[str, Any] = field(default_factory=dict)
    incomplete_wait_ms: int = 900
    invalid_confidence_threshold: float = 0.9


@dataclass
class HalfDuplexConfig:
    """Half-duplex (VAD+TurnSense) turn-decision configuration."""

    vad_threshold: float = 0.8
    min_speech_duration_ms: int = 128
    min_silence_duration_ms: int = 800
    barge_in_enabled: bool = True
    barge_in_min_speech_ms: int = 300       # min user speech before barge-in fires
    max_accumulate_segments: int = 8        # cap on accumulated incomplete segments
    silence_trigger_ms: int = 100           # silence payload sent to trigger decode
    turnsense: TurnSenseCfg = field(default_factory=TurnSenseCfg)


def _default_turnsense_paths() -> Dict[str, str]:
    """Auto-locate TurnSense model under common project roots.

    支持的布局（任一命中即用，v1.0 优先因体积小、适合实时）：
      <code>/Fun-ASR-deploy/checkpoints/TurnSense/pretrained_models/
          model_fp32.onnx + am.mvn            （直接放根下）
          v1.0/model_fp32.onnx + am.mvn
          v1.1/model_fp32.onnx + am.mvn
      <code>/FullDuplexDemo/TurnSense/pretrained_models/
          model_fp32.onnx + am.mvn
    """
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    rels = [
        os.path.join("Fun-ASR-deploy", "checkpoints", "TurnSense", "pretrained_models"),
        os.path.join("FullDuplexDemo", "TurnSense", "pretrained_models"),
        os.path.join("..", "Fun-ASR-deploy", "checkpoints", "TurnSense", "pretrained_models"),
        os.path.join("..", "FullDuplexDemo", "TurnSense", "pretrained_models"),
    ]
    for rel in rels:
        d = os.path.join(root, rel)
        if not os.path.isdir(d):
            continue
        # 直接放根下
        if os.path.exists(os.path.join(d, "model_fp32.onnx")):
            return {
                "model_path": os.path.join(d, "model_fp32.onnx"),
                "cmvn_path": os.path.join(d, "am.mvn"),
            }
        # v1.0 / v1.1 子目录（优先 v1.0，体积小、实时延迟低）
        for ver in ("v1.0", "v1.1"):
            vd = os.path.join(d, ver)
            if os.path.exists(os.path.join(vd, "model_fp32.onnx")):
                return {
                    "model_path": os.path.join(vd, "model_fp32.onnx"),
                    "cmvn_path": os.path.join(vd, "am.mvn"),
                }
    return {"model_path": "", "cmvn_path": ""}


# ---------------------------------------------------------------------------
# TurnSense wrapper (lazy singleton)
# ---------------------------------------------------------------------------


class TurnSenseManager:
    """Lazily builds a shared TurnSenseModule + executor, like funasr_wss_server."""

    def __init__(self, cfg: TurnSenseCfg) -> None:
        self.cfg = cfg
        self.module = None
        self.executor: Optional[asyncio.ThreadPoolExecutor] = None

    def ensure(self) -> bool:
        if self.module is not None:
            return True
        if not self.cfg.enabled:
            return False
        paths = _default_turnsense_paths()
        model_path = self.cfg.model_path or paths.get("model_path", "")
        cmvn_path = self.cfg.cmvn_path or paths.get("cmvn_path", "")
        if not model_path or not os.path.exists(model_path):
            logger.warning("TurnSense model not found (%s); VAD-only trigger", model_path)
            return False
        try:
            import sys
            deploy_dir = os.path.join(root_for_deploy(), "Fun-ASR-deploy")
            if deploy_dir not in sys.path:
                sys.path.insert(0, deploy_dir)
            from turnsense_module import TurnSenseModule  # noqa: E402
            from turnsense_config import TurnSenseConfig  # noqa: E402

            frontend_conf = dict(self.cfg.frontend_conf)
            frontend_conf.setdefault("cmvn_file", cmvn_path)
            ts_config = TurnSenseConfig(
                enabled=True,
                model_path=model_path,
                cmvn_file=cmvn_path,
                audio_seconds=self.cfg.audio_seconds,
                clip_mode=self.cfg.clip_mode,
                frontend_conf=frontend_conf,
            )
            self.module = TurnSenseModule(ts_config)
            self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="turnsense")
            logger.info("TurnSense ready (model=%s)", model_path)
            return True
        except Exception as exc:
            logger.error("TurnSense init failed: %s", exc)
            return False

    async def predict(self, audio: np.ndarray) -> Optional[Dict[str, Any]]:
        if not self.ensure() or self.module is None or self.executor is None:
            return None
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self.executor, self.module.predict, audio)

    def shutdown(self) -> None:
        if self.executor is not None:
            self.executor.shutdown(wait=False)
            self.executor = None


def root_for_deploy() -> str:
    """Workspace root — parent of MiniCPM-o-Demo and Fun-ASR-deploy.

    __file__ = <root>/MiniCPM-o-Demo/runtime/half_duplex.py
    → dirname x3 = <root>
    """
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Decision result
# ---------------------------------------------------------------------------


class TurnDecision:
    TRIGGER = "trigger"      # user finished speaking → decode reply
    DEFERRED = "deferred"    # incomplete → accumulate, wait for more
    DISCARD = "discard"      # invalid/noise → do nothing


def _trim_trailing_silence(audio: np.ndarray) -> np.ndarray:
    """Port of funasr_wss_server._trim_trailing_silence."""
    if len(audio) == 0:
        return audio
    energy_threshold = 0.005
    frame_size = 512
    last_speech_idx = len(audio)
    for i in range(len(audio) - frame_size, -1, -frame_size):
        frame = audio[i:i + frame_size]
        rms = float(np.sqrt(np.mean(frame ** 2)))
        if rms > energy_threshold:
            last_speech_idx = min(i + frame_size * 3, len(audio))
            break
    return audio[:last_speech_idx].astype(np.float32)


async def _turnsense_decision(
    segment: np.ndarray,
    accumulated: List[np.ndarray],
    ts_mgr: TurnSenseManager,
    cfg: TurnSenseCfg,
) -> str:
    """Port of funasr_wss_server._handle_turnsense_decision → decision enum."""
    if not ts_mgr.ensure():
        return TurnDecision.TRIGGER  # TurnSense unavailable → VAD-only trigger

    audio = np.concatenate(accumulated + [segment]) if accumulated else segment
    trimmed = _trim_trailing_silence(audio)
    if len(trimmed) < 1600:
        return TurnDecision.DISCARD

    result = await ts_mgr.predict(trimmed)
    if result is None:
        return TurnDecision.TRIGGER

    label = result.get("label")
    probs = result.get("probabilities", {})
    logger.info("TurnSense: label=%s probs=%s duration=%.2fs", label, probs, len(audio) / 16000.0)

    if label == "complete":
        return TurnDecision.TRIGGER
    if label == "incomplete":
        return TurnDecision.DEFERRED
    if label == "invalid":
        invalid_conf = probs.get("invalid", 0.0)
        if invalid_conf < cfg.invalid_confidence_threshold:
            return TurnDecision.TRIGGER  # fallback
        return TurnDecision.DISCARD
    return TurnDecision.TRIGGER


def _b64_float32_to_np(b64: str) -> Optional[np.ndarray]:
    """Decode base64 float32 PCM (16kHz mono) to np.ndarray."""
    try:
        raw = base64.b64decode(b64)
        return np.frombuffer(raw, dtype=np.float32).copy()
    except Exception:
        return None


def _np_float32_to_b64(audio: np.ndarray) -> str:
    """Encode np.ndarray float32 to base64 float32 PCM."""
    return base64.b64encode(audio.astype(np.float32).tobytes()).decode()


# ---------------------------------------------------------------------------
# Half-duplex session state machine
# ---------------------------------------------------------------------------


class HalfDuplexState:
    LISTENING = "listening"
    GENERATING = "generating"


class HalfDuplexSession:
    """Owns VAD + TurnSense state for one frontend session.

    The caller (worker) supplies hooks:
      push(payload: dict)    — forward an input.append `input` payload to backend
      interrupt()            — call backend HTTP /interrupt (barge-in)
      send_event(dict)       — emit a control event back to the frontend
    """

    def __init__(
        self,
        *,
        config: Optional[HalfDuplexConfig] = None,
        push: Callable[[Dict[str, Any]], Any],
        interrupt: Callable[[], Any],
        send_event: Callable[[Dict[str, Any]], Any],
    ) -> None:
        self.cfg = config or HalfDuplexConfig()
        self.push = push
        self.interrupt = interrupt
        self.send_event = send_event

        self.state = HalfDuplexState.LISTENING
        self.vad = StreamingVAD(StreamingVadOptions(
            threshold=self.cfg.vad_threshold,
            min_speech_duration_ms=self.cfg.min_speech_duration_ms,
            min_silence_duration_ms=self.cfg.min_silence_duration_ms,
        ))
        self.ts_mgr = TurnSenseManager(self.cfg.turnsense)

        self.accumulated: List[np.ndarray] = []
        self.accum_gen = 0
        self.pending = False              # TurnSense incomplete, watchdog armed
        self._barge_in_detected = False
        self._vad_speech_started_at = 0.0

    # ------------------------------------------------------------------
    # Feed a frontend chunk (audio base64 float32 16kHz + optional frames)
    # ------------------------------------------------------------------
    async def feed(self, audio_b64: str, video_frames: Optional[List[str]] = None) -> None:
        audio_np = _b64_float32_to_np(audio_b64)
        if audio_np is None or len(audio_np) == 0:
            audio_np = np.zeros(1600, dtype=np.float32)  # 100ms silence guard

        # Always feed VAD (even while generating, to detect barge-in)
        was_speaking = self.vad.is_speaking
        segment = self.vad.feed(audio_np)
        if not was_speaking and self.vad.is_speaking:
            self._vad_speech_started_at = time.monotonic()
            logger.info("[HD] 🎤 检测到语音开始")

        if self.state == HalfDuplexState.GENERATING:
            await self._on_chunk_while_generating()
            return

        # LISTENING: stream into KV as force_listen prefill ("边听边看")
        await self.push({
            "audio": audio_b64,
            "video_frames": video_frames or [],
            "force_listen": True,
        })

        if segment is not None:
            seg_s = len(segment) / 16000.0
            logger.info("[HD] ⏸ VAD 检测到停顿, 语音段 %.1fs", seg_s)
            await self._on_speech_segment(segment)

    async def _on_chunk_while_generating(self) -> None:
        """While model is replying: detect barge-in (user starts speaking)."""
        if not self.cfg.barge_in_enabled or self._barge_in_detected:
            return
        if self.vad.is_speaking:
            elapsed_ms = (time.monotonic() - self._vad_speech_started_at) * 1000
            if elapsed_ms >= self.cfg.barge_in_min_speech_ms:
                self._barge_in_detected = True
                logger.info("Barge-in detected — interrupting generation")
                await self.send_event({"type": "turn.barge_in"})
                await self.interrupt()

    async def _on_speech_segment(self, segment: np.ndarray) -> None:
        """A complete VAD speech segment ended — run TurnSense decision."""
        decision = await _turnsense_decision(
            segment, self.accumulated, self.ts_mgr, self.cfg.turnsense,
        )

        if decision == TurnDecision.TRIGGER:
            logger.info("[HD] 🧠 TurnSense=complete → 触发回复")
            self.accumulated = []
            self.pending = False
            self.accum_gen += 1
            await self._trigger_reply()
        elif decision == TurnDecision.DEFERRED:
            logger.info("[HD] ⏳ TurnSense=incomplete → 等待用户继续 (watchdog %.0fms)",
                        self.cfg.turnsense.incomplete_wait_ms)
            self.accumulated.append(segment)
            if len(self.accumulated) > self.cfg.max_accumulate_segments:
                self.accumulated = self.accumulated[-self.cfg.max_accumulate_segments:]
            self.accum_gen += 1
            gen = self.accum_gen
            self.pending = True
            await self.send_event({"type": "turn.turnsense", "label": "incomplete"})
            asyncio.create_task(self._watchdog(gen))
        else:  # DISCARD
            logger.info("[HD] 🚫 TurnSense=invalid → 丢弃(噪音/过短)")
            await self.send_event({"type": "turn.turnsense", "label": "invalid"})

    async def _trigger_reply(self) -> None:
        """User finished — send a decode trigger (chunk WITHOUT force_listen)."""
        self.state = HalfDuplexState.GENERATING
        self._barge_in_detected = False
        logger.info("[HD] ▶ 触发模型回复")
        await self.send_event({"type": "turn.turnsense", "label": "complete"})

        # Trigger decode: short silence chunk, no force_listen → backend decodes.
        silence = np.zeros(int(16000 * self.cfg.silence_trigger_ms / 1000), dtype=np.float32)
        await self.push({
            "audio": _np_float32_to_b64(silence),
            "video_frames": [],
            # force_listen intentionally omitted → decode
        })

    async def _watchdog(self, gen: int) -> None:
        """Port of funasr_wss_server._turnsense_watchdog."""
        try:
            wait_s = self.cfg.turnsense.incomplete_wait_ms / 1000.0
            await asyncio.sleep(wait_s)
            if not self.pending or self.accum_gen != gen:
                return  # superseded by a newer segment
            self.pending = False
            self.accum_gen += 1
            logger.info("TurnSense incomplete timeout — forcing reply")
            self.accumulated = []
            await self._trigger_reply()
        except asyncio.CancelledError:
            pass

    # ------------------------------------------------------------------
    # Backend event feed (from worker's backend_to_client loop)
    # ------------------------------------------------------------------
    async def on_backend_event(self, event: Dict[str, Any]) -> bool:
        """Returns True if the event should be forwarded to the frontend."""
        etype = str(event.get("type") or "")
        if etype == "response.output.delta":
            kind = str(event.get("kind") or "")
            if kind == "listen":
                logger.info("[HD] 👂 模型 → Listen")
                self._reset_to_listening()
            elif kind == "text":
                txt = str(event.get("text") or "")
                if txt:
                    logger.info("[HD] 💬 模型回复: %s", txt.strip())
            elif kind == "audio":
                logger.info("[HD] 🔊 模型 → 说话 (audio chunk)")
        elif etype == "response.done":
            full = str(event.get("text") or "")
            if full:
                logger.info("[HD] ✅ 回复完成: %s", full.strip())
            else:
                logger.info("[HD] ✅ 回复完成 (无文本)")
            self._reset_to_listening()
        return True

    def _reset_to_listening(self) -> None:
        if self.state == HalfDuplexState.GENERATING:
            self.state = HalfDuplexState.LISTENING
            self.vad.reset()  # drop VAD state from the reply period
            self._barge_in_detected = False
            logger.info("Back to listening")

    def shutdown(self) -> None:
        self.ts_mgr.shutdown()
