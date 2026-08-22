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
class VadCfg:
    """VAD 配置：支持 Silero（默认）或 FunASR fsmn-vad。

    vad_model: "silero" | "fsmn"（fsmn 为 FunASR fsmn-vad，默认）
    fsmn_model_dir: fsmn-vad 模型目录（含 config.yaml 等）
    vad_tail_sil: 尾静音（ms），fsmn 的 max_end_silence_time，默认 600
    vad_max_len: 最大段长（ms），fsmn 的 max_single_segment_time，默认 60000
    vad_threshold: Silero 阈值（仅 silero 用）
    min_speech_duration_ms / min_silence_duration_ms: Silero 参数
    """

    vad_model: str = "fsmn"                 # "silero" | "fsmn"
    fsmn_model_dir: str = ""                # FunASR fsmn-vad 模型目录
    vad_tail_sil: int = 600                 # 尾静音 ms -> max_end_silence_time
    vad_max_len: int = 60000                # 最大段长 ms -> max_single_segment_time
    vad_chunk_size: int = 1000              # fsmn 内部处理窗口 ms（1s 可流式分句）
    # Silero 参数（vad_model=silero 时用）
    vad_threshold: float = 0.8
    min_speech_duration_ms: int = 128
    min_silence_duration_ms: int = 800


@dataclass
class HalfDuplexConfig:
    """Half-duplex (VAD+TurnSense) turn-decision configuration."""

    vad: VadCfg = field(default_factory=VadCfg)   # VAD 选择与参数
    barge_in_enabled: bool = True
    barge_in_min_speech_ms: int = 300       # min user speech before barge-in fires
    max_accumulate_segments: int = 8        # cap on accumulated incomplete segments
    silence_trigger_ms: int = 100           # silence payload sent to trigger decode
    turnsense: TurnSenseCfg = field(default_factory=TurnSenseCfg)


class FSMNVADManager:
    """惰性加载 fsmn-vad ONNX 模型（单例，跨 session 共享）。

    用 onnxruntime（容器已有），无需 funasr 包。模型目录需含：
      model.onnx + config.yaml + am.mvn
    参考 FunASR onnx 流式实现：Fsmn_vad_online。
    """

    _instance = None

    @classmethod
    def get(cls, model_dir: str) -> "FSMNVADManager":
        if cls._instance is None or cls._instance.model_dir != model_dir:
            cls._instance = cls(model_dir)
        return cls._instance

    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.model = None
        self._load()

    def _load(self):
        if not self.model_dir:
            # 自动探测：vad-runtime/model、FunASR onnx、或本包内置 model 目录
            root = root_for_deploy()
            here = os.path.dirname(os.path.abspath(__file__))
            cands = [
                os.path.join(root, "vad-runtime", "model"),
                os.path.join(here, "fsmn_vad_onnx", "model"),   # 容器内/本包内置
                os.path.join(root, "FunASR", "runtime", "python", "onnxruntime", "models", "vad"),
            ]
            found = next((c for c in cands if os.path.exists(os.path.join(c, "model.onnx"))), None)
            if found is None:
                raise RuntimeError(f"fsmn-vad onnx 模型未找到，请设置 fsmn_model_dir 或放模型到 {cands[0]}")
            self.model_dir = found

        # 导入 FunASR onnx 流式 VAD（本仓库 runtime/fsmn_vad_onnx/）
        try:
            from runtime.fsmn_vad_onnx.vad_bin import Fsmn_vad_online
        except ImportError:
            import sys
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from fsmn_vad_onnx.vad_bin import Fsmn_vad_online
        self._Fsmn_vad_online = Fsmn_vad_online
        self.max_end_sil = None  # 在 make_vad 时按参数创建
        logger.info("fsmn-vad onnx model dir: %s", self.model_dir)

    def make_vad(self, max_end_silence_time: int, max_single_segment_time: int,
                 chunk_size: int = 1000) -> "FSMNVAD":
        vad = self._Fsmn_vad_online(
            model_dir=self.model_dir,
            max_end_sil=max_end_silence_time,
            intra_op_num_threads=1,
        )
        return FSMNVAD(vad, max_single_segment_time)


class FSMNVAD:
    """fsmn-vad ONNX 流式适配器，接口对齐 StreamingVAD。

    基于 FunASR onnx 的 Fsmn_vad_online（runtime/fsmn_vad_onnx/vad_bin.py），
    用 onnxruntime 逐块推理。feed 返回完整语音段（VAD 检测到段结束时）。

    feed(audio_np) -> Optional[np.ndarray]
    is_speaking 属性
    reset()
    """

    SAMPLE_RATE = 16000

    def __init__(self, vad_online, max_single_segment_time: int = 60000):
        self.vad = vad_online                      # Fsmn_vad_online 实例
        self.max_single_segment_time = max_single_segment_time
        self.param_dict: Dict[str, Any] = {}       # 流式状态（in_cache/frontend/vad_scorer）
        self._speech_buffer: List[np.ndarray] = []
        self._abs_pos_ms = 0                       # 已处理的绝对位置 (ms)
        self._seg_start_ms = -1                    # 当前语音段开始位置 (ms)
        self._is_speaking = False

    @property
    def is_speaking(self) -> bool:
        return self._is_speaking

    def reset(self) -> None:
        self.param_dict = {}
        self._speech_buffer = []
        self._abs_pos_ms = 0
        self._seg_start_ms = -1
        self._is_speaking = False

    def feed(self, audio_np: np.ndarray) -> Optional[np.ndarray]:
        """喂入 float32 16kHz 音频，VAD 段结束时返回完整语音段。"""
        if len(audio_np) == 0:
            return None

        # 调 FunASR onnx 流式 VAD（param_dict 维护 frontend/in_cache/vad_scorer 状态）
        self.param_dict["is_final"] = False
        segments = self.vad(audio_np, param_dict=self.param_dict)
        # segments: [[start_ms, end_ms], ...]，start/end 为 -1 表示只有边界

        chunk_dur_ms = int(len(audio_np) / self.SAMPLE_RATE * 1000)

        if segments and isinstance(segments, list):
            # segments 形如 [[[start,-1]], ...]：外层是 batch，取 batch0 的段列表
            seg_list = segments[0] if segments and isinstance(segments[0], list) else segments
            if isinstance(seg_list, list) and seg_list and isinstance(seg_list[0], (list, tuple)):
                for seg in seg_list:
                    if len(seg) < 2:
                        continue
                    start_ms, end_ms = seg[0], seg[-1]
                    if start_ms != -1 and self._seg_start_ms < 0:
                        # 语音开始：开始累积
                        self._seg_start_ms = start_ms
                        self._speech_buffer = []
                        self._is_speaking = True
                    if end_ms != -1 and self._seg_start_ms >= 0:
                        # 语音结束：返回累积的语音段
                        segment = np.concatenate(self._speech_buffer) if self._speech_buffer else audio_np
                        self._is_speaking = False
                        self._seg_start_ms = -1
                        self._speech_buffer = []
                        return segment

        # 语音进行中：累积当前块
        if self._is_speaking:
            self._speech_buffer.append(audio_np)

        self._abs_pos_ms += chunk_dur_ms
        return None

    def flush(self) -> Optional[np.ndarray]:
        """强制结束当前语音段（is_final=True 触发 VAD 输出最后段）。"""
        if self._is_speaking:
            self.param_dict["is_final"] = True
            try:
                self.vad(np.zeros(160, dtype=np.float32), param_dict=self.param_dict)
            except Exception:
                pass
            segment = np.concatenate(self._speech_buffer) if self._speech_buffer else None
            self._is_speaking = False
            self._seg_start_ms = -1
            self._speech_buffer = []
            return segment
        return None


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
        active_model: str = "minicpm",
    ) -> None:
        self.cfg = config or HalfDuplexConfig()
        self.push = push
        self.interrupt = interrupt
        self.send_event = send_event
        # 后端模型类型：minicpm（free-duplex 采样，需要 1s 静音驱动循环）
        # 或 qwen3omni（turn-based，触发 chunk 即完整回复，保持一次性触发）。
        self.active_model = active_model

        self.state = HalfDuplexState.LISTENING
        # 选择 VAD：fsmn（FunASR，默认）或 silero（StreamingVAD）
        vad_cfg = self.cfg.vad
        if vad_cfg.vad_model == "fsmn":
            mgr = FSMNVADManager.get(vad_cfg.fsmn_model_dir)
            self.vad = mgr.make_vad(
                max_end_silence_time=vad_cfg.vad_tail_sil,        # 尾静音 ms
                max_single_segment_time=vad_cfg.vad_max_len,      # 最大段长 ms
                chunk_size=vad_cfg.vad_chunk_size,                # 处理窗口 ms
            )
            logger.info("[HD] 使用 fsmn-vad (tail_sil=%dms, max_len=%dms, chunk=%dms)",
                        vad_cfg.vad_tail_sil, vad_cfg.vad_max_len, vad_cfg.vad_chunk_size)
        else:
            self.vad = StreamingVAD(StreamingVadOptions(
                threshold=vad_cfg.vad_threshold,
                min_speech_duration_ms=vad_cfg.min_speech_duration_ms,
                min_silence_duration_ms=vad_cfg.min_silence_duration_ms,
            ))
        self.ts_mgr = TurnSenseManager(self.cfg.turnsense)

        self.accumulated: List[np.ndarray] = []
        self.accum_gen = 0
        self.pending = False              # TurnSense incomplete, watchdog armed
        self._barge_in_detected = False
        self._vad_speech_started_at = 0.0

        # 回复驱动状态（仅 MiniCPM free-duplex 使用）
        self._reply_ended = False         # 模型已输出 <|listen|>，回复结束
        self._last_reply_activity = 0.0   # 最近一次 text/audio delta 时间
        self._drive_task = None           # 1s 静音驱动任务
        self._reply_stall_task = None     # 回复停滞看门狗

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
        """User finished — trigger the model to reply.

        - Qwen3-Omni（turn-based）：一个触发 chunk 即产生完整回复，保持一次性触发。
        - MiniCPM-o（free-duplex）：首 chunk 带 force_reply（C++ 一次性，保证模型开口），
          然后 1s 静音驱动循环，让模型基于累积的音频 KV 自然多 chunk 回复，直到它
          自己输出 <|listen|> 结束 turn。
        """
        self.state = HalfDuplexState.GENERATING
        self._barge_in_detected = False
        self._reply_ended = False
        self._last_reply_activity = time.monotonic()
        logger.info("[HD] ▶ 触发模型回复")
        await self.send_event({"type": "turn.turnsense", "label": "complete"})

        if self.active_model != "minicpm":
            # Qwen3-Omni: turn-based，触发 chunk 即完整回复，无静音循环。
            silence = np.zeros(int(16000 * self.cfg.silence_trigger_ms / 1000), dtype=np.float32)
            await self.push({
                "audio": _np_float32_to_b64(silence),
                "video_frames": [],
                # force_listen intentionally omitted → decode
                "force_reply": True,
            })
            return

        # MiniCPM-o: 首 chunk 带 force_reply（C++ 一次性 → 首 token 非 listen，确保开口）。
        silence = np.zeros(16000, dtype=np.float32)  # 1 秒
        await self.push({
            "audio": _np_float32_to_b64(silence),
            "video_frames": [],
            "force_reply": True,
        })
        # 启动 1s 静音驱动循环 + 停滞看门狗。
        self._drive_task = asyncio.create_task(self._drive_reply_loop())
        self._reply_stall_task = asyncio.create_task(self._reply_stall_watchdog())

    async def _drive_reply_loop(self) -> None:
        """每 1s 发一个静音 chunk（无 force_reply / force_listen），自然驱动模型继续说话。

        模型每 chunk 输出 <|chunk_eos|>（本轮 chunk 结束），下一个静音 chunk 触发下一次
        decode → 下一个自然 chunk。直到模型输出 <|listen|>（on_backend_event 置
        _reply_ended）或收到 barge-in / 超时。
        """
        try:
            while self.state == HalfDuplexState.GENERATING and not self._reply_ended:
                await asyncio.sleep(1.0)
                if self.state != HalfDuplexState.GENERATING or self._reply_ended:
                    break
                try:
                    await self.push({
                        "audio": _np_float32_to_b64(np.zeros(16000, dtype=np.float32)),
                        "video_frames": [],
                        # force_listen/force_reply 均不设 → 自然 decode
                    })
                except Exception as exc:
                    # session 已关闭 / 后端不可用 → 停止驱动，避免异常泄漏
                    logger.info("[HD] 静音驱动停止 (push failed: %s)", exc)
                    break
        except asyncio.CancelledError:
            pass

    async def _reply_stall_watchdog(self) -> None:
        """若模型长时间无 text/audio delta（回复停滞），强制回到 listening，防止卡死。"""
        stall_s = self.cfg.turnsense.incomplete_wait_ms / 1000.0 * 4  # 默认 3.6s，够多 chunk
        stall_s = max(stall_s, 6.0)  # 至少 6s
        try:
            while not self._reply_ended and self.state == HalfDuplexState.GENERATING:
                await asyncio.sleep(stall_s)
                if self._reply_ended or self.state != HalfDuplexState.GENERATING:
                    break
                if time.monotonic() - self._last_reply_activity >= stall_s:
                    logger.warning("[HD] reply stall timeout — forcing back to listening")
                    self._reply_ended = True
                    self._reset_to_listening()
        except asyncio.CancelledError:
            pass

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
                logger.info("[HD] 👂 模型 → Listen（本轮回复结束）")
                # MiniCPM: 只有 <|listen|> 表示模型真正让出话轮 → 复位。
                # Qwen3-Omni 走 response.done 复位（下面）。
                self._reply_ended = True
                self._reset_to_listening()
            elif kind == "text":
                txt = str(event.get("text") or "")
                if txt:
                    self._last_reply_activity = time.monotonic()
                    logger.info("[HD] 💬 模型回复: %s", txt.strip())
            elif kind == "audio":
                self._last_reply_activity = time.monotonic()
                logger.info("[HD] 🔊 模型 → 说话 (audio chunk)")
        elif etype == "response.done":
            full = str(event.get("text") or "")
            if full:
                logger.info("[HD] ✅ 回复完成: %s", full.strip())
            else:
                logger.info("[HD] ✅ 回复完成 (无文本)")
            # MiniCPM free-duplex: 每个静音 chunk 都产生一个 response.done（该 chunk 的
            # 文本），这是"chunk 边界"而非"turn 结束"。只有 listen 才复位。
            # Qwen3-Omni turn-based: response.done 即完整回复结束。
            if self.active_model != "minicpm":
                self._reply_ended = True
                self._reset_to_listening()
        return True

    def _reset_to_listening(self) -> None:
        if self.state == HalfDuplexState.GENERATING:
            self.state = HalfDuplexState.LISTENING
            self.vad.reset()  # drop VAD state from the reply period
            self._barge_in_detected = False
            # 清理回复驱动任务，防止旧循环泄漏到下次触发。
            if self._drive_task is not None and not self._drive_task.done():
                self._drive_task.cancel()
            if self._reply_stall_task is not None and not self._reply_stall_task.done():
                self._reply_stall_task.cancel()
            self._drive_task = None
            self._reply_stall_task = None
            logger.info("Back to listening")

    def shutdown(self) -> None:
        self.ts_mgr.shutdown()
