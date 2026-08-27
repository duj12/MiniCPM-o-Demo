"""
TurnSense 配置模块 — 用于 Fun-ASR-deploy
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Any

# 模型统一放项目根 checkpoints/（runtime/turnsense/turnsense_config.py 上溯三级 = 项目根）
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class TurnSenseConfig:
    """TurnSense 模型 + 判决参数配置"""
    # 主开关
    enabled: bool = False

    # 模型文件路径（默认 v1.0 8bit int8；half_duplex 探测后可能覆盖为具体版本）
    model_path: str = os.path.join(
        ROOT_DIR, "checkpoints", "TurnSense", "pretrained_models", "v1.0", "model_int8.onnx"
    )
    cmvn_file: str = os.path.join(
        ROOT_DIR, "checkpoints", "TurnSense", "pretrained_models", "v1.0", "am.mvn"
    )

    # 标签
    labels: tuple = ("complete", "incomplete", "invalid")

    # 音频截取参数
    audio_seconds: int = 8
    sampling_rate: int = 16000
    clip_mode: str = "tail"

    # 前端特征配置
    frontend_conf: Dict[str, Any] = field(default_factory=lambda: {
        "cmvn_file": os.path.join(
            ROOT_DIR, "checkpoints", "TurnSense", "pretrained_models", "v1.0", "am.mvn"
        ),
        "fs": 16000,
        "window": "hamming",
        "n_mels": 80,
        "frame_length": 25,
        "frame_shift": 10,
        "lfr_m": 7,
        "lfr_n": 6,
        "dither": 0.0,
    })

    # ===== 判决参数 =====
    # incomplete 等待超时（毫秒）
    incomplete_wait_ms: int = 1000

    # invalid 丢弃置信度阈值
    invalid_confidence_threshold: float = 0.7

    # 语音时长兜底（秒）— 超过此时长的语音即使 invalid 也发送
    min_speech_duration_for_fallback: float = 6.0
