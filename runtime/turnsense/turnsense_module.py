"""
TurnSense 模块 — 用于 Fun-ASR-deploy 集成

功能: 封装 TurnSense 模型，判断用户语音是否表达完整
三分类: complete / incomplete / invalid

调用方式:
    ts = TurnSenseModule(config)
    result = ts.predict(audio_array)
"""

import os
import sys
import time
import numpy as np
import logging

# ===== 自动查找 cuDNN 9 库路径 =====
# onnxruntime-gpu(>=1.19) 需要 cuDNN 9 在 LD_LIBRARY_PATH 中。
# 搜索策略：
#   1) pip 安装的 nvidia/cudnn/lib（conda site-packages）
#   2) /proc/self/maps 中 torch 已加载的 cuDNN 路径
#   3) 系统 CUDA 安装目录 /usr/local/cuda/lib64/
#   4) ldconfig 注册的路径
# 找到一个可用即止。

def _add_cudnn_to_path():
    _candidate_dirs = []

    # 1) pip 安装的 nvidia/cudnn/lib
    _candidate_dirs.append(os.path.join(
        os.path.dirname(os.path.dirname(sys.executable)),
        "lib",
        f"python{sys.version_info.major}.{sys.version_info.minor}",
        "site-packages", "nvidia", "cudnn", "lib",
    ))

    # 2) 从 /proc/self/maps 看 torch 已加载的 cuDNN 位置
    try:
        with open("/proc/self/maps") as _f:
            for _line in _f:
                if "libcudnn" in _line:
                    _parts = _line.strip().split()
                    if len(_parts) >= 6:
                        _d = os.path.dirname(_parts[-1])
                        if _d not in _candidate_dirs:
                            _candidate_dirs.append(_d)
    except Exception:
        pass

    # 3) 系统 CUDA 目录
    for _cuda_home in ("/usr/local/cuda", "/usr/local/cuda-12",
                       os.environ.get("CUDA_HOME", "")):
        if _cuda_home:
            for _sub in ("lib64", "lib"):
                _p = os.path.join(_cuda_home, _sub)
                if os.path.isdir(_p) and _p not in _candidate_dirs:
                    _candidate_dirs.append(_p)

    # 4) ldconfig -p 注册的路径
    try:
        import subprocess
        _out = subprocess.check_output(["ldconfig", "-p"], text=True, stderr=subprocess.DEVNULL)
        for _line in _out.splitlines():
            if "libcudnn" in _line and "=>" in _line:
                _p = _line.split("=>", 1)[1].strip()
                _d = os.path.dirname(_p)
                if _d not in _candidate_dirs:
                    _candidate_dirs.append(_d)
    except Exception:
        pass

    # 去重后加入 LD_LIBRARY_PATH
    _existing = os.environ.get("LD_LIBRARY_PATH", "").split(":")
    _added = []
    for _d in _candidate_dirs:
        if _d and os.path.isdir(_d) and _d not in _existing:
            _existing.insert(0, _d)
            _added.append(_d)
    if _added:
        os.environ["LD_LIBRARY_PATH"] = ":".join(_existing)
        logging.getLogger(__name__).info(f"Added cuDNN lib to LD_LIBRARY_PATH: {_added}")

_add_cudnn_to_path()

import onnxruntime as ort

from .turnsense_config import TurnSenseConfig
from .audio_frontend import AudioFrontend

logger = logging.getLogger(__name__)


class TurnSenseModule:
    """
    TurnSense 话语完整性判断模块

    使用 ONNX 模型对用户语音进行三分类：
    complete（完整）/ incomplete（不完整）/ invalid（无效）
    """

    def __init__(self, ts_config: TurnSenseConfig):
        """
        初始化 TurnSense 模块

        Args:
            ts_config: TurnSense 配置对象
        """
        self.config = ts_config
        self.labels = list(ts_config.labels)
        self.enabled = ts_config.enabled

        if not self.enabled:
            logger.info("TurnSense is DISABLED")
            return

        # 初始化音频前端（特征提取器）
        logger.info("Initializing TurnSense AudioFrontend...")
        self.frontend = AudioFrontend(**ts_config.frontend_conf)

        # 加载 ONNX 模型
        logger.info(f"Loading TurnSense model from: {ts_config.model_path}")
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.inter_op_num_threads = 1
        sess_options.intra_op_num_threads = 4

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.session = ort.InferenceSession(
            ts_config.model_path,
            sess_options=sess_options,
            providers=providers
        )
        logger.info(f"TurnSense model loaded successfully, provider={self.session.get_providers()[0]}")

    def predict(self, audio: np.ndarray) -> dict:
        """
        对音频进行话语完整性判断

        Args:
            audio: numpy 数组，float32，采样率 16kHz，shape (N,)
                   值域 [-1, 1]

        Returns:
            dict: {
                "label": str,  # "complete" / "incomplete" / "invalid"
                "prediction_id": int,
                "probabilities": {label: float, ...}
            }
        """
        _t0 = time.time()
        if not self.enabled:
            # 未启用时，默认返回 complete
            return {
                "label": "complete",
                "prediction_id": 0,
                "probabilities": {"complete": 1.0, "incomplete": 0.0, "invalid": 0.0}
            }

        # 截取音频（最后 N 秒）
        audio = self._truncate_audio(audio)

        # 提取特征
        feats, feat_len = self.frontend.extract_features(audio)
        feats = np.asarray(feats, dtype=np.float32)[np.newaxis, ...]  # (1, T, D)
        feat_len = np.asarray([int(feat_len)], dtype=np.int64)  # (1,)

        # ONNX 推理
        outputs = self.session.run(None, {
            "feats": feats,
            "feat_lengths": feat_len,
        })
        logits = outputs[0]  # shape: (1, num_classes)

        # 后处理
        probs = self._softmax(logits[0])
        pred_id = int(np.argmax(probs))

        result = {
            "label": self.labels[pred_id],
            "prediction_id": pred_id,
            "probabilities": {
                label: float(probs[i]) for i, label in enumerate(self.labels)
            }
        }

        _elapsed = time.time() - _t0
        logger.info(f"TurnSense result: {result['label']} "
                    f"(probs: {result['probabilities']}) "
                    f"elapsed={_elapsed:.3f}s")
        return result

    def _truncate_audio(self, audio: np.ndarray) -> np.ndarray:
        """截取音频到指定长度（从尾部截取）"""
        max_samples = self.config.audio_seconds * self.config.sampling_rate
        if len(audio) <= max_samples:
            return audio
        if self.config.clip_mode == "tail":
            return audio[-max_samples:]
        return audio[:max_samples]

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        """计算 softmax"""
        x = x - np.max(x)
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x)