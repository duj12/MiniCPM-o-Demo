"""Orchestrator 配置。

所有外部服务地址走配置，便于在不同环境切换（开发机 / 106 / 生产）。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

# ---------------- 外部服务默认地址（阶段 0 实测确认） ---------------- #

# AEC：speech_frontend 的 /ws/asr_frontend
DEFAULT_AEC_URL = "ws://192.168.88.253:30255/ws/asr_frontend"
# ASR：Fun-ASR 协议
DEFAULT_ASR_URL = "ws://192.168.88.101:31366"
# TTS：gRPC
DEFAULT_TTS_HOST = "192.168.88.253"
DEFAULT_TTS_PORT = 31058
# OmniLLM：MiniCPM-o-Demo gateway
DEFAULT_OMNI_URL = "wss://127.0.0.1:8006/v1/realtime?mode=video"


@dataclass
class Settings:
    """运行配置。环境变量优先，其次是这里的默认值。"""

    # 监听
    host: str = "0.0.0.0"
    port: int = 8100

    # 外部服务
    aec_url: str = field(default_factory=lambda: os.environ.get(
        "ORCH_AEC_URL", DEFAULT_AEC_URL))
    asr_url: str = field(default_factory=lambda: os.environ.get(
        "ORCH_ASR_URL", DEFAULT_ASR_URL))
    tts_host: str = field(default_factory=lambda: os.environ.get(
        "ORCH_TTS_HOST", DEFAULT_TTS_HOST))
    tts_port: int = field(default_factory=lambda: int(os.environ.get(
        "ORCH_TTS_PORT", str(DEFAULT_TTS_PORT))))
    omni_url: str = field(default_factory=lambda: os.environ.get(
        "ORCH_OMNI_URL", DEFAULT_OMNI_URL))

    # 功能开关（便于分阶段验证/降级）
    enable_aec: bool = True
    enable_asr: bool = True
    enable_omni: bool = True
    enable_tts: bool = True
    enable_face: bool = False        # 阶段 4 后开启
    verify_ssl: bool = False         # gateway 用自签证书

    # ---- 回声消除模式（可被前端 session.start 覆盖）----
    #   "browser" —— 浏览器原生 AEC（getUserMedia 的 echoCancellation）。
    #                在设备侧工作，**不受网络延迟影响**；不连云端 AEC，
    #                省一次云往返。
    #   "service" —— 云端算法 AEC。需要参考轨按声学延迟 D 预对齐
    #                （本模块自适应测量）；实测抑制 9~11dB，弱于浏览器
    #                原生，但可验证算法链路。
    #   "off"     —— 都不做。
    # 默认 service：音频端点用云端算法 AEC（需延迟校准，见 /v1/calibrate）。
    # 若该服务不可用或延迟过大，前端可切到 browser。
    # ⚠️ 默认 **browser**（浏览器原生 AEC）：开箱可用，不依赖离线实测的 D。
    #    算法服务 AEC 需要外放 + 每台设备离线测一次 D（见 run_orch.sh），
    #    当作默认值会让"刚跑起来"的会话回声消不掉 —— 实测 D=250 这种占位值
    #    抑制只有 0.3dB，等于不工作。
    aec_mode: str = field(default_factory=lambda: os.environ.get(
        "ORCH_AEC_MODE", "browser"))

    # 声学延迟 D（毫秒）—— **每台设备一个固定常量**，离线测一次。
    #
    # 为什么是常量而不是运行时自适应：AEC 的容忍窗只有 ±5ms，而自适应
    # 估计器在真机（iPhone）上收敛不了（峰比恒 ≈1.0）；更糟的是它一旦被
    # 单个噪声峰钉死在错值上就**永久失效** —— 表现为"长回复好好地说着，
    # 突然就无法打断、开始识别自己说的话了"。
    #
    # 现在 D 的语义变干净了：参考轨的落位时刻来自浏览器的 armed 承诺
    # （精确），所以 D **只剩**「扬声器→麦克风的物理延迟 + 设备音频 I/O
    # 缓冲」这一小块 —— 网络往返、浏览器主线程抖动、mic 在途积压全部不再
    # 计入。这是一个真正的设备常量。
    #
    # 测法：开着 ORCH_DUMP_AUDIO 跑一轮真实会话，然后
    #     python -m orchestrator.tests.measure_delay \
    #         --mic <前缀>-<sid>-mic.wav --raw <前缀>-<sid>-raw.wav \
    #         --verify-url ws://192.168.88.253:30255/ws/asr_frontend
    # 结果写进 ORCH_AEC_DEFAULT_DELAY_MS（或按设备存进 DelayStore）。
    #
    # ⚠️ 250 是**未测量时的占位值**，不是可用值 —— 它会让算法 AEC 基本
    #    不生效（实测抑制 0.3dB）。请以实测值为准。
    aec_default_delay_ms: float = field(default_factory=lambda: float(
        os.environ.get("ORCH_AEC_DEFAULT_DELAY_MS", "250")))
    # 延迟记录持久化路径
    delay_store_path: Optional[str] = field(default_factory=lambda:
        os.environ.get("ORCH_DELAY_STORE"))

    # downstream 桩模式：asr | omni | echo | none
    downstream_mode: str = "omni"

    # TTS 参数
    tts_type: str = "mltts"
    tts_speaker_id: str = "17"
    # 流式合成：LLM 文本 delta 一到就喂 TTS，音频一出就播（首声快很多）。
    # 关掉则退回「整条回复一次合成」。用 ORCH_TTS_STREAMING=0 回退。
    tts_streaming: bool = field(default_factory=lambda: os.environ.get(
        "ORCH_TTS_STREAMING", "1") not in ("0", "false", "False", ""))

    # OmniLLM
    omni_system_prompt: str = "你是一个实时视频对话助手。请一边观看用户传来的实时画面，一边倾听并用自然口语即时回复。"
    # 回复触发方式：
    #   "asr"       —— **按 ASR 最终文本触发**（推荐）。OmniLLM 持续以
    #                  force_listen 累积视听上下文（"边听边看"），收到
    #                  ASR 的 2pass-offline 时由我方补触发。回复是"看到
    #                  截止此刻的画面 + 听到整句语音"后生成的。
    #   "turnsense" —— 服务端 VAD+TurnSense 判决（原行为）。
    omni_turn_trigger: str = field(default_factory=lambda: os.environ.get(
        "ORCH_OMNI_TRIGGER", "asr"))

    # 人脸（阶段 4）
    face_lib_path: Optional[str] = field(default_factory=lambda: os.environ.get(
        "ORCH_FACE_SO"))
    face_model_dir: Optional[str] = field(default_factory=lambda: os.environ.get(
        "ORCH_FACE_MODELS"))
    face_db_path: Optional[str] = field(default_factory=lambda: os.environ.get(
        "ORCH_FACE_DB"))

    # 调参
    tick_interval_s: float = 0.05
    playback_delay_ms: int = 200
    # 会话收尾超时：等 ASR 最终结果 + 各组件 flush 的总上限。
    # 实测 ASR 从 is_speaking=false 到 is_final 约 1~2s，留足余量。
    drain_timeout_s: float = 20.0

    @classmethod
    def from_env(cls) -> "Settings":
        s = cls()
        if os.environ.get("ORCH_DOWNSTREAM_MODE"):
            s.downstream_mode = os.environ["ORCH_DOWNSTREAM_MODE"]
        if os.environ.get("ORCH_ENABLE_FACE"):
            s.enable_face = os.environ["ORCH_ENABLE_FACE"] not in ("0", "false", "")
        return s
