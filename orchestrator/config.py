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
    #    算法服务 AEC 需要外放 + 每台设备离线测一次 D
    #    （见 orchestrator/run_orch.sh），
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

    # ---- InteractionCore + Agent 决策链路 ----
    # 打开后**回复不再由 OmniLLM 生成**：IC 做 SOP 决策 → Agent 生成 →
    # Agent 回调 orchestrator 的 /v1/speak 播报。OmniLLM 只产出音视频描述。
    #
    # ⚠️ **默认开**。两个例外：
    #   · `downstream_mode == "echo"` 时保持纯桩（测试用它起"永不 Speak"的服务）
    #   · IC 连不上 → 明确告警并**降级回 OmniLLM 回复**
    #     （否则会变成"能识别、永远不回复"，与之前 OmniLLM 断线那个 bug
    #      同一病理，极难排查）
    interaction_enabled: bool = field(default_factory=lambda: os.environ.get(
        "ORCH_INTERACTION", "1") not in ("0", "false", "False", ""))
    #: InteractionCore 的 gRPC 地址。**由编排服务去连**（不是客户端连）——
    #: 这样 web 和 python 客户端自动都具备。
    ic_grpc: str = field(default_factory=lambda: os.environ.get(
        "ORCH_IC_GRPC", "localhost:50051"))
    #: **告诉 Agent 回连 IC 用哪个地址** —— 与 `ic_grpc` 是**两件事**。
    #:
    #: `ic_grpc`   是**编排服务自己去连** IC 的地址。
    #: `ic_advertise` 是**Agent 回连** IC 的地址（Agent 在 192.168.89.102，
    #: 它收到 IC 的 Action 后要 `POST {target}/interaction/on_*`）。
    #:
    #: ⚠️ 同机部署时两者可以一样（都是 localhost）。但**跨主机时不填这个就错**
    #:    —— 实测踩过：105 把 `127.0.0.1:50051` 告诉 Agent，而这个地址从
    #:    Agent 视角看是**它自己**，于是
    #:      · 105 会话期间：Agent 把 Action 派到错误的地方
    #:      · 105 会话结束归还后：留下的还是这个错地址 → **106 的会话也跟着错**
    #:    这就是两台机器互相影响的真实通道之一。
    #: 空 = 退回 `ic_grpc`（同机部署的常见情形，行为与旧版一致）。
    ic_advertise: str = field(default_factory=lambda: os.environ.get(
        "ORCH_IC_ADVERTISE", ""))
    #: 会话结束时把 Agent 的 IC 目标**归还到哪** —— 应当是**大家共用的**
    #: 那个 IC，不是本机自己的。
    #:
    #: ⚠️ 多机部署**必须**配它。实测踩过：105 归还成自己的地址
    #:    （`192.168.89.105:50051`），于是 106 的下一个会话又得抢一次，
    #:    每次都打冲突告警，而且抢的窗口内 106 的 Action 是派错的。
    #: 空 = 退回 `ic_grpc`（单机部署语义正确：本机 IC 就是共享的那个）。
    ic_restore: str = field(default_factory=lambda: os.environ.get(
        "ORCH_IC_RESTORE", ""))
    #: Agent Platform 地址（IC 的四类 Action 派给它）
    agent_url: str = field(default_factory=lambda: os.environ.get(
        "ORCH_AGENT_URL", "http://192.168.89.102:8081"))
    #: IC 决策下发给客户端（UI / replay JSONL）的**稳态心跳间隔**（秒）。
    #: 状态变化时**立即**下发，不受它影响 —— 这个值只管「状态没变时多久
    #: 补一条」，用来证明决策还在跑、viz 时间轴不会看起来断掉。
    #: 设 0 = 关掉心跳（只在变化时发）。
    ic_report_interval_s: float = field(default_factory=lambda: float(
        os.environ.get("ORCH_IC_REPORT_S", "10")))

    # TTS 参数
    tts_type: str = "mltts"
    tts_speaker_id: str = "17"
    # 流式合成：LLM 文本 delta 一到就喂 TTS，音频一出就播（首声快很多）。
    # 关掉则退回「整条回复一次合成」。用 ORCH_TTS_STREAMING=0 回退。
    tts_streaming: bool = field(default_factory=lambda: os.environ.get(
        "ORCH_TTS_STREAMING", "1") not in ("0", "false", "False", ""))

    # OmniLLM 系统提示词。
    #
    # ⚠️ **实测结论（2026-09，qwen3omni 后端）**：
    #   · 输入很短的音频（1~3s、几乎没信息量）时，模型听不出中文内容，
    #     会**自己发挥**——最坏情况整段跑成英文。
    #   · **加语种约束有效**：同一段 1.2s 音频，加「始终用中文」后
    #     6/6 都是中文（不加时见过全英文）。实测见下。
    #   · ⚠️ **「没信息就别回复」压不住**：即使写明"保持沉默"，6 次里
    #     只有 1 次真的短，其余仍会发挥几十上百字。**提示词越长越绕，
    #     越容易跑偏**（有一次写长了直接退化成 210 字全英文）。
    #     所以这里**刻意写得短**，别再加长篇约束 —— 要真正"不回复"，
    #     得在**触发侧**做（比如 ASR 置信度太低就不触发），而不是靠提示词。
    #   · 语音播报场景下还要**避免 Markdown**（模型爱输出列表/加粗），
    #     所以点明"口语化、不要格式标记"。
    #
    # 可用 `ORCH_OMNI_SYSTEM_PROMPT` 覆盖（改措辞不用动代码）。
    # 客户端也能在 session.start 里传 `system_prompt` 覆盖它（优先级更高）。
    omni_system_prompt: str = field(default_factory=lambda: os.environ.get(
        "ORCH_OMNI_SYSTEM_PROMPT",
        "你是一个实时视频对话助手。请一边观看用户传来的实时视频画面，"
        "一边倾听并用自然口语即时回复，及时回应画面中出现的内容。"
        "你是由魔珐科技开发的人工智能助手：XmovOmni。"
        "请始终使用中文普通话回复，不要使用英文。"
        "回复要口语化、简短自然，不要用 Markdown 或列表符号。"
    ))
    # 回复触发方式：
    #   "asr"       —— **按 ASR 最终文本触发**（推荐）。OmniLLM 持续以
    #                  force_listen 累积视听上下文（"边听边看"），收到
    #                  ASR 的 2pass-offline 时由我方补触发。回复是"看到
    #                  截止此刻的画面 + 听到整句语音"后生成的。
    #   "turnsense" —— 服务端 VAD+TurnSense 判决（原行为）。
    omni_turn_trigger: str = field(default_factory=lambda: os.environ.get(
        "ORCH_OMNI_TRIGGER", "asr"))

    # 人脸（阶段 4）—— 实现走 board-face-and-cloud-infer/G1 的官方 g1face 包
    #
    # ⚠️ `face_g1_root` 与 `face_model_dir` 是**两个不同的根**，不能合并：
    #    检测模型（blazeface 等）由 C 侧按 `g1_face_create(model_dir)` 的入参解析，
    #    而身份模型 `models/buffalo_l/` 由 G1IdentifyRuntime 按 g1_root 解析。
    #    不设 g1_root 时自动探测「与本仓库同级的 board-face-and-cloud-infer/G1」。
    face_g1_root: Optional[str] = field(default_factory=lambda: os.environ.get(
        "ORCH_G1_ROOT"))
    face_lib_path: Optional[str] = field(default_factory=lambda: os.environ.get(
        "ORCH_FACE_SO"))
    face_model_dir: Optional[str] = field(default_factory=lambda: os.environ.get(
        "ORCH_FACE_MODELS"))
    face_db_path: Optional[str] = field(default_factory=lambda: os.environ.get(
        "ORCH_FACE_DB"))
    #: 唤醒判据：track 连续在场多少毫秒算唤醒。2000 = C 侧 `wake_ms_high`，
    #: 也 = IC 的 passerby 阈值（低于它 IC 判 passerby、永不 GREET），两边必须同值。
    face_wake_dwell_ms: int = field(default_factory=lambda: int(os.environ.get(
        "ORCH_FACE_WAKE_DWELL_MS", "2000")))
    #: 身份识别总开关。关掉只做检测/唇动/唤醒。
    face_identify: bool = field(default_factory=lambda: os.environ.get(
        "ORCH_FACE_IDENTIFY", "1") not in ("0", "false", "False", ""))
    #: 在线注册：**默认关**。上游默认把不在库的人注册进内存库，
    #: 在编排场景会把每个路人都写成"熟人"。我们另外也永不落盘。
    face_no_enroll: bool = field(default_factory=lambda: os.environ.get(
        "ORCH_FACE_NO_ENROLL", "1") not in ("0", "false", "False", ""))
    #: 识别阈值（LOW/MEDIUM 分界）。0.36 来自上游 100 轮交叉验证。
    face_threshold: float = field(default_factory=lambda: float(os.environ.get(
        "ORCH_FACE_THRESHOLD", "0.36")))

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
