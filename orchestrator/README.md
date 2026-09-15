# Orchestrator —— 音视频输入 + TTS 输出模块

把浏览器端的**音视频采集**与 **TTS 播报**做成一个可独立部署的服务：
音频经云端 AEC 清洗后扇出给 ASR 与 OmniLLM，视频驱动人脸/唇动，TTS
合成结果回浏览器播放并闭环进 AEC 参考。

**本模块不含 Policy / Agent** —— 只预留接口，决策逻辑后续接入。

```
┌─ Browser ─────────────────────────────────────────────────────────┐
│ ① mic 16k float32, 100ms/块 ─────────────────────┐                 │
│ ② TTS PCM 播放 + playback 回执 ──────────────────┼──┐              │
│ ③ camera ─┬─ 25fps 小图(人脸) ───────────────────┼──┼──┐           │
│           └─ 1fps  大图(OmniLLM) ────────────────┼──┼──┼──┐        │
│ ④ 播放 TTS 音频 ←────────────────────────────────┼──┼──┼──┼────┐   │
└──────────────────────────────────────────────────┼──┼──┼──┼────┼───┘
                    WS /v1/orchestrator
┌──────────────────────────────────────────────────┼──┼──┼──┼────┼───┐
│ Orchestrator                                      │  │  │  │    │   │
│  · SampleClock  16k 采样时钟（唯一时间基准）       │  │  │  │    │   │
│  · RefTrack     TTS PCM 按回执落到 mic 时钟        │  │  │  │    │   │
│  · AecClient ─────────────────────────────────────┼──┘  │  │    │   │
│  · FaceWorker（G1 + FaceService，专用线程）────────┼─────┘  │    │   │
│  · AsrClient / OmniClient / TtsClient ────────────┼────────┼────┘   │
│  · Downstream（Policy/Agent 预留）────────────────┼────────┼────────┤
└───┬──────────────┬──────────────┬───────────────┬┴────────┴────────┘
    │              │              │               │
[AEC 服务]      [ASR 服务]    [Qwen3-Omni]    [TTS 服务]
 WebSocket      WebSocket      gateway :8006     gRPC
```

**音频链路顺序：`mic → AEC → {ASR, OmniLLM}`**。
AEC 只调一次，扇出在其后 —— 绝不两路各自调 AEC（它是有状态流式算法，
状态会被破坏）。

---

## 快速开始

### 依赖

```bash
# 运行环境：Python 3.10（106 上为 py310）
pip install websockets fastapi uvicorn numpy
# TTS 需要 grpc + TTS 仓库的 protos
pip install grpcio protobuf
```

### 启动

```bash
cd MiniCPM-o-Demo
python -m orchestrator.main --port 8100
```

打开 `http://<host>:8100/` 是自带的验证页（授权摄像头麦克风即可跑通全链路）。

### 常用参数

| 参数 | 说明 |
|---|---|
| `--port` | 监听端口（默认 8100） |
| `--downstream-mode` | 桩模式：`asr`（ASR 最终结果→TTS）/ `omni`（OmniLLM 回复→TTS）/ `echo` / `none` |
| `--no-aec` / `--no-asr` / `--no-omni` / `--no-tts` | 分开关闭各组件（便于定位问题） |
| `--mock-tts` | 用本地正弦代替 TTS 服务（离线验证链路） |

### 环境变量

所有外部服务地址走配置，见 `orchestrator/config.py`：

```bash
export ORCH_AEC_URL="ws://192.168.88.253:30255/ws/asr_frontend"
export ORCH_ASR_URL="ws://192.168.88.101:31366"
export ORCH_TTS_HOST="192.168.88.253"
export ORCH_TTS_PORT="31058"
export ORCH_OMNI_URL="wss://127.0.0.1:8006/v1/realtime?mode=video"
export ORCH_DOWNSTREAM_MODE="omni"
```

### 人脸（默认关，需显式开启）

```bash
export ORCH_ENABLE_FACE=1
export ORCH_FACE_SO="$CODE/board-face-and-cloud-infer/G1/lib/libsdk_stream.so"
export ORCH_FACE_MODELS="$CODE/board-face-and-cloud-infer/G1/models"
export ORCH_FACE_DB="$CODE/faceidentification/data/face_db.npz"
export G1_FACE_DEBUG=0     # ⚠️ 必须！否则 create 就写最多约 4GB 视频
```

---

## HTTP 接口

| 路径 | 用途 |
|---|---|
| `GET /` | 验证页（`static/orchestrator-test.html`） |
| `GET /healthz` | 健康检查 + 活跃会话数 |
| `GET /stats` | 各活跃会话的原始统计 |
| `GET /metrics` | 全局指标 + 最近会话快照（可接 Prometheus） |
| `WS /v1/orchestrator` | 会话主通道 |

---

## WS 协议

### 客户端 → 服务端

```json
{"type":"session.start","identity":{...},"seeded_delay_samples":0}
{"type":"audio","audio_base64":"...","t_ms":0}        // float32 raw, 16kHz
{"type":"video_face","frame_base64":"...","t_ms":0}   // 25fps 小图
{"type":"video_omni","frame_base64":"...","t_ms":0}   // 1fps 大图
{"type":"playback","response_id":"r1","phase":"started","ctx_time":812.4,"seq":0}
{"type":"session.stop"}
```

**音频在线保持 float32**（与采集层一致）：云端 AEC 与 OmniLLM 都要
float32，只有 ASR 要 int16 —— 服务端转一次优于浏览器转了再转回来。
带宽：100ms float32 16k 单声道 = 6.4KB → base64 8.5KB × 10/s = **85KB/s**。

### 服务端 → 客户端

```json
{"type":"session.ready","session_id":"...","sample_rate":16000}
{"type":"tts.start","response_id":"r1","text":"...","sample_rate":24000}
{"type":"tts.audio","response_id":"r1","seq":0,"audio_base64":"..."}  // int16 PCM 24k
{"type":"tts.end","response_id":"r1"}
{"type":"tts.cancel","response_id":"r1","reason":"bargein"}
{"type":"asr","phase":"partial|final","text":"...","t_ms":0}
{"type":"face.state","tracks":[...],"identity":{...}}
{"type":"error","code":"...","message":"..."}
```

### ⚠️ 播放回执是 AEC 的关键一环

AEC 的参考信号必须与**实际播出**的时刻对齐（不是音频到达 orchestrator
的时刻 —— 中间隔着发送、抖动缓冲、播放器 200ms 提前量）。浏览器必须在
起播时报 `playback.started`、取消时立即报 `playback.cancelled` ——
后者让云端 `truncate()` 参考轨，否则 AEC 会拿着没播出的音频当参考，
**主动误适配去追一个不存在的回声，比不给参考更糟**。

#### ⚠️⚠️ 但 `ended` **绝不能**截断参考轨（真机头号故障）

这是真机上「算法 AEC 完全没起作用」的**根本原因**，务必不要退回：

`truncate(rid, from_sample=clock.now())` 的语义是「从**当前会话时刻**
往后清空」，它假定音频已经播到那儿了。而：

  · TTS 是**整段一次性**送达并整段落位的（`executor._speak_inner`）
  · `tts.end` 到达浏览器时，浏览器**才刚开始播**（还有 200ms 提前量）

于是"当前位置"远在整段音频之前 → **整段参考被清掉，只剩到达那一瞬间的
一小截**。真机实测（会话 `s-6d21f3ab5f3c`）：TTS 报 5.85s、ref 写入区间
也正好 5.85s，但实际非零只有 **0.6s**（日志 `非零 6/152 = 4%`）——参考
只剩 10%，AEC 等于没有 farend。用户听到的现象是"ref 里只剩开头几个字、
还被拉得很长"。

**只有 `cancelled` 才截断**。播放是排好程的（WebAudio 按 `nextAt` 连续
排），`ended` 只表示"音频已全部交给播放器"，**不代表已经播完**；没用上
的部分由后续播报覆盖或被环形缓冲淘汰，不需要主动清。
回归护栏：`tests/test_ref_truncate_bug.py`。

修复前后对比（真机）：

| 指标 | 修复前 | 修复后 |
|---|---|---|
| farend 非零占比 | 4% | **43%** |
| ref 写入区间 | 5.85s | **23.6s** |
| ERLE | 8.3 dB | **31.6 dB** |

### ⚠️⚠️ 声学延迟 D 必须准到 ±5ms（算法 AEC 的生死线）

实测（`tests/test_aec_live_fidelity.py`，真实 AEC 服务）：

| 补偿 D | 回声抑制 |
|---|---|
| 84ms（真值） | **12.6 dB** |
| 90ms | 2.5 dB |
| 250ms（旧默认值） | **0.3 dB**（等于完全不工作） |

**容忍窗只有约 ±5ms** —— 差 6ms 就从 12.6dB 掉到 2.5dB。所以：

- 默认值 `aec_default_delay_ms` 是 **0**（"假定浏览器按约定提前量起播"），
  不是 250。250 是把**全程往返**当成了参考轨要补的残差 —— 而
  `RefTrack.place()` 的落位已含 200ms 提前量，D 只需补剩下的声学延迟。
- 服务端**不做**时延对齐（`SD_AEC` 是硬编码的 ONNX 模型；仓库里的
  `GCCPHATDelayEstimator`/`LinearAEC` 在流式路径中是死代码，唯一"对齐"
  是 alpha predictor 里 k=10 帧 ≈100ms 的学习式 lookback）。
  **所以调用方必须自己保证样本级预对齐** —— 这正是 `RefTrack` 的职责。
- 前端会显示 D 的三态（未测量 / 收敛中 / 已收敛）。**显示"未测量"时
  算法 AEC 基本不会生效**，别把它当成一个正常数字。

### farend 量纲（应当归一化，但别高估其影响）

TTS 返回 **int16**（±32768），麦克风是 **[-1,1] float32**。
`RefTrack.place()` 会归一化 —— 这是"不该靠模型兜底"的正确做法。

⚠️ 但**实测影响很小**（`tests/test_aec_scale_bug.py`，同一段 mic 只改
farend 量纲）：

| farend | ERLE | 近端保真 |
|---|---|---|
| 归一化 [-1,1] | 5.0 dB | +1.1 dB |
| int16 量纲 | 5.8 dB | +0.3 dB |
| 放大 100× | 5.9 dB | +0.2 dB |

模型对量纲不敏感。归一化保留，但它**不是**"回声消不掉"的原因 ——
真正的主因是上面那条 D 对齐（容忍窗 ±5ms）。

---

## 下游接口（Policy / Agent 预留）

`orchestrator/downstream/interface.py` 定义了完整契约。下游是**纯决策**：
不碰传输、不持 socket、不 sleep，只返回 action 由 Orchestrator 执行。

```python
class Downstream(Protocol):
    def describe(self) -> dict: ...
    async def on_session_start(self, ctx) -> list[DownstreamAction]: ...
    async def on_event(self, ev: DownstreamEvent) -> list[DownstreamAction]: ...
    async def on_session_end(self, reason: str) -> None: ...
```

**本阶段用 `PassthroughDownstream` 桩**（把 ASR 最终结果或 OmniLLM 回复
直接作为 `Speak`），让链路可端到端验证。Policy 真实实现接入时只替换这个
实例，其余不动。

**排序契约**：事件按**到达顺序**投递，不保证 `t` 单调 —— Orchestrator
不为排序回压（实时路径上那需要无界重排窗口）。下游须容忍迟到事件，
需要时序推理时用 `Tick` 自建重排窗口。

---

## 运维要点

### 容量

实测（106，真实 AEC+ASR+OmniLLM+TTS，8s 音频/路）：

| 并发 | 建连中位 | 会话耗时 | 失败 |
|---|---|---|---|
| 4 | 895ms | ~12s | 0/4 |
| 8 | 1549ms | ~14s | 0/8 |
| **16** | **4359ms** | **15~26s** | **1/16** |

**拐点在 8~16 路之间。** 原因是 AEC 服务**每条 WS 连接 = 一个独占
`StreamInference` 实例**，且不能靠 `--workers` 横向扩（会复制多份显存）。
生产扩容需 AEC 服务多实例 + 会话亲和路由。

### 观测

`GET /metrics` 返回全局与会话指标。日志里每次会话结束有一行摘要：

```
指标: up=13s audio_in=95chk/10s aec=48seg(首窗331ms) asr=10p/2f
      omni=223d/1done tts=1call(总171ms) face=0frm drop=0
```

### ⚠️ 排「回声没消掉」：**先看波形，别看指标**

这是本项目最贵的一课。ERLE / 峰比 / 非零占比 / mic_rms 这些间接指标
已经把人带偏过**两次**（一度推出"麦克风里没有回声"这种与"ASR 一直能
识别到播报"**直接矛盾**的结论，浪费了一轮真机验证）。

开启会话级音频转储：

```bash
ORCH_DUMP_AUDIO=/data/.../orchdump/s python -m orchestrator.main ...
```

会话结束时写三个 wav（默认关闭，实时路径零开销）：

| 文件 | 内容 | 用它能判断 |
|---|---|---|
| `<前缀>-<sid>-mic.wav` | 浏览器送来的**原始麦克风**（AEC 之前） | 里面有没有回声 |
| `<前缀>-<sid>-ref.wav` | 我们算的 **farend**（以为在播什么） | 与实际播报是否一致、长度对不对 |
| `<前缀>-<sid>-aec.wav` | **AEC 输出**（ASR 听的就是这个） | 回声消掉多少 |

三个波形一比即可定论。真机故障就是这样定位的：`ref.wav` 只有 0.6s 有
内容而 TTS 报了 5.85s → 直接指向 `truncate` bug（见上文）。

前端侧也有对应的「导出诊断」（环境/约束/D/ERLE/日志）与
「导出校准录音」（校准通道的 mic + 播放信号，配
`tests/analyze_calib_wav.py` 可区分"没播出来/太轻/被设备侧消掉"三种失败）。

### 收尾时序（易踩）

会话结束分两步：

1. **drain（可慢）**：AEC 推完尾部窗口 → OmniLLM 说完整当前这轮 →
   ASR 等最终结果 → 等 TTS 音频真正送达浏览器
2. **close（必须快）**：断开所有连接

收尾放在**独立后台任务**里跑（`SHUTDOWN_TASKS`）—— 客户端断开时处理
协程已被取消，直接在 `finally` 里 await drain 会撞 `CancelledError`，
表现为 uvicorn 的 ASGI 异常 + 尾部数据丢失。

---

## 测试

```bash
# 无外部服务（本地骨架 + MockTTS）
python -m orchestrator.tests.test_e2e_local --no-aec --no-asr --no-omni --mock-tts --downstream-mode none

# 完整链路（需真实服务；⚠️ 验证 ASR final/TTS 闭环**必须**用真实语音，
# 合成信号会被 ASR 判为 invalid，永远不产出最终结果）
python -m orchestrator.tests.test_e2e_local --downstream-mode omni \
  --wav assets/ref_audio/ref_minicpm_signature.wav --expect-tts

# 并发容量
python -m orchestrator.tests.test_concurrent --n 8 --wav assets/ref_audio/ref_minicpm_signature.wav

# 拆除路径（资源泄漏）
python -m orchestrator.tests.test_teardown --rounds 20 --wav assets/ref_audio/ref_minicpm_signature.wav

# 人脸模块（需 G1 库 + 模型）
python -m orchestrator.tests.test_face --so <libsdk_stream.so> --models <models> \
  --mjpeg <camera_original.mjpeg> --tsv <camera_capture_timestamps.tsv>

# 单元测试（无外部依赖）
python -m orchestrator.tests.test_clock
python -m orchestrator.tests.test_ref_track    # 含 D 恢复 + 量纲回归护栏

# 校准（无外部依赖 / 或走真实端点）
python -m orchestrator.tests.test_calibrate        # 算法：已知延迟的合成回声
python -m orchestrator.tests.test_calibrate_e2e    # 端点：模拟浏览器走全流程

# ⭐ 保真验证：线上链路重建 vs 离线实验（需 AEC 服务）
#    回答"网页真机调用能否产出与离线模拟实验等价的输出"
python -m orchestrator.tests.test_aec_live_fidelity \
  --far assets/ref_audio/ref_minicpm_signature.wav \
  --near assets/ref_audio/ref_en_dlc_1.wav \
  --out /tmp/aec_live        # ⚠️ 106 的 / 分区已满，用 /data/... 下的路径

# 服务量测（阶段 0 工具）
python -m orchestrator.tests.probe_aec --duration 60
python -m orchestrator.tests.probe_asr --wav <16k.wav> --verbose
python -m orchestrator.tests.probe_tts          # 需在 106（要 grpc + protos）
```

见 `orchestrator/tests/README.md` 的实测结果记录。

---

## 目录

```
orchestrator/
  main.py              FastAPI WS 服务（会话装配、收尾编排）
  session.py           OrchestratorSession —— 串起所有流水线
  clock.py             SampleClock / AudioFrame / TrackBuffer
  protocol.py          浏览器 ↔ 服务端的 WS 协议
  config.py            配置（外部服务地址走环境变量）
  metrics.py           指标（延迟滑动窗口 + 计数器）
  audio/
    resample.py        有状态重采样（24k→16k，跨块保相位）
    ref_track.py       TTS 参考轨 + 声学延迟估计（GCC-PHAT）+ 量纲归一化
  calibrate.py         主动校准：双段啁啾 + 起播锚点 + 自洽性校验
  calibrate_endpoint.py /v1/calibrate 独立校准通道
  delay_store.py       D 的持久化（新旧值差 >150ms 直接覆盖，不做滑动平均）
  aec/client.py        AEC WebSocket 客户端
  asr/client.py        ASR 客户端 + 消息解析
  omni/client.py       OmniLLM 客户端（复用 StreamingChatClient）
  tts/client.py        TTS 客户端（同步 stub + 线程池）+ MockTtsClient
  face/
    g1.py              G1 库 ctypes 绑定
    worker.py          专用线程 + 有界队列
    signals.py         数据契约
    local_provider.py  G1 + FaceService 组合
  downstream/          下游接口契约 + 确定性桩
  actions/executor.py  执行 action（TTS 合成、播放、取消）
  tests/               验证脚本与实测记录
```
