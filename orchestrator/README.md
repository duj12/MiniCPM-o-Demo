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
python -m orchestrator.tests.test_ref_track

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
    ref_track.py       TTS 参考轨 + 声学延迟估计（GCC-PHAT）
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
