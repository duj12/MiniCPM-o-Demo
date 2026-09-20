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

---

## 启动服务

### 生产：用 `orchestrator/run_orch.sh`

**推荐用脚本，不要手工 `nohup`** —— 人脸相关的环境变量漏一个，
表现就是"人脸模块突然没了"（`face=False`），很难查。

```bash
cd MiniCPM-o-Demo
setsid nohup bash orchestrator/run_orch.sh </dev/null >orch.log 2>&1 &
```

⚠️ **从哪个目录跑都行** —— 脚本按自身位置推导 `REPO` 与 `CODE`
（`CODE` = 与 `MiniCPM-o-Demo` 同级，`board-face-and-cloud-infer` /
`faceidentification` 都在那里）。

脚本集中管理了：算法 AEC 的延迟 D、音视频转储、人脸模块、HTTPS 证书。
每个变量都可用环境变量覆盖（如 `ORCH_PORT=9000 bash orchestrator/run_orch.sh`），
`ORCH_PY` 可换 Python 解释器。

看启动结果：

```bash
tail -5 orch.log
# 期望：能力: aec=True asr=True omni=True tts=True face=True downstream=omni
#       HTTPS 已启用：https://0.0.0.0:8100
```

### 开发/调试：直接起

```bash
cd MiniCPM-o-Demo
python -m orchestrator.main --port 8100
```

### 参数

| 参数 | 说明 |
|---|---|
| `--host` / `--port` | 监听地址/端口（默认 `0.0.0.0:8100`） |
| `--downstream-mode` | 桩模式：`asr` / `omni`（默认）/ `echo` / `none` |
| `--ssl-cert` / `--ssl-key` | 启用 HTTPS（**成对给**）。见下面「为什么必须 HTTPS」 |
| `--no-aec` / `--no-asr` / `--no-omni` / `--no-tts` | 分开关闭各组件（便于定位问题） |
| `--mock-tts` | 用本地正弦代替 TTS 服务（离线验证链路） |

### ⚠️ 为什么必须 HTTPS

浏览器只在**安全上下文**（`https://` 或 `localhost`）里给 `getUserMedia`
（麦克风/摄像头）。用 `http://<局域网IP>:8100` 打开时 `navigator.mediaDevices`
**直接是 undefined** —— 页面看着正常，就是采不到任何音视频。

自签证书（`certs/cert.pem`）的 **SAN 里必须包含你实际访问用的地址**，
否则浏览器不认（**Edge/Chrome 只看 SAN，完全忽略 CN**）：

```
X509v3 Subject Alternative Name:
    IP Address:192.168.89.106, IP Address:127.0.0.1, DNS:localhost, DNS:ubuntu
```

换机器/换 IP 就要重新签（`openssl req -x509 -config ...`，`v3_req` 里带
`subjectAltName`），或者直接用 `mkcert`。首次访问浏览器会拦一次，
点「继续访问」即可。

**Tailscale**：它是以明文回源 `http://127.0.0.1:8100` 的，8100 改 HTTPS
之后要同步改回源，否则那条路会断：

```bash
sudo tailscale serve --bg --https=443 https+insecure://localhost:8100
```

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

### 音视频转储（排障必备）

```bash
export ORCH_DUMP_AUDIO="$CODE/orchdump/s"
```

设了就会在**会话结束时**把输入落盘（不设 = 零开销）：

| 文件 | 内容 |
|---|---|
| `s-<sid>-mic.wav` | 原始麦克风（**含回声**） |
| `s-<sid>-ref.wav` | 已按 D 补偿的参考轨（喂给 AEC 的那份） |
| `s-<sid>-raw.wav` | **未补偿**的原始参考轨（测 D 必须用这个） |
| `s-<sid>-aec.wav` | AEC 输出 |
| `s-<sid>-face.mjpeg` + `.tsv` | 25fps 人脸帧，**原样存的 JPEG** |
| `s-<sid>-omni.mjpeg` + `.tsv` | 1fps Omni 帧 |

`<sid>` 就是网页状态栏显示的**会话 id** —— 网页上看到哪个，就去
`orchdump/` 里找哪个。

⚠️ **会话进行中不会有文件**，是 `close()` 时才写。
⚠️ 视频存的是 **MJPEG 不是 mp4**：发给 Omni/face 的就是原始 JPEG 字节，
转 mp4 要重编码（有损）且丢掉逐帧时间对应，没法复现问题。要看的话自己转：
`ffmpeg -i s-x-face.mjpeg out.mp4`。

---

## 决策链路：InteractionCore + Agent（默认开）

回复**不是** OmniLLM 生成的。OmniLLM 只产出音视频描述，**谁来开口、说什么**
由这两者决定：

```
感知（ASR state / 人脸 state / TTS 播放态）
        │  gRPC apply_*          10Hz~25Hz 写状态
        ▼
  InteractionCore（gRPC，默认 localhost:50051）
        │  Tick() → Action       我们每 50ms 问一次
        ▼
  ANSWER/INSERT/YIELD/END  →  Agent Platform（HTTP）
  GREET/UTTER              →  本地模板文案，直接播
        │
        ▼
  Agent 生成回复文本 → POST /v1/speak → TTS → 扬声器
```

**职责边界**（PRD §3.2 的「谁写」）：

| 谁 | 写什么 |
|---|---|
| orchestrator | `apply_face/track/lip/identity`（人脸 state）、`apply_vad/asr`（ASR state）、**`apply_playback_active`**（表达层事实） |
| Agent | `apply_agent(status, session_end_pending)` —— 它自己的投影，我们不写 |

**为什么回复要绕回 orchestrator 播**：浏览器播放必须和 AEC 参考轨、播放回执
严格对齐（回声消除靠它）。Agent 直接调 TTS 会绕过这些，回声消不掉。

### 配置

```bash
export ORCH_INTERACTION=1                    # 默认就是 1；=0 关掉
export ORCH_IC_GRPC="localhost:50051"        # InteractionCore 地址
export ORCH_AGENT_URL="http://192.168.89.102:8081"
```

⚠️ **IC 由谁启动**：本服务**不启动也不管理 IC 进程**，只要求一个可达的地址。
离线套件 `interactioncore/scripts/run_offline_sop_suite.py` 会自己起 IC；
真实联调时由部署方起：

```bash
cd interactioncore && python -m interaction.grpc_server --port 50051
```

### ⚠️ IC 连不上 → 降级为 OmniLLM 回复

IC 是默认的**唯一回复来源**，连不上会变成「能识别、但永远不回复」——
与我们修过的 OmniLLM 断线 bug 同一病理，**极难排查**。所以有显式降级：

```
建会话时试连 IC
  ├─ 成功 → InteractionDownstream（IC 决策 + Agent 回复）
  └─ 失败 → ⚠️ 告警 + 回退 PassthroughDownstream（OmniLLM 回复）
```

日志里会明确打：

```
[sid] InteractionCore 不可用（localhost:50051：...）—— 降级为 OmniLLM 回复
```

**降级只有一层**，已经是 OmniLLM 就不再降。想主动关掉 IC 用
`ORCH_INTERACTION=0`，或客户端在 `session.start` 里传 `{"ic":{"enabled":false}}`。

### 环境依赖

`interactioncore` 要求 `grpcio>=1.84.0` / `protobuf>=7.35.1`（比 TTS 的
gencode 新）。实测**旧 gencode + 新 runtime 兼容**，升级不影响 TTS 客户端。

```bash
cd interactioncore && pip install -e .
```

### Agent 侧要做什么

**Agent 需要调我们的 `POST /v1/speak` 把回复文本送进来** ——
完整接口说明 + curl / Python 示例见
[`docs/agent-integration.md`](docs/agent-integration.md)。

---

## HTTP 接口

| 路径 | 用途 |
|---|---|
| `GET /` | 验证页（`static/orchestrator-test.html`） |
| `GET /healthz` | 健康检查 + 活跃会话数 |
| `GET /stats` | 各活跃会话的原始统计 |
| `GET /metrics` | 全局指标 + 最近会话快照（可接 Prometheus） |
| `POST /v1/speak` | **Agent 回调**：整段播报（见下节） |
| `POST /v1/speak/stream` | **Agent 回调**：流式播报 |
| `WS /v1/orchestrator` | 会话主通道 |

---

## 客户端怎么用

有两种：**网页**（真人对着麦克风说话）和**离线回放**（喂预录音视频，
可重复、可断言）。排查问题建议两个都用 —— 网页复现、回放复验。

### ① 网页（`static/orchestrator-test.html`）

浏览器打开 `https://<host>:8100/`（**必须 https**，见上面「为什么必须 HTTPS」），
授权麦克风和摄像头，点「开始会话」。

页面上的东西：

| 区域 | 看什么 |
|---|---|
| ASR 字幕 | 流式转写 + 五个状态量 `[说/抢/信/完]` |
| 播报记录 | TTS 文本，**边生成边显示**（流式合成） |
| 音频面板 | `AudioContext` 状态、**TTS 播放中/剩余 Ns**、设备输出延迟 |
| 计数器 | `TTS 帧 / 已调度 / 峰值` —— 判断"有没有声音" |
| 会话 id | **和 `orchdump/` 的文件名对应**，排障时记下它 |

几个开关：

- **回声消除**：默认**浏览器原生 AEC**（开箱可用）。要试算法服务 AEC
  需选它 + **外放 + 离线实测 D**，否则回声消不掉（占位值 D=250 实测
  抑制只有 0.3dB，等于不工作）
- **没有摄像头也能用**：自动降级为纯音频（页面会明确提示），
  OmniLLM 照常对话，只是人脸检测/唇动不可用

### ② 离线回放（`orchestrator_replay.py`）

用 Python 扮演浏览器，把预录音视频按**真实实时节奏**喂给编排服务，
并处理回来的 TTS / ASR / 人脸状态。适合回归验证与复现。

```bash
# 最简：位置参数，按扩展名自动判断音频/视频
python orchestrator_replay.py assets/ref_audio/ref_minicpm_signature.wav \
    --host 192.168.89.106

# 视频（自动抽帧：人脸 320×240 @24fps、Omni 1280×720 @1fps）
python orchestrator_replay.py assets/video/turnbased/121.mp4 --host 192.168.89.106
```

（`assets/video/` 下还有几段更长的，如 `test.mp4` 约 7 分钟 —— 适合跑
「长时间多轮对话」这类场景。）

**默认全开**：播放输入音频、播放 TTS、开窗显示、verbose。通常不用加参数。

| 想要 | 加什么 |
|---|---|
| 静音 | `--mute` |
| 不开窗 | `--no-show` |
| 只播 TTS / 只播输入音频 | `--no-play-audio` / `--no-play-tts` |
| 连老式明文服务 | `--ws`（默认 `wss://`，且**不校验自签证书**） |
| 存服务端回来的 TTS | `--save-tts /tmp/tts` |
| 存带叠加层的视频 | `--save-video out.mp4` |
| 模拟插话 | `--bargein-at 5,12`（在第 5、12 秒打断） |
| 换个播放设备 | `--audio-device <名字或编号>` |
| 安静输出 | `--no-verbose` |
| **接 InteractionCore** | `--ic-grpc localhost:50051` |
| 指定 Agent 地址 | `--ic-agent-url http://…` |
| 关掉 IC（走 OmniLLM） | `--no-ic` |
| IC 事件落 JSONL | `--ic-events-out events.jsonl` |

⚠️ **`--replay-speed` 不要乱调**：服务端按实时流设计，加速会让 AEC/ASR
表现失真。默认 1.0 是**慢放**（音频按 100ms/块真实节奏发）。

⚠️ **本工具不模拟声学回声** —— mic 通道就是文件里的干净音频，TTS 播报
不会被"回采"进输入通道。所以它测不了 AEC 的消除效果（要测回声闭环用
`tests/test_duplex_sim.py`，它会合成 `mic = 干净人声 + gain × 喇叭[t−D]`）。

跑完会打印汇总：TTS 段数与总时长、ASR 部分/最终结果、最终配置、墙钟。

---

## WS 协议

### 客户端 → 服务端

```json
{"type":"session.start","identity":{...},"seeded_delay_samples":0,
 "caps":["playback_anchor"]}                          // ← 能力位，缺了会退回预测落位
{"type":"audio","audio_base64":"...","t_ms":0,
 "ctx_time":812.4,"epoch":12345}                      // float32 16kHz + 本块首采样的 ctx 时刻
{"type":"video_face","frame_base64":"...","t_ms":0}   // 25fps 小图
{"type":"video_omni","frame_base64":"...","t_ms":0}   // 1fps 大图
{"type":"playback","response_id":"r1","phase":"armed","start_ctx":812.6}      // 承诺起播时刻
{"type":"playback","response_id":"r1","phase":"started","start_ctx":812.6}    // 实际排程时刻
{"type":"playback","response_id":"r1","phase":"ended"}
{"type":"playback","response_id":"r1","phase":"cancelled",
 "sample_offset":48000,"stop_ctx":815.9}              // 实测播出量 + 实际停下时刻
{"type":"session.stop"}
```

`ctx_time` / `epoch` / `start_ctx` / `stop_ctx` 都是**浏览器 AudioContext
时钟**上的量，服务端只经 `SampleClock.ctx_to_sample()` 换算后使用，
**绝不直接当会话采样位置**（早期那么做过，写入位置变成大负数、参考轨
读出来全是 0）。`epoch` 是 context 代号：换了 context 后 `currentTime`
归零，异 epoch 的锚点会被全部作废。

**音频在线保持 float32**（与采集层一致）：云端 AEC 与 OmniLLM 都要
float32，只有 ASR 要 int16 —— 服务端转一次优于浏览器转了再转回来。
带宽：100ms float32 16k 单声道 = 6.4KB → base64 8.5KB × 10/s = **85KB/s**。

### 服务端 → 客户端

```json
{"type":"session.ready","session_id":"...","sample_rate":16000,"lead_ms":200}
{"type":"tts.start","response_id":"r1","text":"...","sample_rate":24000,"lead_ms":200}
{"type":"tts.audio","response_id":"r1","seq":0,"audio_base64":"..."}  // int16 PCM 24k
{"type":"tts.end","response_id":"r1"}
{"type":"tts.cancel","response_id":"r1","reason":"bargein"}
{"type":"asr","phase":"partial|final","text":"...","t_ms":0}
{"type":"face.state","tracks":[...],"identity":{...}}
{"type":"error","code":"...","message":"..."}
```

### ⚠️⚠️ 参考轨的播出时刻必须由浏览器**承诺**（本轮修复的核心）

这是「第一句能消回声、后面就失效」的根因，也是本模块最重要的一条契约。

AEC 的参考信号必须与**实际播出**的时刻对齐到毫秒级。曾经的做法是服务端
**预测**：「送完音频时的 `clock.now()` + 播放提前量」。这个预测有两层误差：

  ① `clock.now()` 由浏览器送来的 100ms mic 块推进，**比真实时刻慢
     0~100ms 且逐块抖动**（浏览器主线程同时在跑 25fps 抓帧）；
  ② 浏览器是收到 `tts.audio` 之后 `ctx.currentTime + 提前量` 才起播，
     中间还隔着网络往返与浏览器处理耗时。

**关键在于这个误差不是常量，而是逐句变化的** —— `test_duplex_sim.py`
实测：同一次会话里，第二句播报的预测误差比第一句**大出 888ms**。而
AEC 的容忍窗只有 ±5ms，固定常量 D 只能吸收"恒定的偏差"，吸收不了"变化的
偏差"，于是会话刚开场时碰巧对上、之后就一路散掉。

**现在的做法**：让浏览器**承诺**播出时刻，而不是让服务端猜。

```
① 服务端发 tts.start（含 lead_ms）
② 浏览器：at = ctx.currentTime + lead      ← 承诺，此时音频还没到
   立刻回 playback{phase:'armed', start_ctx: at}
③ 服务端：T_play = clock.ctx_to_sample(at) ← 精确换算，然后才 place() 参考轨
④ 浏览器按 at 排程播放
⑤ 浏览器回 playback{phase:'started', start_ctx: 实际排程时刻}
   服务端只比对偏差并告警，**不据此修正参考轨**
```

②的往返**完全藏在 TTS 合成的 508ms 首帧延迟里**（`tts.start` 在合成之前
发出），零额外延迟。③的换算靠每个 mic 块自带的 `ctx_time` 拟合出
「浏览器时钟 ↔ 会话采样」的仿射映射 —— 两个时钟域数的是同一路 16kHz 流，
所以这条映射是**精确**的（实测残差 < 0.1ms）。

**副产品**：网络往返、浏览器主线程抖动、mic 在途积压**全部从 D 里剔除**，
D 退化成「扬声器→麦克风的物理延迟 + 设备音频 I/O 缓冲」—— 每台设备一个
**固定常量**，离线测一次即可（见下文）。

#### 为什么是「承诺」而不是「事后上报 + 修正」

事后修正需要在参考轨上提供 `realign()`：重采样器是有状态的、不能重放，
得额外保留每句的 x16、记下 truncate 的切口、理清 realign×truncate×resize
的交互 —— 那是本仓库 bug 历史最重的文件。而且修正到达之前，AEC 拿到的是
**位置错的参考**，它会主动误适配（比不给参考更糟）。用承诺协议把这类状态
直接设计掉。

#### `started` 只告警、不修正

`started` 带回真实排程时刻，服务端算出偏差，超过 5ms 就 `logger.warning`
并在 `session.stats.anchor_delta_ms` 里露出。**不据此重写参考轨**：几毫秒
的对齐误差损失几个 dB 抑制，而"截断重写"一旦时机错位就是整段回声漏进
ASR（正是我们要修的故障）。

#### 降级路径必须可见

客户端没声明 `playback_anchor` 能力位（旧前端）或承诺超时（500ms）时，
服务端退回预测落位，并把 `anchor_source` 标成 `"predicted"`。页面上的
「落位锚点」会显著地显示成红色 —— **它不是个可以忽略的细节**，看到
"服务端预测"就意味着回声对齐已经不可信了。

回归护栏：`tests/test_duplex_sim.py`（`--anchor armed` vs `--anchor legacy`
的对照）、`tests/test_bargein_ref.py`、`tests/test_clock.py`。

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

#### ⚠️ 打断（barge-in）必须由服务端主动做，且要掐掉已排程的音频

现象：上一句还没播完就插话，**新回复的回声完全消不掉**、被 ASR 整段识别。

四个缺陷叠加：

  ① 前端 `PcmPlayer.stop()` 是**空操作**（gain 设 0 又立刻设回 1），而
     WebAudio 里 `node.start()` 排程过的 buffer **无法取消** → 旧句照播；
  ② 前端 `beginResponse` 里 `nextAt = max(nextAt, now+0.2)`，而 `nextAt`
     还停在**旧句末尾** → 新句被排到旧句之后；
  ③ 服务端把新句落位在"当前时刻 + 提前量"，与②的实际排程不符；
  ④ 新句开始时**没人打断旧句** —— `_current_response_id` 被直接覆盖，
     旧句的落位留在轨上。

→ mic 里的回声来自**旧句**，farend 写的是**新句**，两者无关，AEC 失效。

修法：

  · 前端 `stop()`：登记每个已排程的 `AudioBufferSourceNode`，逐个
    `stop(0)` 掐断（这是 WebAudio 里唯一能取消已 start 源的办法），
    并复位 `nextAt` 让下一句从零重排
  · 服务端 `_speak_inner` 开新句前调 `_interrupt_current()`（同步、原子）
  · **截断点 =「已经播到哪」的偏小估计 + 300ms 余量**：
      - 还没起播 → 从 `started` 起清 → 整句清干净
      - 已播到中途 → 保留 `[started, cut)`
      - ⚠️ **偏差方向不能搞反**：`resize`（前端回执到达后的精确校正）
        **只能缩短**参考轨。所以临时截断必须**少清** —— 留多了由
        `resize` 精确剪掉，**留少了永远补不回来**，那就是"有回声、没
        farend"，AEC 直接失效。
      - 估计用 `clock.ctx_now_estimate()`（按锚点算浏览器此刻的播出位置），
        不是 `clock.now()` —— 后者是"已 ingest 的采样数"，系统性落后
        100~300ms。实测：用 `clock.now()` 时打断后的参考轨比实际播出量
        **少 6421 采样（401ms）**，那段回声完全没有参考。
  · `RefTrack.truncate` 按 chunk 求交集，**只清本 response 自己的区间** ——
    早先无差别清 `[from, 末尾)` 会误伤期间落位的其它 response

回归护栏：`tests/test_bargein_ref.py`。

### ⚠️⚠️ 声学延迟 D 必须准到 ±5ms（算法 AEC 的生死线）

实测（`tests/test_aec_live_fidelity.py`，真实 AEC 服务）：

| 补偿 D | 回声抑制 |
|---|---|
| 84ms（真值） | **12.6 dB** |
| 90ms | 2.5 dB |
| 250ms（旧默认值） | **0.3 dB**（等于完全不工作） |

**容忍窗只有约 ±5ms** —— 差 6ms 就从 12.6dB 掉到 2.5dB。所以：

- 服务端**不做**时延对齐（`SD_AEC` 是硬编码的 ONNX 模型；仓库里的
  `GCCPHATDelayEstimator`/`LinearAEC` 在流式路径中是死代码，唯一"对齐"
  是 alpha predictor 里 k=10 帧 ≈100ms 的学习式 lookback）。
  **所以调用方必须自己保证样本级预对齐** —— 这正是 `RefTrack` 的职责。

#### D 现在是**每台设备一个固定常量**，离线测一次

落位时刻由浏览器承诺之后（见上文），D 里**只剩**「扬声器→麦克风的物理
延迟 + 设备音频 I/O 缓冲」这一小块：网络往返、浏览器主线程抖动、mic 在途
积压都不再计入。既然是个设备常量，就没有理由在运行时去猜。

**运行时自适应已废弃**（`AcousticDelayTracker` 从实时路径撤下，搬到
`orchestrator/tools/delay_estimate.py`）。原因是它在真机上收敛不了，且
失败模式很糟：回声弱时 GCC-PHAT 找不到真峰，偶尔噪声凑出一个假峰就被
**立刻采纳**，D 从此钉死在错值上、回声再也消不掉 —— 日志实录
「声学延迟自适应: 4000 → 1 采样（0ms）」，此后表现为"长回复好好地说着，
突然就无法打断、开始识别自己说的话了"。

测法：

```bash
# ① 开着四路转储跑一轮真实会话（算法服务 AEC、外放、别戴耳机）
ORCH_DUMP_AUDIO=/tmp/orchdump/s python -m orchestrator.main --port 8100

# ② 用 dump 算 D，并让真实 AEC 在 D±10ms 上验证它
python -m orchestrator.tests.measure_delay \
    --mic /tmp/orchdump/s-xxxx-mic.wav \
    --raw /tmp/orchdump/s-xxxx-raw.wav \
    --verify-url ws://192.168.88.253:30255/ws/asr_frontend

# ③ 把打印出来的 ORCH_AEC_DEFAULT_DELAY_MS 写进服务端环境变量
```

⚠️ 必须用 `-raw.wav`（**未补偿**的原始参考轨）。`-ref.wav` 是喂给 AEC 的
那一份（已按 D 补偿），拿它互相关只能得到**残差**。
⚠️ dump 必须是本次改动之后重新采的 —— 改动前落位误差是变化的，单个 D
吸收不了，脚本会报出很大的散度（这本身就是有用的诊断）。

前端会显示 D 的来源（已实测 / 默认值）。**显示"默认值（未实测）"时算法
AEC 基本不会生效**，别把它当成一个正常数字。

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

会话结束时写四个 wav（默认关闭，实时路径零开销）：

| 文件 | 内容 | 用它能判断 |
|---|---|---|
| `<前缀>-<sid>-mic.wav` | 浏览器送来的**原始麦克风**（AEC 之前） | 里面有没有回声 |
| `<前缀>-<sid>-ref.wav` | 我们算的 **farend**（以为在播什么，**已按 D 补偿**） | 与实际播报是否一致、长度对不对 |
| `<前缀>-<sid>-raw.wav` | **未补偿**的原始参考轨 | 离线测 D（`tests/measure_delay.py` 用它） |
| `<前缀>-<sid>-aec.wav` | **AEC 输出**（ASR 听的就是这个） | 回声消掉多少 |

波形一比即可定论。真机故障就是这样定位的：`ref.wav` 只有 0.6s 有内容而
TTS 报了 5.85s → 直接指向 `truncate` bug（见上文）。

⚠️ 测 D 必须用 `-raw.wav`：`-ref.wav` 已按当时的 D 补偿过，拿它互相关只能
得到残差，不是绝对 D。

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
python -m orchestrator.tests.test_clock         # 含 ctx↔会话采样 锚点映射
python -m orchestrator.tests.test_ref_track     # 含 D 恢复 + 量纲回归护栏
python -m orchestrator.tests.test_bargein_ref   # 打断 + armed 承诺落位
python -m orchestrator.tests.check_html         # 前端结构 + 锚点协议发送端

# ⭐⭐ 全双工仿真：**证明**参考轨对齐了（本轮修复的核心验证）
#    armed 应当逐句恒定、legacy 应当抖到几百毫秒 —— 对照着看才说明问题
python -m orchestrator.tests.test_duplex_sim --scenario turn --anchor armed  --mock-tts --no-asr
python -m orchestrator.tests.test_duplex_sim --scenario turn --anchor legacy --mock-tts --no-asr
python -m orchestrator.tests.test_duplex_sim --scenario barge --anchor armed --mock-tts --no-asr
# 接真服务（验证 ASR 不混入 TTS 内容，需真实语音）
python -m orchestrator.tests.test_duplex_sim --scenario turn --anchor armed \
  --wav-a assets/ref_audio/ref_minicpm_signature.wav \
  --wav-b assets/ref_audio/ref_en_dlc_1.wav --d-true-ms 84

# 离线测 D（用 ORCH_DUMP_AUDIO 的 dump；--verify-url 会用真实 AEC 复核）
python -m orchestrator.tests.measure_delay --mic <-mic.wav> --raw <-raw.wav>

# 主动校准通道（**已从验证页移除**，服务端端点保留给离线工具/诊断用）
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
  actions/executor.py  执行 action（TTS 合成、播放、取消、armed 承诺落位）
  tools/
    delay_estimate.py  声学延迟估计（GCC-PHAT）—— **离线**，实时路径已不用
  tests/               验证脚本与实测记录
    test_duplex_sim.py 全双工仿真（假浏览器 + 合成回声；armed vs legacy 对照）
    measure_delay.py   离线测 D（用 ORCH_DUMP_AUDIO 的 mic/raw dump）
```
