# Omni-LLM-Server — Qwen3-Omni 全双工流式对话服务

基于 llama.cpp-omni 的 C++ 后端，提供音视频全双工流式对话服务。支持：
- **Qwen3-Omni**（主后端，turn-based VAD+TurnSense 分句回复）
- **MiniCPM-o 4.5**（可选，free-duplex 模型自主判决）

服务 = **gateway**（网页入口 :8006）+ **worker-backend**（模型推理 :22400/:22500），
镜像构建时自动从 git 拉取 llama.cpp-omni 源码、从 SVN 下载模型到 `checkpoints/`。

---

## 一、模型目录约定

模型统一放项目根 `checkpoints/` 下（三子目录，可从 SVN 下载或用软链指向已有模型）：

```
checkpoints/
├── qwen3omni-gguf/            # Qwen3-Omni 主模型
│   ├── Qwen3-Omni-30B-A3B-Instruct-Q4_K_S.gguf
│   └── mmproj-Qwen3-Omni-30B-A3B-Instruct-Q8_0.gguf
├── fsmn-vad-onnx/             # FSMN-VAD（语音活动检测）
│   └── model_quant.onnx       # 8bit 量化模型
└── TurnSense/                 # TurnSense（语义完整性判定）
    └── pretrained_models/v1.0/
        ├── model_int8.onnx    # 8bit 量化模型（默认）
        └── am.mvn
```

从 SVN 下载（`svn://svn-local.xmov.ai/repository/AlgModels/OmniLLM/latest`）：
```bash
svn export svn://svn-local.xmov.ai/repository/AlgModels/OmniLLM/latest ./checkpoints
```

> `checkpoints/` 已加入 `.gitignore`（不进 git）；CI 构建时自动从 SVN export。

---

## 二、服务端部署（Docker Compose）

### 前置
- Docker + Compose v2、[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- 一张 ≥24GB 显存的 NVIDIA GPU（Qwen3-Omni 30B-A3B 量化）
- 模型已放好（见上，`checkpoints/`）

### 1. 生成证书 + 配置
```bash
mkdir -p certs data
openssl req -x509 -newkey rsa:2048 -nodes -days 365 \
  -keyout certs/key.pem -out certs/cert.pem -subj "/CN=minicpm-o"
cp .env.example .env   # 或手改 .env
```

`.env` 关键项：
```bash
ACTIVE_MODEL=qwen3omni
MODEL_HOST_PATH=/path/to/your/models   # 默认 ./checkpoints（项目内三子目录）
TURNSENSE_HOST_PATH=./checkpoints/TurnSense
MTMD_BACKEND_DEVICE=CUDA1              # mmproj 放 GPU1，平衡双卡显存
CPP_GPU_ID=0                           # GPU 编号（多卡逗号分隔）
GATEWAY_HOST_PORT=8006
```

### 2. 构建并启动
```bash
# 构建镜像（llama.cpp-omni 自动 git clone duj12/dev；模型从 ./checkpoints 打进镜像）
docker compose -f docker-compose.cpp.yml up -d --build
# 只构建
docker compose -f docker-compose.cpp.yml build cpp-worker-backend
```

> 模型默认从项目内 `./checkpoints` 挂载到容器 `/models`。可用 `MODEL_HOST_PATH` 覆盖
> 到别的目录（如 SVN 下载目录）。镜像内 `COPY . .` 也会带上 checkpoints（CI 构建时）。

### 3. 健康检查
```bash
docker compose -f docker-compose.cpp.yml ps            # 等 healthy（模型加载 ~1-2min）
curl -sk https://127.0.0.1:8006/                       # gateway
docker logs -f omni-llm-cpp-backend             # worker 日志（VAD/TurnSense 就绪）
```

### 4. 停止
```bash
docker compose -f docker-compose.cpp.yml down
```

---

## 三、Gateway 网页客户端

服务起来后浏览器打开 **`https://<host>:8006/`**（内网机器用 `https://192.168.89.105:8006/`）。
证书是自签的，浏览器需信任/忽略告警。首页按模式选择：

| 页面 | 路径 | 说明 |
|------|------|------|
| 首页 | `/` | 模式选择入口 |
| Turn-based Chat | `/turnbased.html` | 按钮触发回复；支持离线音视频上传理解，低延迟 |
| Omni 全双工 | `/omni/omni.html` | 实时音视频全双工，模型自主发言 |
| Audio 全双工 | `/audio-duplex/audio_duplex.html` | 纯音频全双工对话 |
| Half-duplex (VAD+TurnSense) | `/half-duplex/half_duplex.html` | VAD 检测停顿 + TurnSense 语义完整才回复 |
| 管理 | `/admin.html` | Worker/会话状态 |

> 音视频实时采集页需要麦克风/摄像头，浏览器只在 **https 或 localhost** 下提供
> `navigator.mediaDevices` —— 所以 gateway 默认 HTTPS。

---

## 四、命令行客户端 `streaming_chat_demo.py`

独立 Python 客户端，直连 gateway 的 `/v1/realtime?mode=video` WebSocket，做文件回放 /
并发压力测试。后端类型（qwen3omni/minicpm）由服务端 `session.created` 的 `active_model`
自动判定，`turn_decision` 按后端推导，无需手动指定。

### 单路回放（视频理解）
```bash
python streaming_chat_demo.py --video assets/video/turnbased/121.mp4 \
    --prompt "你是一个多模态助手，请简练回复用户的问题。" \
    --host 192.168.89.105 --port 8006
```

### 纯音频交互（只给音频文件）
```bash
python streaming_chat_demo.py --audio assets/audio/xxx.wav
```

### 并发压力测试（N 路）
```bash
python streaming_chat_demo.py --video assets/video/turnbased/121.mp4 \
    --concurrency 10 --gpu-ids 0,1 --host 192.168.89.105 --port 8006
```

### 实时采集（麦克风 + 摄像头）
```bash
pip install sounddevice
python streaming_chat_demo.py --realtime
```

### 常用参数
| 参数 | 默认 | 说明 |
|------|------|------|
| `--host` / `--port` | `192.168.89.106` / `8006` | gateway 地址（生产入口） |
| `--concurrency N` | `1` | 并发路数，`>1` 进入并发测试并监控 GPU |
| `--gpu-ids` | `0,1` | 并发时监控的 GPU |
| `--video` | 空 | 视频路径（回放模式） |
| `--audio` | 空 | 音频路径（纯音频交互 / 替换视频音轨） |
| `--audio-chunk-ms` | `1000` | 音频块大小（VAD 需 ≥25ms） |
| `--max-audio-s` | 全 | 限制音频时长（长音频约 25 tok/s 注意 KV） |
| `--kv-budget` | `20000` | 单分句视频帧 token 预算 |
| `--tail-silence-s` | `2.0` | 收尾补发静音（VAD 闭合最后一段） |
| `--drain-idle-s` | `5.0` | 收尾空闲兜底（正常靠 response_id 判定不等超时） |
| `--show-reply-text` | 关 | 并发结果打印每轮完整回复文本 |
| `--realtime` | — | 实时采集模式 |

### 输出指标（每轮）
```
turn#1: TTFT=0.23s in_audio=9.0s in_video=9.0s reply=1.2s (120ch, 240ch/s)
```
- **TTFT**：首字延迟（服务端 `turn.turnsense=complete` → 首个文本 token，用服务端时间戳）
- **in_audio_s / in_video_s**：本轮输入音视频时长
- **reply_s**：纯生成耗时（首字 → 本轮结束）
- **speed_cps**：生成速度（字符/秒）

并发汇总输出每路每轮（`--show-reply-text` 显示文本）+ 全体 P50/P90/P99。

---

## 五、CI（GitLab）

`.gitlab-ci.yml` 构建并推送两个镜像：
- **`omnillm-cpp-backend`**（worker + 后端：C++ llama 服务 + worker.py + VAD/TurnSense）
- **`omnillm-gateway`**（gateway 网页入口，不加载模型）

流程：
1. `svn export svn://svn-local.xmov.ai/repository/AlgModels/OmniLLM/latest ./checkpoints`（按分支映射）
2. `docker compose -f docker-compose.cpp.yml build cpp-worker-backend`（llama.cpp-omni git clone + 模型进镜像）
3. `docker compose -f docker-compose.cpp.yml build gateway`
4. push 两个镜像并打 tag

---

## 六、目录结构（关键）

```
Omni-LLM-Server/
├── checkpoints/              # 模型权重（gitignored，CI/SVN 生成）
├── docker-compose.cpp.yml    # C++ backend 部署（qwen3omni 推荐）
├── docker/Dockerfile.cpp-worker-backend  # 镜像（llama.cpp-omni git clone）
├── docker/entrypoint-cpp-worker-backend.sh
├── gateway.py                # gateway（网页入口 :8006）
├── worker.py                 # worker（转发 :22400）
├── py_backend/               # PyTorch backend（可选）
├── runtime/
│   ├── half_duplex.py        # VAD+TurnSense 分句状态机
│   ├── turnsense/            # TurnSense 运行代码（vendor）
│   └── fsmn_vad_onnx/        # FSMN-VAD 运行代码
├── streaming_chat_demo.py    # 命令行客户端
└── static/                   # 网页前端
```
