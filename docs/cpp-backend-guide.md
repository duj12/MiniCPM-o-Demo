# C++ 后端（llama.cpp-omni）使用指南

本文档面向使用 llama.cpp-omni C++ 推理后端的开发者和使用者，涵盖服务部署、运维和 API 调用。

---

## 目录

- [服务架构](#服务架构)
- [前置条件](#前置条件)
- [快速启动](#快速启动)
- [环境变量说明](#环境变量说明)
- [常用运维命令](#常用运维命令)
- [API 调用指南](#api-调用指南)
  - [轮次对话（Turn-based）](#轮次对话turn-based)
  - [全双工（Full-duplex）](#全双工full-duplex)
  - [音频输出与音色克隆](#音频输出与音色克隆)
  - [实时流 vs 文件回放](#实时流-vs-文件回放)
    - [方式一：实时采集（浏览器/Python）](#方式一实时采集浏览器python-客户端)
    - [方式二：从视频文件提取帧（测试用）](#方式二从视频文件提取帧测试用)
  - [协议说明](#协议说明)
- [测试脚本](#测试脚本)
- [常见问题](#常见问题)

---

## 服务架构

```
Client (浏览器/Python/curl)
    │  WSS (:8006)
Gateway — HTTPS 入口、路由、排队
    │  WS (internal)
Worker — 协议转发、状态管理
    │  WS (internal)
Backend (llama-omni-server) — 模型推理
```

## 前置条件

- Docker + Compose v2 插件
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- 模型权重（GGUF 格式），目录结构如下：

```
MiniCPM-o-4_5-gguf/
├── MiniCPM-o-4_5-Q4_K_M.gguf       # 主模型（推荐 4-bit）
├── vision/MiniCPM-o-4_5-vision-F16.gguf
├── audio/MiniCPM-o-4_5-audio-F16.gguf
├── tts/MiniCPM-o-4_5-tts-F16.gguf
├── tts/MiniCPM-o-4_5-projector-F16.gguf
└── token2wav-gguf/
```

## 快速启动

所有操作在 `MiniCPM-o-Demo/` 目录下执行。

### 1. 生成 SSL 证书（首次）

```bash
mkdir -p certs
openssl req -x509 -newkey rsa:2048 -nodes -days 365 \
  -keyout certs/key.pem -out certs/cert.pem -subj "/CN=minicpm-o"
```

### 2. 配置 .env

```bash
# 将模型路径改为你的实际路径
cat > .env << 'EOF'
GGUF_MODEL_HOST_PATH=/data/models/MiniCPM-o-4_5-gguf
GGUF_MODEL_FILE=MiniCPM-o-4_5-Q4_K_M.gguf
CPP_GPU_ID=0
GATEWAY_HOST_PORT=8006
LLAMA_SERVER_EXTRA_ARGS=-c 32768
EOF
```

### 3. 构建并启动

```bash
docker compose -f docker-compose.cpp.yml up -d --build
```

### 4. 确认就绪

```bash
# 查看后端日志，看到 "worker ready" 即启动完成
docker compose -f docker-compose.cpp.yml logs --tail=5 cpp-worker-backend

# 健康检查
curl -k https://127.0.0.1:8006/status
```

首次构建约 10-15 分钟（编译 llama.cpp），模型加载约 1-2 分钟。

### 5. 停止

```bash
docker compose -f docker-compose.cpp.yml down
```

---

## 环境变量说明

`MiniCPM-o-Demo/.env` 文件配置：

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `GGUF_MODEL_HOST_PATH` | **必填** 模型权重目录的宿主机路径 | — |
| `GGUF_MODEL_FILE` | 主模型 GGUF 文件名 | `MiniCPM-o-4_5-Q4_K_M.gguf` |
| `CPP_GPU_ID` | 使用的 GPU 编号 | `0` |
| `GATEWAY_HOST_PORT` | 网关对外端口（HTTPS） | `8006` |
| `LLAMA_SERVER_EXTRA_ARGS` | 后端额外启动参数 | `-c 8192` |

> **`LLAMA_SERVER_EXTRA_ARGS` 建议**：`-c 32768`。增大 KV cache 可避免 `failed to find a memory slot` 错误，尤其多路并发时更稳定。也支持其他 llama-server 参数（如 `--no-cache-prompt`）。

---

## 常用运维命令

```bash
# ── 日志 ──
docker compose -f docker-compose.cpp.yml logs -f gateway             # 网关实时日志
docker compose -f docker-compose.cpp.yml logs -f cpp-worker-backend  # 后端实时日志
docker compose -f docker-compose.cpp.yml logs --tail=50 cpp-worker-backend  # 最近50行

# ── 容器管理 ──
docker compose -f docker-compose.cpp.yml up -d --no-build cpp-worker-backend  # 重启后端（不重构建）
docker compose -f docker-compose.cpp.yml restart cpp-worker-backend           # 重启

# ── 构建 ──
docker compose -f docker-compose.cpp.yml build cpp-worker-backend    # 重新构建镜像
docker compose -f docker-compose.cpp.yml up -d --build               # 构建+启动

# ── 健康检查 ──
curl -k https://127.0.0.1:$GATEWAY_PORT/status

# ── 直接检查后端 ──
docker exec minicpm-o-demo-cpp-worker-1 curl -sf http://127.0.0.1:22500/v1/health
```

---

## API 调用指南

服务启动后，通过 `wss://<host>:<port>/v1/realtime`（默认 `wss://127.0.0.1:8006/v1/realtime`）提供 WebSocket API。

### 轮次对话（Turn-based）

适用于文本问答、图片/视频理解等一次性交互。输入一次，等待完整回复。

**Python 示例：**

```python
import asyncio, json, ssl, base64
import websockets

GATEWAY = "wss://192.168.89.106:8006/v1/realtime"

async def turn_based_example():
    ssl_ctx = ssl.create_default_context()
    ssl_ctx.check_hostname = False
    ssl_ctx.verify_mode = ssl.CERT_NONE

    # 读取视频（也可传纯文本或图片）
    with open("video.mp4", "rb") as f:
        video_b64 = base64.b64encode(f.read()).decode()

    async with websockets.connect(
        GATEWAY, max_size=128*1024*1024, ssl=ssl_ctx
    ) as ws:
        # 1. 等待 queue_done → 发送 session.init
        await asyncio.wait_for(ws.recv(), timeout=10)
        await ws.send(json.dumps({
            "type": "session.init",
            "payload": {"mode": "turn_based"},
        }))

        # 2. 接收 session.created
        msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=10))
        session_id = msg.get("session_id")
        print("Session:", session_id)

        # 3. 发送输入
        await ws.send(json.dumps({
            "type": "input.append",
            "input": {
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "描述这个视频"},
                        {"type": "video", "data": video_b64,
                         "name": "test.mp4", "duration": 10},
                    ],
                }],
                "streaming": False,        # 非流式模式
                "tts": {"enabled": False}, # 关闭语音合成
                "use_tts_template": False,
                "omni_mode": True,
                "image": {"max_slice_nums": 1},
            },
        }, ensure_ascii=False))

        # 4. 接收响应
        text = ""
        while True:
            msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=180))
            t = msg.get("type")
            if t == "response.output.delta" and msg.get("kind") == "text":
                text += msg.get("text", "")
            elif t == "response.done":
                # 非流式模式文本在 response.done.text 中
                text = msg.get("text", "") or text
                break
            elif t in ("session.closed", "error"):
                break

        # 5. 关闭会话
        await ws.send(json.dumps({
            "type": "session.close", "reason": "done",
        }))
        print("回复:", text)

asyncio.run(turn_based_example())
```

> **注意**：`streaming=False` 时文本在 `response.done.text` 字段返回，`response.output.delta` **不会**有文本内容。如需流式逐字输出，设 `"streaming": True`。

### 全双工（Full-duplex）

适用于实时语音对话、视频通话。音频按 100ms 帧切分，逐帧发送，模型自主决定说话时机。

**Python 示例：**

```python
import asyncio, json, ssl, base64
import websockets

GATEWAY = "wss://192.168.89.106:8006/v1/realtime"
FRAME_BYTES = 3200  # 100ms, 16kHz, 16-bit PCM

async def full_duplex_example():
    ssl_ctx = ssl.create_default_context()
    ssl_ctx.check_hostname = False
    ssl_ctx.verify_mode = ssl.CERT_NONE

    async with websockets.connect(
        GATEWAY, max_size=128*1024*1024, ssl=ssl_ctx
    ) as ws:
        # 握手
        await asyncio.wait_for(ws.recv(), timeout=10)
        await ws.send(json.dumps({
            "type": "session.init",
            "payload": {"mode": "full_duplex"},
        }))
        while True:
            msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=30))
            if msg.get("type") == "session.created":
                break
        print("Session:", msg.get("session_id"))

        # 逐帧发送音频（首帧可附带 video_frames/text）
        for i in range(20):  # 发送 2 秒音频
            # 实际场景：从麦克风采集音频帧
            audio_chunk = b'\x00' * FRAME_BYTES  # 静音帧
            audio_b64 = base64.b64encode(audio_chunk).decode()

            payload = {
                "type": "input.append",
                "input": {
                    "audio": audio_b64,
                    "max_slice_nums": 1,
                },
            }
            # 首帧可附带信息
            if i == 0:
                payload["input"]["text"] = "你好，请介绍一下你自己"
                # 也可附带视频帧（从视频文件提取 JPEG）:
                # payload["input"]["video_frames"] = [frame_jpeg_b64]

            await ws.send(json.dumps(payload, ensure_ascii=False))

            # 接收模型响应
            try:
                resp = json.loads(
                    await asyncio.wait_for(ws.recv(), timeout=10)
                )
                kind = resp.get("kind", "")
                if kind == "text":
                    print("模型:", resp.get("text", ""))
                elif kind == "listen":
                    pass  # 模型切到听状态
                elif kind == "audio":
                    print("→ 收到音频块")
            except asyncio.TimeoutError:
                pass

            await asyncio.sleep(0.1)  # 实时节奏

        # 关闭
        await ws.send(json.dumps({
            "type": "session.close", "reason": "done",
        }))
```


### 实时流 vs 文件回放

全双工模式支持两种输入方式，**协议完全相同**，只是数据来源不同：

#### 方式一：实时采集（浏览器/Python 客户端）

音频和视频分别独立采集，各自以固定帧率发送到同一个 WS 连接：

```
音频: 麦克风 → 100ms PCM 帧 → base64 → input.append(audio=...)
视频: 摄像头 → canvas JPEG 帧 → base64 → input.append(video_frames=[...])
```

**浏览器前端（已有实现，无需修改）：**

```javascript
// audio: getUserMedia → AudioWorklet(100ms chunks) → base64
// video: getUserMedia → video element → canvas.toDataURL('image/jpeg') → base64
media.onChunk = (chunk) => {
    const msg = {
        type: 'audio_chunk',
        audio_base64: arrayBufferToBase64(chunk.audio.buffer),  // 100ms PCM
    };
    if (chunk.frameBase64) {
        msg.frame_base64_list = [chunk.frameBase64];            // JPEG 帧
    }
    session.sendChunk(msg);
};
```

**Python 客户端（需要额外安装）：**

```bash
pip install sounddevice opencv-python
```

```python
import asyncio, json, ssl, base64, cv2, sounddevice as sd
import numpy as np
import websockets

SAMPLE_RATE = 16000
FRAME_SAMPLES = 1600  # 100ms

async def realtime_capture():
    ctx = ssl.create_default_context()
    ctx.check_hostname = False; ctx.verify_mode = ssl.CERT_NONE

    async with websockets.connect(
        "wss://192.168.89.106:8006/v1/realtime",
        max_size=128*1024*1024, ssl=ctx
    ) as ws:
        # 握手
        await asyncio.wait_for(ws.recv(), 10)
        await ws.send(json.dumps({
            "type": "session.init",
            "payload": {"mode": "full_duplex"},
        }))
        m = json.loads(await asyncio.wait_for(ws.recv(), 10))
        print("Session:", m.get("session_id"))

        # 打开摄像头
        cap = cv2.VideoCapture(0)
        first_frame = True

        # 音频回调（sounddevice 在独立线程调用）
        def audio_callback(indata, frames, time, status):
            nonlocal first_frame
            audio_b64 = base64.b64encode(indata.tobytes()).decode()

            payload = {
                "type": "input.append",
                "input": {"audio": audio_b64, "max_slice_nums": 1},
            }

            # 首帧附带视频帧
            if first_frame:
                ret, frame = cap.read()
                if ret:
                    _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                    payload["input"]["video_frames"] = [
                        base64.b64encode(jpeg.tobytes()).decode()
                    ]
                first_frame = False

            # 直接发送（需在 asyncio 线程中执行）
            asyncio.run_coroutine_threadsafe(
                ws.send(json.dumps(payload, ensure_ascii=False)),
                loop
            )

        # 启动音频流
        loop = asyncio.get_event_loop()
        with sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1,
            blocksize=FRAME_SAMPLES, callback=audio_callback,
        ):
            await asyncio.sleep(30)  # 采集 30 秒

        cap.release()
        await ws.send(json.dumps({"type": "session.close", "reason": "done"}))
```

#### 方式二：从视频文件提取帧（测试用）

测试时无法使用真实的麦克风/摄像头，因此需要用 ffmpeg 从视频文件中提取音频 PCM 和视频 JPEG 帧，模拟实时流。

```python
import subprocess, base64

def extract_audio_pcm(video_path):
    """提取音频为 16kHz 16-bit PCM 原始数据"""
    result = subprocess.run([
        "ffmpeg", "-y", "-i", video_path, "-vn",
        "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        "-f", "s16le", "pipe:1",
    ], capture_output=True, check=True)
    return result.stdout

def extract_keyframes(video_path, max_n=3):
    """从视频中均匀提取 N 帧 JPEG，返回 base64 列表"""
    probe = subprocess.run(["ffprobe", "-v", "error",
        "-select_streams", "v:0", "-show_entries", "stream=duration",
        "-of", "csv=p=0", video_path], capture_output=True, text=True)
    duration = float(probe.stdout.strip() or 30)
    frames = []
    for i in range(max_n):
        pos = duration * (i + 1) / (max_n + 1)
        r = subprocess.run(["ffmpeg", "-y", "-ss", str(pos),
            "-i", video_path, "-vframes", "1", "-q:v", "5",
            "-f", "mjpeg", "pipe:1"], capture_output=True, check=True)
        frames.append(base64.b64encode(r.stdout).decode())
    return frames

def split_audio_frames(pcm_data, frame_ms=100):
    """将 PCM 切成 100ms 帧（16kHz=3200 bytes/帧）"""
    fb = int(16000 * frame_ms / 1000) * 2
    return [pcm_data[i:i+fb].ljust(fb, b'\x00')
            for i in range(0, len(pcm_data), fb)]

# 使用示例
pcm = extract_audio_pcm("video.mp4")
chunks = split_audio_frames(pcm)          # 音频帧列表
frames = extract_keyframes("video.mp4")   # JPEG base64 列表

for i, chunk in enumerate(chunks[:100]):  # 发 10 秒
    audio_b64 = base64.b64encode(chunk).decode()
    payload = {"type": "input.append", "input": {
        "audio": audio_b64, "max_slice_nums": 1,
    }}
    if i == 0 and frames:
        payload["input"]["video_frames"] = frames  # 首帧携带视频
    if i == 0:
        payload["input"]["text"] = "请描述这个视频"
    await ws.send(json.dumps(payload, ensure_ascii=False))
    # ... 接收响应 ...
    await asyncio.sleep(0.15)
```

### 协议说明

#### 消息时序（Turn-based）

```
Client → Server: session.init      {"type": "session.init", "payload": {"mode": "turn_based"}}
Server → Client: session.created   {"type": "session.created", "session_id": "..."}
Client → Server: input.append      {"type": "input.append", "input": {...}}
Server → Client: output.delta      {"type": "response.output.delta", "kind": "text", "text": "..."}
Server → Client: response.done     {"type": "response.done", "text": "完整文本"}
Client → Server: session.close     {"type": "session.close", "reason": "done"}
```

#### 消息时序（Full-duplex，每帧循环）

```
Client → Server: session.init      {"type": "session.init", "payload": {"mode": "full_duplex"}}
Server → Client: session.created   {"type": "session.created", "session_id": "..."}
Client → Server: input.append      {"type": "input.append", "input": {"audio": "...", ...}}
Server → Client: output.delta      {"type": "response.output.delta", "kind": "listen"}  ← 模型在听
                                   或 {"kind": "text", "text": "..."}                      ← 模型在说
                                   或 {"kind": "audio", "data": "..."}                     ← 音频输出
```

### 音频输出与音色克隆

#### 控制音频输出

在 `session.init` 中设置 `use_tts` 控制会话级别的 TTS 开关（默认 `true`）：

```json
{
  "type": "session.init",
  "payload": {
    "mode": "turn_based",
    "use_tts": false       // 关闭所有音频输出
  }
}
```

在 `input.append` 中通过 `tts.enabled` 或 `use_tts_template` 控制单次请求的音频输出：

```json
{
  "type": "input.append",
  "input": {
    "messages": [...],
    "use_tts_template": true,          // 开启 TTS 音频输出
    "tts": {"enabled": true}           // 同上，另一种写法
  }
}
```

> `use_tts_template` 与 `tts.enabled` 任设其一即开启。两者都关闭则只输出文本。

音频输出在 `response.output.delta` 中通过 `kind: "audio"` 返回 base64 PCM 数据（float32），`response.done` 中也可能包含完整音频：

```json
// 流式音频块
{"type": "response.output.delta", "kind": "audio", "data": "base64..."}
// 非流式完整音频（在 response.done 中）
{"type": "response.done", "text": "...", "audio": "base64..."}
```

#### 音色克隆（Voice Cloning）

通过设置参考音频，可以让模型模仿特定音色说话。参考音频是 **base64 编码的 WAV 文件**（16kHz, 16-bit, 单声道，3-10 秒）。

在 `session.init` 中设置会话级别的音色：

```python
import base64

with open("ref_audio.wav", "rb") as f:
    ref_audio_b64 = base64.b64encode(f.read()).decode()

payload = {
    "type": "session.init",
    "payload": {
        "mode": "turn_based",
        "voice": {
            "ref_audio": ref_audio_b64,
        },
    },
}
```

也可在 `input.append` 中临时覆盖音色：

```python
payload = {
    "type": "input.append",
    "input": {
        "messages": [...],
        "use_tts_template": True,
        "tts": {
            "enabled": True,
            "ref_audio_data": ref_audio_b64,
        },
    },
}
```

> 注：`session.init` 中还有一个 `voice.tts_ref_audio` 字段，但当前后端实现中它和 `ref_audio` 等价，直接用 `ref_audio` 即可。

#### input.append 参数

| 字段 | 类型 | 说明 |
|------|------|------|
| `messages` | array | Turn-based 消息列表（全双工无需此字段） |
| `audio` | string | 音频 base64（16kHz, 16-bit PCM WAV，全双工必需） |
| `video_frames` | array | JPEG 帧 base64 列表（首帧附带） |
| `text` | string | 文本输入（全双工模式可用） |
| `streaming` | bool | 是否流式输出（默认 false） |
| `use_tts_template` | bool | 启用 TTS 音频输出 |
| `tts` | object | TTS 配置 `{"enabled": true, "ref_audio_data": "base64..."}` |
| `omni_mode` | bool | 是否开启 Omni 多模态模式 |
| `enable_thinking` | bool | 启用思考链输出 |
| `force_listen` | bool | 全双工模式强制当前帧切到听状态 |
| `image` | object | 图像参数，`{"max_slice_nums": 1}` |
| `generation` | object | 生成参数，`{"max_new_tokens": 512, "length_penalty": 1.1}` |

---

## 测试脚本

项目 `tests/` 目录下提供了一系列测试脚本：

| 脚本 | 测试内容 | 说明 |
|------|----------|------|
| `test_large_text.py` | 超大文本（20MB） | 测试 WS 消息体上限 |
| `test_video_inference.py` | 基本视频推理 | 命令行指定视频路径 |
| `test_video_concurrent.py` | 并发视频推理 | 多个视频同时进行 turn_based 推理 |
| `test_video_turnbased.py` | 单轮视频理解 | 测试 `assets/video/turnbased/` 目录下视频 |
| `test_video_fullduplex.py` | 全双工流式视频 | 模拟逐帧发送音频+视频帧 |
| `test_multi_session_concurrency.py` | 多路并发 | `--simplex N --duplex M` 参数控制路数 |
| `test_vram_concurrency.py` | 并发+显存测试 | 文本/视频/全双工 三种模式，自动追踪 VRAM 峰值 |

**运行方式：**

```bash
# 在远程服务器上通过 conda 环境运行
/home/dujing/miniconda3/envs/py310/bin/python tests/test_video_turnbased.py

# 或激活环境后运行
export PATH=/home/dujing/miniconda3/envs/py310/bin:$PATH
python tests/test_video_turnbased.py
```

---

## 常见问题

### Q: 启动报 `GGUF_MODEL_HOST_PATH is missing`？

A: `.env` 文件缺失或 `GGUF_MODEL_HOST_PATH` 配置不正确。确保从 `MiniCPM-o-Demo/` 目录执行命令，且 `.env` 文件存在于该目录。

### Q: `failed to find a memory slot for batch of size N`？

A: KV cache 不足。在 `.env` 中增大 `-c`：
```
LLAMA_SERVER_EXTRA_ARGS=-c 32768
```
然后 `docker compose -f docker-compose.cpp.yml up -d --no-build cpp-worker-backend`。

此错误的原因是 `n_seq_max=4`（默认 4 路并发）时每路分得 `n_ctx / 4` 个 cell。`-c 32768` 下每路分得 8192 个 cell，足够大部分对话场景。

### Q: 全双工无响应，客户端一直超时？

A: 检查服务是否包含 2026-07-16 之后的修复。旧版本中 `force_listen` 产生的 `__IS_LISTEN__` 事件被 `text_done_flag` 检查提前丢弃，导致全双工模式下客户端收不到任何响应。

### Q: 单轮对话返回空文本？

A: 非流式模式（`streaming: false`）的文本在 `response.done.text` 字段中，不在 `response.output.delta`。请参考上方 API 调用代码。

### Q: 如何估算并发能力和显存占用？

A: 与模型量化精度和 KV cache 大小密切相关。以下是在 RTX 3090（24GB）上的实测数据，测试工具为 `tests/test_vram_concurrency.py`。

**测试环境：**
- 模型：`MiniCPM-o-4_5-Q4_K_M.gguf`（4-bit）
- 配置：`-c 32768 --parallel 4`（每路 LLM KV cache = 8192）
- TTS KV cache：512（独立限制，不占 LLM 的 KV cache）
- Token2Wav：共享模式（仅首路初始化，后续复用）
- GPU 0 空闲显存：~9,161 MiB（含 TTS 模型 + 其他服务占用）

**纯文本并发测试（"Say hello in 3 words"）：**

| 并发路数 | 通过 | wall time |
|:-------:|:---:|:---------:|
| 1 | ✅ | 1.1s |
| 2 | ✅ | 1.3s |
| 3 | ✅ | 1.5s |
| 4 | ✅ | 2.4s |

**单轮视频并发测试（121.mp4, 9.6MB, turn_based）：**

| 并发路数 | 通过 | wall time |
|:-------:|:---:|:---------:|
| 1 | ✅ | 7.3s |
| 2 | ✅ | 10.9s |
| 3 | ✅ | 13.5s |
| 4 | ✅ | 17.6s |

**全双工流式视频并发测试（121+pad.mp4, 73MB, 模拟实时采集）：**

模拟真实摄像头+麦克风输入：音频按 100ms 帧发送，视频按 5fps 在对应音频帧时间点附带 JPEG 帧。每路发送 30 帧（3 秒）。

| 并发路数 | 通过 | wall time | VRAM 峰值 | 说明 |
|:-------:|:---:|:---------:|:---------:|------|
| 1 | ✅ | 15.8s | 10,941 MiB | 含 TTS 完整加载 |
| 2 | ✅ | 21.5s | 10,941 MiB | 显存稳定 |
| 3 | ✅ | 29.0s | 10,941 MiB | 显存稳定 |
| **4** | **✅** | **39.3s** | **10,945 MiB** | **4 路全通过** |

**禁用 TTS 的全双工流式视频并发测试（`--no-tts` 启动）：**

跳过 TTS 模型、权重和 Token2Wav 加载，节省约 2.3GB 基线显存。适用于不需要语音输出、只关注文本回复的场景。

| 并发路数 | 通过 | wall time | VRAM 峰值 | 说明 |
|:-------:|:---:|:---------:|:---------:|------|
| 1 | ✅ | 15.4s | 8,371 MiB | 基线 6,863 MiB |
| 2 | ✅ | 21.3s | 8,371 MiB | 含 TTS 节省 2.3GB |
| 3 | ✅ | 28.3s | 8,371 MiB | 显存稳定 |
| **4** | **✅** | **38.1s** | **8,373 MiB** | **4 路全通过** |

**对比总结：**

| 指标 | 有 TTS | 无 TTS（`--no-tts`） |
|:----|:------:|:-------------------:|
| 基线显存 | 9,161 MiB | **6,863 MiB** (-25%) |
| 每路增量 | ~1.8 GB | ~1.5 GB |
| 4 路全双工 | ✅ 稳定通过 | ✅ **4/4 通过** |
| 适用场景 | 需要语音回复 | 仅文本回复 |

**`--no-tts` 启动方式：**

```bash
# .env 中 LLAMA_SERVER_EXTRA_ARGS 添加 --no-tts
# 或命令行启动时直接附加
LLAMA_SERVER_EXTRA_ARGS="-c 32768 --parallel 4 --no-tts" \
docker compose -f docker-compose.cpp.yml up -d --no-build cpp-worker-backend
```

**并发测试命令：**

```bash
# 激活 conda 环境后，在 MiniCPM-o-Demo 目录执行

# 纯文本 4 路
python tests/test_vram_concurrency.py

# 单轮视频 4 路
python tests/test_vram_concurrency.py --mode video

# 全双工视频 1 路（完整 38s）
python tests/test_vram_concurrency.py --mode duplex --video --duplex-frames 0

# 全双工视频 4 路（每路 3s 音视频流，5fps 模拟摄像头）
python tests/test_vram_concurrency.py --mode duplex --video \
    --max-concurrency 4 --duplex-frames 30 --duplex-video-fps 5

# 全双工静音+文本（不依赖视频文件）
python tests/test_vram_concurrency.py --mode duplex --max-concurrency 2
```

**全双工流式测试参数：**

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--duplex-frames` | 每路音频帧数（0=全部） | 10 |
| `--duplex-video-fps` | 模拟摄像头帧率 | 5 |
| `--duplex-gap` | 帧间隔秒数 | 0.15 |
| `--stagger` | 每路启动间隔秒数 | 2.0 |

**结论：**
- `Q4_K_M` + `-c 32768 --parallel 4` 下，单卡 RTX 3090 可稳定支持 **4 路全双工视频并发（含 TTS）**
- 关键优化点：per-session KV cache 按 `n_ctx / n_parallel` 分配、TTS KV cache 限制为 512、Token2Wav GPU session 共享
- 全双工模式和单工模式的显存增量接近（均约 1.8GB/路），因为两者共享相同的 per-session KV cache 减分配策略
- 如需更高并发，可考虑使用 `--no-tts` 启动参数跳过 TTS 模型加载（省约 2-3GB 基线显存），或换用更大显存卡

### Q: 视频太大报 `WebSocket` 连接断开？

A: httplib 默认 WS 最大消息体为 16MB，base64 编码后约 12MB 视频会超出。已在 `ws_handler.cpp` 和 `server-omni.cpp` 开头设置：
```cpp
#define CPPHTTPLIB_WEBSOCKET_MAX_PAYLOAD_LENGTH (128 * 1024 * 1024)
```
若仍有问题可继续增大此值。

### Q: 如何指定不同的 GPU 来运行？

A: 修改 `.env` 中 `CPP_GPU_ID=1` 即可。服务器有多张 GPU 时，每张卡可部署一个独立的 C++ worker 实例。

### Q: 如何查看 GPU 显存占用？

```bash
ssh 服务器地址 nvidia-smi
```
正常运行时显存占用约 6.8GB（无 TTS）或 9.1GB（有 TTS），每增加一路并发额外占用约 1.5-1.8GB。
