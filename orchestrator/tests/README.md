# orchestrator/tests —— 阶段 0 量测脚本与结果

三个外接服务 + OmniLLM 的**实测**记录。这些数字与源码推断有出入，**以实测为准**。

## 脚本

| 脚本 | 测什么 | 在哪里跑 |
|---|---|---|
| `probe_aec.py` | AEC 服务（`/ws/asr_frontend`）长时流、窗口节奏、样本守恒 | 本机或 106（需 `speech_frontend` 同层） |
| `probe_asr.py` | ASR 服务连通性、部分结果延迟、字段契约、turnsense 形状 | 本机或 106 |
| `probe_tts.py` | TTS gRPC 合成、首帧延迟、RTF、CHAR_TIME_MAP | **必须在 106**（需 grpc + `TTS/protos`） |

## 实测结果汇总

### OmniLLM（Qwen3-Omni，gateway :8006）

用 `streaming_chat_demo.py --video assets/video/turnbased/121.mp4` 实跑：

```
turn#1: TTFT=0.26s in_audio=8.0s in_video=8.0s reply=0.2s (45ch, 233ch/s)
turn#2: TTFT=0.79s in_audio=8.0s in_video=8.0s reply=0.6s (131ch, 237ch/s)
平均首字延迟 0.52s  平均速度 235 ch/s  PASS
```

- **跑在 docker 容器里**：`omni-llm-gateway`（8006）+ `omni-llm-cpp-backend`（22400/22500）
- ⚠️ **`ACTIVE_MODEL=qwen3omni` 是容器环境变量，覆盖了 `config.json` 里的 `"active_model": "minicpm"`**
  → 判断生效后端**必须看 `session.created.active_model`**，不能读配置文件
- 这条路验证了「复用 `StreamingChatClient` 当客户端连 gateway」可行

### AEC（`ws://192.168.88.253:30255/ws/asr_frontend`）

⚠️ **重要：`d:\work\code\asr_frontend\` 是独立仓库，与 `speech_frontend` 的
git 子模块版本不同。** 最初读的旧版是 **V1（环形缓冲）**，实际部署的是
**V2（滑窗）**。任何推断必须以子模块最新代码为准。

```
asr_frontend/streamer/
  stream_inference.py      101 行 —— 分发器（按 stream_mode 选 v1/v2）
  stream_inference_v1.py   809 行 —— 旧环形缓冲
  stream_inference_v2.py   464 行 —— **当前默认**
```

**V2 真实语义**（`stream_inference_v2.py:254` `_extract_chunk_output`）：
**每个输入 chunk 对应一个等长输出**（用带上下文的窗口推理，只取当前 chunk
的净输出；`give_up` 补偿重叠）。因此**输入输出一一对应、样本严格守恒**。

**V2 参数**（`:61-65`）：

```python
window_samples   = int(target_sr * decode_window)   # 实测 3200 (=0.2 @16k)
_stride          = int(window_samples * 0.75)       # 2400
_give_up_length  = (window_samples - _stride) // 2  # 400
_min_infer_samples = int(window_samples * 0.2)      # 640 = 40ms 最小累积
```

| 项 | **实测**（`probe_aec.py`） |
|---|---|
| 窗口/输出粒度 | **3200 样本（200ms）** |
| 首窗延迟 | **126–183 ms** |
| 60s 长时流 | ✅ 无断流、无错误 |
| 样本守恒 | ✅ **1.0000**（合成参考 / 静音 / 省略 farend 三种模式都是） |

**纠正两处基于 V1 的错误推断**：

| 旧推断（V1） | **V2 实际** |
|---|---|
| 首窗累积 9600 样本（600ms 预热） | **无预热**；仅 40ms 最小累积门槛 |
| 直通分支 20% 时间拉伸 | **不存在**（commit `12521fe` 已修） |

**结论**：AEC 服务可用性/稳定性/实时性**全部达标**，**延迟预算比原计划宽松**。

> 待确认：首尾窗会被裁剪（实测 min=1200 / max=5200），尾部 flush 行为需用真实音频再验。

### ASR（`ws://192.168.88.101:31366`）

真实中文语音（`assets/ref_audio/ref_minicpm_signature.wav`，6.02s）实测，识别准确。

**`2pass-offline` 字段契约**（Policy 对齐要用）：

| 字段 | 实测 | 单位 |
|---|---|---|
| `timestamp` | `[[519,692,"呃",0.847],…]` | **字级毫秒 + 每字置信度** |
| `vad_segments` | `[[290, 5980]]` | 毫秒 |
| `start_time`/`end_time` | `290`/`5980` | **毫秒** |
| `confidence` | `{"avg":0.95652,"token":{…}}` | — |
| `stamp_sents` | 分句级 `ts_list` | 毫秒 |

**`turnsense` 实际形状**（与最初设计假设不同）：

```json
{"mode":"turnsense","label":"invalid","prediction_id":2,
 "probabilities":[0.212, 0.212, 0.576],
 "segment_start":330,"segment_end":5190,"speech_duration":4.86}
```

- `probabilities` 是 **3 元素数组**（complete/incomplete/invalid），**不是 dict**
- `segment_start`/`segment_end` 毫秒；`speech_duration` **秒**
- ⚠️ **条件触发，不是每段都发**：合成信号触发 1 次，真实语音 **0 次** → 下游必须容忍缺失

**部分结果延迟**：首个 `2pass-online` **950–1450ms** —— 高于 Policy 实时决策的理想值，
**所以 barge-in 不能依赖 ASR**（走本地 VAD + 人脸唇动，见决策 3）。

### TTS（`192.168.88.253:31058`，gRPC）

`mltts` / `speaker_id=17`：

| 项 | 实测 |
|---|---|
| `get_version` | 前端 `V1.1.20260902` / 后端 `v1.18.20241121` |
| `check_input_text` | `result=True` ✓ |
| PCM | 108000 样本 = **4.50s @ 24kHz int16** ✓ |
| 幅度 | `[-18103, 17095]`，rms 2260（非静音） |
| **首帧延迟** | **508 ms** |
| 总耗时 | **649 ms**（RTF **0.144**） |
| CHAR_TIME_MAP | ✓ 21 项 `[["你",0.0,0.142],…]`（字符, 起, 止 秒） |

**关键：首帧占了总耗时的 78%。** 整段 `inference` 要等 508ms 才出第一块。
→ 考虑改用 `stream_inference`（按句），可把首字延迟降到句级。本阶段先用 `inference`，
`TtsClient` 按可切换设计。

## 复现

```bash
# AEC（需要 speech_frontend 在同层 code/ 目录）
python -m orchestrator.tests.probe_aec --duration 60
python -m orchestrator.tests.probe_aec --duration 30 --farend-mode omit   # 直通

# ASR（用真实语音才有意义）
python -m orchestrator.tests.probe_asr --wav assets/ref_audio/ref_minicpm_signature.wav --verbose

# TTS（必须在 106 上，需 grpc + TTS/protos）
ssh 192.168.89.106 'cd /data/megastore/Projects/DuJing/code && \
  /home/dujing/miniconda3/envs/py310/bin/python \
  MiniCPM-o-Demo/orchestrator/tests/probe_tts.py'
```

## 阶段 0 未完成项

- **106 缺 `libopencv-dev`**（编译 G1 人脸库的前置；需 sudo 密码）
  `sudo apt install -y build-essential pkg-config libopencv-dev curl`
- 声学路径延迟 D 未测（需要真机 + 扬声器/麦克风，阶段 3 做）
- 静音 farend 是否过度抑制（需真实含回声音频对比，阶段 3 做）
- 各服务并发上限（阶段 6 压测）
