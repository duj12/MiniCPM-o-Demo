# 客户端接入：106 完整 8PCM 门控

## 与 8767 的区别

| 服务 | 地址 | detect 输入 |
|---|---|---|
| 原 LipFsm-only | `http://192.168.89.106:8767` | 图片帧即可 |
| 完整 8PCM 门控 | `http://192.168.89.106:8768` | 图片帧 + 连续 8 声道 PCM |

接入 8768 时，调用方必须同时修改：

1. 服务地址从 `8767` 改为 `8768`。
2. 每次 `/face/detect` 或 WebSocket detect 都携带 `pcm8_base64`。
3. 同时携带该 PCM 块的起始时间 `pcm8_timestamp_us`。
4. 保证 PCM 与视频采集时间同步，并维持块间连续。

只给 8768 发送图片帧会返回 HTTP 422，不会降级成纯视觉门控。

## HTTP 请求

```json
{
  "session_id": "meeting-001",
  "timestamp_us": 1200000,
  "frame_jpeg_b64": "<JPEG Base64>",
  "pcm8_base64": "<8 声道 PCM Base64>",
  "pcm8_timestamp_us": 1000000
}
```

`pcm8_base64` 格式：

- 16kHz
- 小端 int16
- 8 声道交错，每帧顺序为 `ch0,ch1,...,ch7`
- 数据长度必须能被 `8 * 2` 整除
- ch2–5：近场麦克风
- ch6–7：数字人回采

音频区间必须满足：

```text
pcm8_timestamp_us <= 音频采样时间 < timestamp_us
```

每块音频必须紧接上一块，不能因图片帧率而重复、跳块或使用未来音频。服务最多接受 4 个并发 session，同一 `session_id` 的请求必须串行。

## 响应

现有响应字段保持不变。`speaking`、`lip_state` 和
`state.lip_speaking_confidence` 表示视觉与 8PCM 后的最终门控结论：

- `speaking=true`：当前追踪目标正在说话。
- `speaking=false`：目标未说话、音频方向不匹配或音频只来自数字人回采。

## 常见错误

| HTTP 状态 | 原因 |
|---|---|
| 422 | 门控模式缺少 PCM 字段 |
| 422 | PCM 不是 8 声道 int16 对齐数据 |
| 422 | PCM 时间不连续 |
| 422 | PCM 包含当前视频时间戳之后的采样 |
| 400 | JPEG Base64 无效或图片无法解码 |

## 编排侧接入（106:8101，客户端 → 编排 → 8768）

客户端**不直连** 8768，而是把 8 声道 PCM 随音频块发给编排，由编排拼块转发。
编排这侧只多一个字段：

```json
{
  "type": "audio",
  "audio_base64": "<16k float32 单声道，现有字段>",
  "pcm8_base64": "<16k int16 LE 8 声道交错，新增，可选>",
  "ctx_time": 123.456,
  "epoch": 1
}
```

- `pcm8_base64` 与 `audio_base64` 是**同一时间窗**切出来的（同一块 100ms）。
- 格式同上文的 `pcm8_base64`：16kHz、小端 int16、8 声道交错、长度 `% 16 == 0`。
- 不发这个字段 = 旧行为；发了但编排没开 `ORCH_FACE_PCM8` 会告警一次并忽略。
- **什么时候发由服务端说了算**：`session.ready` 里多一个 `face_pcm8: true|false`。
  客户端照着它决定要不要在 wire 上带这份字节 —— 8ch@16k 的 base64 是
  ~341KB/s，对着 8767（不需要 PCM）纯属白烧带宽，所以默认只有 8101 这类
  8PCM 门控实例才会是 `true`。
- 编排会把「自上次 detect 以来收到的 PCM」拼成**一整块连续数据**发出去，
  时间戳（`pcm8_timestamp_us` / `timestamp_us`）由编排内部的采样游标推出，
  客户端**不需要**给 PCM 打时间戳，只需要保证块与块之间不丢采样。
- 没有新 PCM 的帧编排不会调 detect（8768 缺 PCM 会 422），所以人脸状态
  更新频率降到音频块频率（10Hz），与唇动聚合窗口一致。
- 服务端单请求 PCM 上限 **1 秒**（实测 1008ms 就回 `413 PCM8 exceeds
  service limit`），积压超过 1s 时编排会自动切块连发，客户端不必关心。

### Android 测试页已经接好（`static/orchestrator-test.html`）

音频来源选 **AlsaServer 转发** 时，页面本来就收到 8 声道原始帧（AlsaServer
`ws://127.0.0.1:38888`，24 字节帧头 + s16le 交错帧体）。现在它除了把
ch2-5/ch6-7 下混成 mic/ref，还会**保留一份原始交错字节**，按同样的 100ms
边界切成 `chunk.pcm8`；`session.ready` 回 `face_pcm8: true` 时才真正塞进
`audio` 消息（对着 8100 这种旧实例一个字节都不多传）。
条件不满足（不是 8 声道 / 不是 16k / 缓冲错位）就整段不带 —— 宁可这段不发，
也不发错位的字节（8768 的时间轴错一次会把会话粘死）。

服务端启动参数：

```bash
ORCH_PORT=8101 \
ORCH_FACE_SERVICE_URL=http://192.168.89.106:8768 \
ORCH_FACE_PCM8=1 \
bash orchestrator/run_orch_106.sh
```

## 联调前检查

```bash
curl http://192.168.89.106:8768/health
```

必须返回 `"ok":true`。完整协议见
[`docs/speaking_gate_server.md`](../../docs/speaking_gate_server.md)。
