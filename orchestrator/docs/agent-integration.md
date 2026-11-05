# Agent 接入指南 —— AgentService/Interact

当前链路使用独立 Agent gRPC 服务。InteractionCore 固定在本仓库的
`vendor/interactioncore` submodule，每路宿主会话持有独立的 `Engine`。
IC 的两个同步 Sink 只向宿主回调事件；网络连接和实际播报由宿主负责。

## 一、整体链路

```text
ASR / 人脸 / 唇动 / 视觉 / 真实播放回执
                  │ 本地 apply_* / tick
                  ▼
       每会话 InteractionCore Engine
         │ AgentSink     │ ExpressionSink
         └────────┬──────┘
                  ▼
           orchestrator 宿主
         │                 │
         │ AgentAction     │ Speak / Stop / Pause / Resume
         ▼                 ▼
AgentService/Interact     ActionExecutor → TTS → 浏览器
         │ AgentEvent                         │
         └──────────────► 宿主 ◄── playback ──┘
```

| 谁 | 职责 |
|---|---|
| InteractionCore | 运行交互策略，交付 Agent 动作与表达事件 |
| Agent | 生成回复与状态，通过同一条双向 RPC 返回 |
| orchestrator | 建立每会话 RPC、维护轮次标识、应用 Agent 状态、执行 TTS 和打断 |
| 浏览器 | 播放音频并上报真实播放回执，供宿主更新参考轨与 `apply_playback_active` |

Agent 的 `done` / `IDLE` 只描述生成状态，不代表音频已经播完。
实际播放状态以浏览器回执为准。Agent 文本必须经宿主播放，才能保留 AEC
参考轨、播放定位及暂停/续播的闭环。

## 二、协议与会话隔离

协议源见 [agent.proto](../../agent_protocol/agent.proto)，生成方式和依赖版本见
[Agent wire contract](../../agent_protocol/README.md)。
本仓库副本须与 Agent 服务的 canonical proto 保持一致。

唯一 RPC：

```protobuf
service AgentService {
  rpc Interact(stream AgentAction) returns (stream AgentEvent);
}
```

每路 `OrchestratorSession` 拥有自己的 `AgentClient`、请求队列与 RPC。
每条请求和响应都带 `session_id`、`ci`、`epoch`，Agent 必须原样回传。
宿主拒绝其他会话、旧轮次及已失效连接的事件。多路会话可以共用同一个 Agent
服务地址，服务端也必须按 RPC / 会话隔离上下文。

| 动作 | 宿主发送及预期语义 |
|---|---|
| `ANSWER`（真实用户话轮） | 新建 `ci`，递增 `epoch`，携带 ASR `transcript` |
| `ANSWER`（视觉快照） | 空 `transcript`，`vlm` 携带视觉描述，只更新上下文，不凭空回复 |
| `INSERT` | 复用当前话轮的 `ci/epoch`，放行待播结果 |
| `YIELD` | 复用当前话轮标识；取消生成和旧待播文本，保留 Agent 上下文 |
| `END` | 清除 Agent 上下文与旧待播内容，保留 RPC |

活动轮期间视觉更新复用轮次标识；首次视觉更新或 END 后的新视觉更新建立新标识。
ASR 与视觉分别使用字符串 `transcript`、`vlm`，不再使用 HTTP 时代的
`legacy/dual/dict` 载荷切换。协议还定义了 `identity_id/display_name`、
`language`、`affect`、`app_id`、`ai_intent/ai_style` 和 `mcp_servers`；
具体字段及可选性以 proto 为准。

## 三、Agent 返回什么

| AgentEvent payload | 宿主处理 |
|---|---|
| `ack` | 记录动作是否接收；被拒绝时输出诊断日志 |
| `state` | 将可选 `status/session_end_pending` 投影到本会话 IC |
| `text: FEEDBACK` | 按 `stream_id` 累积，`is_final=true` 后整段播报 |
| `text: ANSWER` | 按流推送正文，复用同一 `stream_id`，最后一片须 `is_final=true` |
| `done` | 标记生成完成，继续等待真实播放回执 |
| `error` | 使本轮待播文本失效，宿主停止当前输出并记录错误 |

`TextChunk.stream_id` 在同一文本流内保持不变，新流使用新值。
FEEDBACK 和 ANSWER 是两类文本，不能把 FEEDBACK 当成正文重复返回。

断流后当前 RPC 终止，宿主停止输出、清本地活动轮，不重放已发送动作。
之后新的真实 ANSWER 可以创建新 RPC 与新话轮；新宿主会话同样创建独立 RPC。
恢复前旧连接、旧 `ci/epoch` 的文本不可继续播放。

## 四、部署与验证

首次在其他机器部署前，需先将 InteractionCore 固定提交
`40921a6c9cff583071aa70c463c65f016607fab2` 发布到 `.gitmodules` 指向的远端，
并由可获取的分支保留该提交。当前提交尚未发布，直接初始化 submodule 会报
`not our ref`；本地已有副本不能替代这个部署前置。

从仓库根目录执行：

```bash
git submodule update --init --recursive
python -m pip install -r orchestrator/requirements.txt
```

本地完整启动命令见 [Orchestrator README](../README.md#开发调试直接起)。
直接启动会读取仓库根目录 `.env`；需同时配置 Agent、Omni 描述与人脸服务，
终端环境变量高于 `.env`。

生产配置保留在 `orchestrator/run_orch_105.sh` / `run_orch_106.sh`：
105 默认连接 `192.168.89.103:50051`，106 默认连接 `192.168.89.102:50051`。
目标必须运行 AgentService，而非旧的 IC gRPC 服务。可用 `ORCH_AGENT_GRPC`
覆盖地址，`ORCH_PY` 覆盖已安装依赖的解释器。

不再启动独立 IC；`ORCH_IC_MODE=grpc`、`ic.grpc`、`ORCH_AGENT_URL`、
HTTP `/interaction/on_*`、`set_target/callback_ic` 和
`/v1/ic/apply_agent` 已退役。无需给 Agent 配置回连宿主的地址。

```bash
python -m pytest orchestrator/tests/test_agent_integration.py -q
# 可选：以实际 Agent 服务源码运行独立进程 loopback
AGENT_TEST_SOURCE=../openai-agent-demo AGENT_TEST_PYTHON=/path/to/agent/python \
  python -m pytest orchestrator/tests/test_agent_integration.py -q
```

## 五、手工播报诊断

HTTP 播报接口继续供手工工具使用；Agent 集成使用上述 gRPC 事件。
多会话时显式指定 `X-Session-Id`，避免依赖单会话兜底。

```bash
curl -k https://<orchestrator>:8100/stats
curl -k -X POST https://<orchestrator>:8100/v1/speak \
  -H "Content-Type: application/json" \
  -H "X-Session-Id: <session_id>" \
  -d '{"text":"您好，这是播报测试。"}'
```

`200/ok:true` 表示已接收，不代表播完；`404` 表示会话不存在。
长文本可逐次 POST `/v1/speak/stream`，每次请求一个 JSON 对象：

```json
{"stream_id":"manual-turn-1","text":"您好，这是","is_final":false}
{"stream_id":"manual-turn-1","text":"流式测试。","is_final":true}
```

同一流复用 `stream_id`，换轮更换，最后一片必须终结。文本会按标点和长度攒批后
送 TTS。用口语文本，Markdown 标记可能被直接念出来。暂停、续播与停止接口见
[orchestrator README](../README.md#http-接口)。
