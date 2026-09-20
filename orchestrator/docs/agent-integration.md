# Agent 接入指南 —— 把回复文本送进 TTS

本文给 **Agent 方**看：你生成完回复文本后，怎么让它**在这个终端上说出声**。

---

## 一、整体链路

```
感知（ASR/人脸/唇动）
        │  gRPC apply_*
        ▼
  InteractionCore ──Tick()→Action──┐
   （P0 SOP 决策）                  │
                                   ▼
                        ANSWER / INSERT / YIELD / END
                                   │  HTTP POST
                                   ▼
                            Agent Platform（你）
                                   │  生成回复文本
                                   ▼
                   POST /v1/speak  ← ★ 本文档讲的接口
                                   │
                                   ▼
                   编排服务（orchestrator）→ TTS → 扬声器
```

**分工**：

| 谁 | 干什么 |
|---|---|
| InteractionCore | 决定**什么时候**该开口（`ANSWER`）、什么时候放行慢结果（`INSERT`） |
| **Agent（你）** | 决定**说什么**，然后调 `/v1/speak` 让我们播出来 |
| 编排服务 | 只负责**播**（TTS + 音频调度 + 回声消除），不决定说什么 |

> **为什么不让 Agent 直接调 TTS 服务？**
> 因为浏览器播放要和 AEC 参考轨、播放回执严格对齐（回声消除靠它）。
> 走编排服务，这些全部自动处理；绕过它会导致回声消不掉。

---

## 二、接口

### 2.1 整段播报 `POST /v1/speak`

回复不长（一次生成完）时用这个。

**请求**

```
POST http://<orchestrator>:8100/v1/speak
Content-Type: application/json
X-Session-Id: <会话 id>          ← 见 §2.3

{
  "text": "您好，我是小智，建议您先挂内科看看。",   // 必填，要播的文字
  "interrupt": false,                              // 可选，true = 先掐断当前播报
  "tts_type": "mltts",                             // 可选，默认 mltts
  "speaker_id": "17"                               // 可选，默认用服务端配置
}
```

**响应**

```json
{"ok": true, "session_id": "ae3e4aa69667"}
```

失败时：

```json
{"ok": false, "error": "找不到活跃会话（请在 X-Session-Id 头或 ?session_id= 里指明）"}
```

| 状态码 | 含义 |
|---|---|
| `200` | 已接收（**不代表已经播完**，只是排进播放队列了） |
| `404` | 找不到会话 —— 会话可能已经结束，或 session id 不对 |
| `422` | 请求体格式错（缺 `text` 字段等） |

### 2.2 流式播报 `POST /v1/speak/stream`

回复很长、希望**边生成边播**（首声更早）时用这个。逐片推送文本片段：

```
POST http://<orchestrator>:8100/v1/speak/stream
Content-Type: application/json
X-Session-Id: <会话 id>

{"stream_id": "agent-turn-42", "text": "您好，我是", "is_final": false}
{"stream_id": "agent-turn-42", "text": "小智，建议您", "is_final": false}
{"stream_id": "agent-turn-42", "text": "先挂内科看看。", "is_final": true}
```

**⚠️ 三条铁律**

1. **`stream_id` 同一轮内必须保持不变**，下一轮**必须换一个新的**
   （比如用 `agent-{轮次}` 或 uuid）。
   *踩过的坑*：用固定常量当 stream_id 时，第二轮的文本会被追加进
   第一轮那个已经收尾的流，表现为**第二轮永远不播*。
2. **最后一片必须 `is_final: true`** —— 否则流不会收尾，最后一句被卡住不播。
3. **片段会被自动攒批**（攒到句末标点或 ~50 字才送去合成）——
   这是服务端为了音质做的，你按 token 粒度推即可，不用自己切句。

### 2.3 怎么拿到 `X-Session-Id`

编排服务在会话建立时下发给前端（`session.ready` 消息里的 `session_id`）。

**Agent 如果自己不是那个前端**，有两种拿法：

- **推荐**：由调用方（编排服务）在 `on_answer` / `on_insert` 的请求里带上。
  目前这四个端点**还没有** session 字段 —— 需要时请找编排服务侧加。
- **临时**：`GET /healthz` 看活跃会话数，`GET /stats` 列出活跃会话 id。
  **单会话部署**下，不传 `X-Session-Id` 也能路由到那唯一一个会话。

---

## 三、调用时机

| InteractionCore 的 Action | 你该做什么 |
|---|---|
| **`ANSWER`** | 收到 `POST /interaction/on_answer`（带 `transcript`）→ 生成回复 → **立刻**调 `/v1/speak` |
| **`INSERT`** | 收到 `POST /interaction/on_insert` → 把**之前备好的慢结果**现在播出来（调 `/v1/speak`） |
| **`YIELD`** | 收到 `POST /interaction/on_yield` → 用户抢话，**立即停止生成**、别再调 `/v1/speak` |
| **`END`** | 收到 `POST /interaction/on_end` → 会话结束，丢弃待播内容 |

> `ANSWER` = 快回复/快工具，直接播；
> 慢任务（要查几秒的）应先备好文本但**别播**，等 `INSERT` 放行 —— 
> 否则会在用户还在说话时插话。

---

## 四、示例代码

### 4.1 curl（最快验证）

```bash
# 看有没有活跃会话
curl -k https://<orchestrator>:8100/stats

# 播一句
curl -k -X POST https://<orchestrator>:8100/v1/speak \
  -H "Content-Type: application/json" \
  -H "X-Session-Id: ae3e4aa69667" \
  -d '{"text":"您好，我是小智，建议您先挂内科看看。"}'
# → {"ok":true,"session_id":"ae3e4aa69667"}
```

> ⚠️ 编排服务默认跑 **HTTPS**（自签证书），所以是 `https://` +
> `curl -k`（跳过证书校验）。

### 4.2 Python

```python
import json
import urllib.request

ORCH = "https://192.168.89.106:8100"
SESSION_ID = "ae3e4aa69667"          # 见 §2.3


def _post(path: str, payload: dict) -> dict:
    req = urllib.request.Request(
        ORCH + path,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "X-Session-Id": SESSION_ID,
        },
        method="POST",
    )
    # 自签证书：跳过校验（生产建议换成带 CA 的正规校验）
    ctx = ssl._create_unverified_context()
    with urllib.request.urlopen(req, timeout=5, context=ctx) as r:
        return json.loads(r.read().decode("utf-8"))


# ---- 整段播（快回复）----
print(_post("/v1/speak", {"text": "您好，我是小智，建议您先挂内科看看。"}))
# → {'ok': True, 'session_id': '...'}


# ---- 流式播（长回复，边生成边播）----
def speak_stream(stream_id: str, chunks):
    for i, piece in enumerate(chunks):
        is_last = (i == len(chunks) - 1)
        print(_post("/v1/speak/stream", {
            "stream_id": stream_id,      # ⚠️ 同一轮不变，换轮必须换
            "text": piece,
            "is_final": is_last,         # ⚠️ 最后一片必须 True
        }))

speak_stream("agent-turn-1", ["您好，我是小智。", "头痛胸痛建议先挂内科。"])
```

### 4.3 在 FastAPI 里接 InteractionCore 的 Action

```python
from fastapi import FastAPI
import threading, uuid

app = FastAPI()
_lock = threading.Lock()
_session_id = ""

@app.post("/interaction/on_answer")
async def on_answer(body: dict):
    """InteractionCore 判定'该回答了'。"""
    transcript = body.get("transcript", "")
    name = body.get("display_name")
    # ↓ 换成你自己的生成逻辑（LLM / 工具 / 快 skill）
    reply = f"{name + '，' if name else ''}您说的是：{transcript}。" \
            f"我这就帮您查。"
    _post("/v1/speak", {"text": reply})
    return {"status": "ok"}

@app.post("/interaction/on_insert")
async def on_insert():
    """放行慢结果 —— 把之前备好的文本现在播出来。"""
    with _lock:
        pending = _take_pending_result()
    if pending:
        _post("/v1/speak", {"text": pending, "interrupt": False})
    return {"status": "ok"}

@app.post("/interaction/on_yield")
async def on_yield():
    """用户抢话 —— 停止生成，别再播。"""
    _cancel_generation()
    return {"status": "ok"}

@app.post("/interaction/on_end")
async def on_end():
    """会话结束。"""
    _clear_pending()
    return {"status": "ok"}
```

---

## 五、常见问题

**Q：接口返回 `ok:true`，但没听到声音？**

按顺序查：

1. **会话是否还活着** —— `curl -k https://<orchestrator>:8100/healthz`，
   `active_sessions` 是不是 0。会话结束后调用会返回 `404`
2. **`X-Session-Id` 对不对** —— 多会话部署下必须给对，否则 404
3. **`text` 是不是空的** —— 空文本会返回 `{"ok":false,"error":"text 为空"}`
4. 看编排服务日志有没有 `Agent 播报（N 字）` —— 有这行说明收到了

**Q：`text` 里能放 Markdown 吗？**

不建议。这是**语音播报**，`**加粗**`、列表符号会被念出来。用平白的口语。

**Q：能连续调多次吗？**

可以。默认是**排队**（依次播）；后一句想打断前一句就带 `"interrupt": true`。

**Q：我们能在播完之后拿到通知吗？**

暂时没有。需要的话告诉编排服务侧加一个回调。

**Q：端口/协议？**

- 默认 `https://<orchestrator>:8100`（自签证书，客户端要跳过校验）
- 也可以用 `http://`（如果编排服务以 HTTP 起），或经 Tailscale 的域名访问
