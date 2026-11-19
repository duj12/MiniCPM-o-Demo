# state_head —— 带置信度的 2.1-lite 状态输出

一个**旁挂**的 HTTP 服务：吃几帧 JPEG，吐 `Social_State_Schema_v2_1_Lite.md`
的 **E2** 状态块（每个槽位 `{label, confidence}`）。

它替掉的是"让 Qwen3-Omni 用自然语言描述画面、再从文本里猜枚举值"那条路 ——
那条路既不可枚举也没有置信度。这里用 **typed question**（closed enum 的问句）
一次前向回答 19~20 个问题，直接拿到每个选项的概率。

Qwen3-Omni **不退休**：它继续负责必须用自由文本表达的槽位
（`FEATURE.appearance`、`ENV.objects`、`POSITION.relative_positions`、
一行 `description`）。两条链在编排侧**并行**跑，谁先到谁先写状态。

```
        ┌── Qwen3-Omni  :8006  →  自由文本槽位（~300ms）
帧 ─────┤
        └── state_head :22520  →  E2 + 置信度   （~250~450ms）
```

## 接口

| 方法 | 路径 | 说明 |
|---|---|---|
| `POST` | `/v1/state` | 主接口，见下 |
| `GET` | `/health` | 就绪、当前端点、在途数、累计统计 |
| `GET` | `/v1/questions?stage=turn` | 当前生效的题集（排错用） |

```jsonc
// POST /v1/state
{
  "frames_b64": ["<裸 base64 或 data URL>", "<1 秒前那帧>"],  // 2 帧走 video，1 帧走 image
  "fps": 1.0,
  "stage": "turn",          // turn=19 问 / full=20 问（多 env.scene）
  "tier2": false,           // true 时追加 3 个长尾动作
  "session_id": "abc"       // 只用于"同会话不重复打"，跨会话完全并发
}
```

```jsonc
{
  "schema_version": "2.1",
  "state": { "ATTENTION": {"orientation": {"label": "facing_camera", "confidence": 0.71},
                           "gaze": {...}},
             "ACTION": {"events": [{"type": "nod", "count": 2, "confidence": 0.83}]},
             "EMOTION": {"label": {...}, "visual": {...}, "conflict": null},
             "FEATURE": {"appearance": null, "accessories": []},
             "BEHAVIOR": {...}, "RELATION": {...}, "ENV": {...}, "POSITION": {...} },
  "confidence": {"ATTENTION.orientation": 0.71, "ACTION.events[0]": 0.83, ...},
  "latency_ms": 312, "model": "OneJev1.6-0.8B", "backend": "onejev",
  "stage": "turn", "n_questions": 19
}
```

错误码：`422` 没有帧（调用方应**保留上一份 E2**）、`502` 推理失败、
`504` 超时、`503` 后端未就绪。

### 三条空值语义（schema §5.1，别混）

| 写法 | 含义 | 触发条件 |
|---|---|---|
| `null` | 本回合**不可观察** | `person_visible` 判假（人不在画面） |
| `unknown` | 枚举槽位**证据不足** | 模型自己选 `unknown`，或概率低于 `STATE_ENUM_MIN_CONF` |
| `[]` | 观察到了、**确实没有** | 人可见，但没有任何事件过 `STATE_ACTION_P` |

`ACTION.events=[]` 与 `null` 对策略完全相反 —— 前者"看着你但没动作"，
后者"根本没看到人"。所以 `person_visible` 是第一问，也是整份 E2 的总闸；
取不到它的概率时**按可见处理**（fail-open），避免一个门控故障把整份状态打成 null。

### 置信度用哪个数

用 `probabilities[label]`（原始概率），**不是** Flide 响应里的 `confidence`。
后者把 top prob 从均匀重标定到"全押"：`(p_max − 1/k)/(1 − 1/k)`，
k=12 时 0.5 会被放大成 0.45→0.45 但 0.9 会放大成 0.89→0.89… 总之刻度偏大，
和 ECE 校准对不上。schema 要的是可校准的概率。

## 环境变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `STATE_BACKEND` | `onejev` | 目前只有 `onejev` 可信（见 `backends.py` 顶部的 ⚠️） |
| `STATE_BACKEND_URLS` | `http://127.0.0.1:22521` | 逗号分隔的 flide 端点，**按请求轮转** |
| `STATE_MODEL_DIR` | `onejev-1.6-0.8b` | `checkpoints/` 下的子目录名 |
| `STATE_TIMEOUT_S` | `3.0` | 到 flide 的超时 |
| `STATE_MAX_CONCURRENCY` | `64` | 宽的连接上限（**不是**推理闸） |
| `STATE_MAX_MEDIA_PX` | `640` | 发前缩图的长边（0=不缩）。延迟主旋钮之一 |
| `STATE_JPEG_QUALITY` | `85` | 缩图后的重编码质量 |
| `STATE_PERSON_VISIBLE_P` | `0.5` | 人可见判定阈值 |
| `STATE_ACTION_P` | `0.5` | 动作入 `events[]` 的阈值 |
| `STATE_ACTION_MAX_EVENTS` | `2` | `events[]` 最多几条 |
| `STATE_ENUM_MIN_CONF` | `0.0` | 低于此值退 `unknown`（默认不拦，按实测调） |
| `STATE_TIER2` | `0` | 默认是否问长尾动作 |
| `STATE_INCLUDE_LIGHTING` | `1` | 是否问 `env.lighting` |
| `STATE_WARMUP` | `1` | 启动后台预热（不阻塞 `/health`） |
| `STATE_REPLICAS` | `1` | entypoint 起几个 flide（端口 22521+i） |
| `STATE_SESSION_CACHE_GB` | `0.5` | flide 前缀缓存上限。**不设就是"模型起来后剩下的全用"**——与 omni 共卡时会把 GPU0 吃到只剩 1.9GB；0 = 不传参 |

`flide serve` 的合批参数在 `docker/entrypoint-state-head.sh` 里：
`STATE_BATCH_WINDOW_MS`（默认 3）、`--frontends 2`、`--planners 4`、
`STATE_NO_CUDA_GRAPHS`（与 omni 共卡必须为 1）。

## 并发模型（为什么这里没有排队）

合批发生在 **Flide 内部**：`--batch-window-ms` 让 GPU 轮次多等几毫秒，
把同时到达的请求拼成一次前向。客户端一旦自己串起来，模型永远只看到 1 个请求，
合批退化成"一次一个"，8 路并发的第 8 个用户就要等满 2 秒。

所以本服务只有两类控制（`scheduler.py`）：

1. **`SessionGate`** —— 按 `session_id` 互斥。同一会话的 turn 与心跳同时到时，
   后到的**先等前一条结束再发**（用它自己的新帧，不拿旧结果冒充）。跨会话零交集。
2. **`Capacity`** —— 默认 64 的宽上限，只防 fd/内存打爆，打满时告警。

**不取消在途请求**：取消救不回已花掉的算力，只会在 Flide 侧多一次 abort。

## 本地/容器内运行

```bash
# 只跑服务（flide 已在 22521 起着）
python -m uvicorn state_head.app:app --host 0.0.0.0 --port 22520

# 端到端延迟与并发曲线（容器内）
python -m state_head.bench_sop --still /app/assets/frame.jpg --repeat 50
python -m state_head.bench_sop --still /app/assets/frame.jpg --repeat 200 --concurrency 1,2,4,8,16

# SOP 逐槽位准确率（需要 ffmpeg 抽帧）
python -m state_head.bench_sop --clips /data/.../assets/video/sop --stage both \
    --out /app/logs/sop_state.json
```

验收线：**8 路并发 p95 ≤ 500ms**，显存增量 ≤ 与 omni 共卡的空闲量，
且不把 omni 的首字延迟拖到超时。

## M0 实测（2026-11-19，106 GPU0 与 omni 共卡）

环境：OneJev1.6-0.8B，19 问 `stage=turn`，单帧 640px，`--no-cuda-graphs`，
`--batch-window-ms 3`，`STATE_SESSION_CACHE_GB=0.5`。omni 为 `-c 200000
--parallel 10`（per-slot 20000）。

| 并发 | p50 | p95 | 吞吐 |
|---|---|---|---|
| 1 | 176ms | 180ms | 5.7 QPS |
| 2 | 232ms | 289ms | 8.1 QPS |
| 4 | 379ms | **455ms** | 10.6 QPS |
| 8 | ~650ms | ~700ms | 12 QPS |
| 16 | ~920ms | ~1250ms | 16 QPS |

- **显存**：GPU0 已用 18879 → 21520 MiB（state head 常驻 **2.6GB**），
  余量 2605 MiB。不设 `STATE_SESSION_CACHE_GB` 时是 22220 MiB、余量只剩 1.9GB。
- **对 omni 的影响**：无会话时首字 8002/8017/8020ms（基线 7970/8003/8038ms，无差异）；
  4 路 state 满负荷时 8175–8230ms（+2~3%），未出现超时；稳态 0 个 502。
- `--batch-window-ms` 取 0/3/8 差别在噪声内（4 路 p95 447/455/464ms），保持 3。
- **冷启动**：容器起来后 flide 尚未就绪的几秒里 `/v1/state` 会返回 502；
  编排侧把它当"本轮没状态"，不会影响文本链路。预热请求在 2.4s 左右。

## 部署

作为第三个镜像挂在现有 CI 链上（`omnillm-state-head`），模型由
`docker/build-image.sh` 从 `checkpoints/<STATE_MODEL_DIR>` **单独**暂存到
`.bundle-state/`（不能塞进 cpp 镜像的 `.bundle-checkpoints/`，那会让 20G 的
cpp 镜像再多 2~4G）。

```bash
# 106
docker compose -f docker-compose.cpp.yml --profile state-head up -d state-head
curl -s http://127.0.0.1:22520/health
```

⚠️ **重启 omni 前先停 state head**（`docker stop omni-llm-state-head`），
否则 omni 重新预分配 KV 可能失败；`depends_on: service_healthy` 保证的是
反过来的顺序（omni 先占好 KV，state head 再进）。

## 文件

| 文件 | 职责 |
|---|---|
| `questions.py` | 题集**唯一真值源**：枚举白名单、问句、`build_questions()` |
| `assemble.py` | answers → E2：阈值、`events[]`、档位折算、空值语义 |
| `backends.py` | `/v1/systemone` 的并发转发（轮转 + 失败换端点） |
| `scheduler.py` | `SessionGate` / `Capacity` / `parse_urls` |
| `app.py` | FastAPI 门面、缩图、lifespan、预热 |
| `bench_sop.py` | 延迟/并发曲线 + SOP 逐槽位准确率 |

**本包不 import `orchestrator/`**。与编排侧的契约是 schema 文档 +
E2 的 JSON 形状 + `orchestrator/tests/test_state_merge.py`。
