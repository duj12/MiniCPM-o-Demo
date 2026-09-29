# 105 进程内 IC + 远端人脸 部署手册

对应分支：`feat/ic-inprocess-105`（MiniCPM-o-Demo）、
`feat/agent-sink-session-envelope`（interactioncore）。

## 为什么是**独立路径**，不是同一份代码

105 与 106 读的是**同一份 NFS 代码** `/data/megastore/Projects/DuJing/code`，
而且两边的 conda env 都把 `interactioncore` 装成了**指向那份共享目录的
editable install**：

```
orch105 (105)  ->  /data/.../DuJing/code/interactioncore
py310   (106)  ->  /data/.../DuJing/code/interactioncore
```

所以在共享目录里改一行 IC 代码，**106 下次重启就会跟着变**。
**git 分支挡不住这件事** —— 远程根本没有 git 仓库，只是同步过来的文件树。

因此 105 的新部署走另一份拷贝：

```
/data/megastore/Projects/DuJing/code-inprocess-105
```

不同路径 = 不同文件，105 只从这里读，106 完全不受影响。
（G1 人脸服务在 105 上早就是这么做的：`.../tianxiaowen/code/g1-face-service-105`。）

## ⚠️ 光设 PYTHONPATH 没用（实测踩过）

pip 的 editable install 装的是一个 **`sys.meta_path` 钩子**：

```python
# __editable___interaction_0_1_0_finder.py
MAPPING = {'interaction': '/data/.../DuJing/code/interactioncore/interaction'}
```

`meta_path` 的查找**优先于 `sys.path`** —— 所以无论把隔离路径放进
`PYTHONPATH` 还是 `sys.path[0]`，`import interaction` 都会命中那份**硬编码的
共享路径**，也就是静默地跑到 106 正在用的代码上。**没有任何报错**，
只表现为「改了代码却不生效」。

解法是启动脚本里生成的 `sitecustomize.py`（解释器启动时自动执行）：

1. 摘掉 `sys.meta_path` 里的 `_EditableFinder`
2. 作废已加载的 `interaction.*`
3. 把隔离路径插到 `sys.path` 最前

## 部署

```bash
# 1) 同步代码（从开发机）
NEW=/data/megastore/Projects/DuJing/code-inprocess-105
scp MiniCPM-o-Demo/orchestrator/{main.py,config.py,run_orch_inprocess_105.sh} \
    dujing@192.168.89.105:$NEW/MiniCPM-o-Demo/orchestrator/
scp MiniCPM-o-Demo/orchestrator/interaction/{client.py,downstream.py,inprocess.py} \
    dujing@192.168.89.105:$NEW/MiniCPM-o-Demo/orchestrator/interaction/
scp MiniCPM-o-Demo/orchestrator/face/remote_provider.py \
    dujing@192.168.89.105:$NEW/MiniCPM-o-Demo/orchestrator/face/
scp interactioncore/interaction/{runtime.py,grpc_server.py} \
    dujing@192.168.89.105:$NEW/interactioncore/interaction/

# 2) 启动（脚本自己会生成 sitecustomize.py）
bash $NEW/MiniCPM-o-Demo/orchestrator/run_orch_inprocess_105.sh
```

⚠️ `main.py` 与 `config.py` **必须一起同步** —— `main.py` 读 `cfg.ic_mode`，
只有 `config.py` 定义它。只同步一个会 `AttributeError`。

## 与 105 现有部署的差异

| | 现有（:50051 + :8100） | 本次（新端口，如 :8101） |
|---|---|---|
| IC | 独立 gRPC 服务 `interaction.grpc_server` | **进程内 Engine**，每会话一份，不需要单独进程 |
| 人脸 | CDLL 加载 `libsdk_stream.so` | **HTTP** 调 `http://192.168.89.105:8767` |
| 并发 | 单例，后来者接管（`_OWNER` 闸门） | 每会话独占，天然隔离 |

## 端口与回退

测试期用**新端口**（如 8101），与现役 8100 并存，互不影响：

```bash
ORCH_PORT=8101 bash $NEW/MiniCPM-o-Demo/orchestrator/run_orch_inprocess_105.sh
```

回退：`ORCH_IC_MODE=grpc` 一行切回远端 IC；
`ORCH_FACE_SERVICE_URL` 留空即回本地 CDLL。

## ⚠️ 切换硬前置：Agent Platform

**Agent 目前直连 IC 的 gRPC `ApplyAgent`**（写 `agent.status` /
`session_end_pending`，policy 的 SOP 39 与 PENDING_ANNOUNCE 分支依赖它）。
进程内模式下**那个 gRPC 端口不存在了**，所以 Agent 必须改为：

```
POST {callback_ic}/apply_agent        # callback_ic 由 IC 的 payload 带过去
{"session_id": "...", "status": "BUSY", "session_end_pending": true}
```

**Agent 不改，SOP 39 / PENDING_ANNOUNCE 两条分支失效。**
编排侧这个端点已经就绪（`POST /v1/ic/apply_agent`，按 `X-Session-Id` 路由），
且 `ORCH_IC_MODE=grpc` 时会**转发**到远端 IC 的 gRPC `ApplyAgent`
—— 所以可以先让 Agent 改指向、再切模式，两步独立上线。

## 管理测试实例（**别用 pkill**）

```bash
# pkill 的模式会匹配到 ssh 自身的命令行，把自己的会话杀掉
# （实测踩过两次，表现为 ssh 突然 exit 255、日志里什么都没有）
# 用 PID 文件：
cat /tmp/orch_8101.pid | xargs -r kill
```

## 验收

```bash
NEW=/data/megastore/Projects/DuJing/code-inprocess-105
cd $NEW/MiniCPM-o-Demo && export PYTHONPATH=$NEW
PY=/home/dujing/miniconda3/envs/orch105/bin/python

# 1) 单元：会话隔离 + apply_agent 语义 + Confidence 枚举（41 项）
$PY -m orchestrator.tests.test_ic_inprocess_isolation

# 2) 两路真会话并发 + apply_agent 路由（11 项）
$PY -m orchestrator.tests.test_ic_multi_session_live --url http://127.0.0.1:8101

# 3) 远端人脸端到端（9 项）
$PY -m orchestrator.tests.test_face_remote_live --url http://127.0.0.1:8101 \
  --mjpeg /data/megastore/Projects/DuJing/code/board-face-and-cloud-infer/G1/sample/camera_original.mjpeg \
  --max-frames 120 --fps 25
```

实测结果（2026-09-29）：41/41、11/11、9/9。
人脸那条里 `wake max=2191ms`（阈值 2000ms）是关键证据 ——
它证明发往服务端的是**单调微秒**而不是音频采样序号
（后者会让 dwell 慢 62.5 倍，永远唤不醒）。

## 踩过的坑（都已在代码里注释）

| 坑 | 症状 | 处理 |
|---|---|---|
| `PYTHONPATH` 盖不过 editable install | 改了代码不生效，无任何报错 | 启动脚本生成 `sitecustomize.py` 摘掉 `sys.meta_path` 钩子 |
| 档位字符串没转 `Confidence` 枚举 | 第一次 `tick` 就崩，会话完全无决策 | `inprocess._CONF_FIELDS` 统一归一 |
| `AgentStatus` 值是小写（`idle`/`busy`） | SOP 39 静默不生效（比较恒 False） | 走枚举反查，大小写都认 |
| 人脸 `close()` 复用死连接 → `Broken pipe` | 服务端会话泄漏，4 个后 503「人脸突然坏了」 | `_post` 连接层失败重连重试一次 |
| 测试不过滤 5Hz state | 误读 `tracks[0].dwell_ms`（那里根本没有）→ 误报没人脸 | 从 `state` / `wake` 取 |
| `pkill` 匹配到 ssh 自身命令行 | ssh 突然 exit 255，日志什么都没有 | 用 PID 文件管理 |

## 人脸服务会话泄漏的观察方法

`GET http://192.168.89.105:8767/health` 看 `sessions`。
⚠️ **要等收尾结束再看** —— 会话 drain 要等 ASR（实测 10~20s），
刚跑完测试立刻看会看到"泄漏"，其实只是还没关。
正常应在 30s 内回落到 0。手动清理：

```bash
curl -s -X POST http://192.168.89.105:8767/face/close \
  -H "Content-Type: application/json" -d '{"session_id":"<sid>"}'
```
