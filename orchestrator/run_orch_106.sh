#!/bin/bash
# 106 生产编排 —— 默认**并发版**（进程内 IC，每会话一份 Engine）。
# `ORCH_IC_MODE=grpc` 一行可回退到单会话版（见下「回退」）。
#
# 用法（106 上）：
#     nohup setsid bash .../orchestrator/run_orch_106.sh </dev/null >/dev/null 2>&1 &
#
#     # 影子实例：不碰 8100 生产，只换端口 —— 回调地址/日志会自动跟着端口走
#     ORCH_PORT=8101 bash .../orchestrator/run_orch_106.sh
#
#     # 回退到 grpc 版（先按 PID 停掉在跑的，见下）
#     ORCH_IC_MODE=grpc nohup setsid bash .../orchestrator/run_orch_106.sh </dev/null >/dev/null 2>&1 &
#
# ## 本脚本管的两件事（都靠环境变量翻，不用换脚本）
#
#   · IC   —— `ORCH_IC_MODE`：`inprocess`（默认）↔ `grpc`（回退，见下）
#   · 人脸 —— `ORCH_FACE_SERVICE_URL`：非空 = 105 的远端服务；
#             **置空 = 回到本地 `.so`**（单会话；见下面人脸段的说明）
#
#   · `ORCH_IC_MODE=inprocess` —— 每路会话在本进程内**独占**一份 IC Engine
#     （`interaction.runtime.Engine`），并发由构造消解，不需要 session 路由。
#     ⇒ 106 上那个独立的 IC 服务（:50051）**不再需要**，本脚本不起它。
#     **但不要杀它** —— 留着就是回退路径。
#   · `ORCH_AGENT_SET_TARGET=0` —— 不再调 Agent 的 `set_ic_target`，
#     改由 IC 在 `AgentSink` 的 payload 里带 `callback_ic`。
#   · 人脸改走 **105 的远端服务**（`http://192.168.89.105:8767`），**不再加载
#     本地 .so** —— 见下面「为什么人脸必须走远端」。
#
# ## 回退到 grpc 版（同一脚本，只翻模式）
#
#     # 1) 停掉在跑的进程 —— **按 PID**。别用 `pkill -f orchestrator.main`：
#     #    那个模式会匹配到 ssh 自己的命令行，把会话一起带走（踩过）
#     ss -ltnp | grep ':8100' | sed -n 's/.*pid=\([0-9]*\).*/\1/p' | xargs -r kill -TERM
#     # 2) 起 grpc 版
#     ORCH_IC_MODE=grpc nohup setsid bash .../orchestrator/run_orch_106.sh </dev/null >/dev/null 2>&1 &
#
#     ⚠️ :50051 上那个独立 IC 服务（`run_ic_106.sh` 起，pid 1286618，2026-09-29
#        起从未重启）**一直在跑 —— 别停它**，它就是回退的依赖。
#
#     ⚠️ 回退**不动人脸** —— 仍走 105:8767。人脸服务化是架构定调（见下），与
#        IC 模式无关。要把人脸也退到本地 .so：`ORCH_FACE_SERVICE_URL=` 置空，
#        并补 `ORCH_FACE_MODELS` / `ORCH_FACE_DB`（106 的本地 .so 是可用的）。
#
#     ⚠️ 回退后若 `agent.status` 写不进去（Agent 已不认 gRPC `ApplyAgent`），
#        按 `run_ic_106.sh` 末尾那段给 IC 服务补
#        `--agent-envelope --callback-ic https://192.168.89.106:8100/v1/ic`。
#
# ## ⚠️ 前置依赖（不满足会**静默**失效，不报错）
#
# Agent（102:8081）必须能从 `on_*` payload 里读 `callback_ic`，并把
# apply_agent 写到 `POST {callback_ic}/apply_agent`。做不到的话
# `agent.status` / `session_end_pending` 永远写不进去 ⇒ policy 的
# **SOP 39 与 PENDING_ANNOUNCE 两条分支失效**。
#
# 判据（会话日志里必须有这一行，没有就是没通）：
#     [<sid>] Agent → IC: apply_agent(status=..., end_pending=...)
#
# ## 为什么必须走共享树
#
# 105 与 106 读的是**同一份** NFS 代码，`interaction` 包在两边都是指向它的
# editable install。改共享树不影响两边**正在跑**的进程（Python 启动时就
# import 完了），只影响下次重启 —— 所以「改了代码」≠「服务更新了」，
# 更新必须重启。见 docs/多会话并发与云服务化-问题与方案.md。
set -eu
SHARED=/data/megastore/Projects/DuJing/code
REPO=$SHARED/MiniCPM-o-Demo
cd "$REPO"

PY="${ORCH_PY:-/home/dujing/miniconda3/envs/py310/bin/python}"
if [ ! -x "$PY" ]; then
  echo "找不到 python: $PY（可用 ORCH_PY=... 覆盖）" >&2
  exit 1
fi

# 先把端口定下来，回调地址跟着它走 —— 影子实例只要换 ORCH_PORT 就自洽
PORT="${ORCH_PORT:-8100}"

# ---- IC：两种模式，`ORCH_IC_MODE` 就是回退开关 ----
#
#   `inprocess`（默认）每会话在本进程内**独占**一份 Engine —— 真并发。
#   `grpc`             回退到 :50051 上的独立 IC 服务（`run_ic_106.sh`）——
#                      单会话，并发靠 IC 侧的 `_OWNER` 闸门「后来者接管」。
#
# ⚠️⚠️ **只翻 `ORCH_IC_MODE`、却把 `ORCH_IC_ADVERTISE` 留在 `:8100` 是最坏的
#       一种半切换**：8100 是 HTTP(S) 端口，而 Agent 会拿 advertise 当 **gRPC
#       目标**去回连 IC → 连不上；而**编排侧看不到任何异常**（回连的不是它）。
#       所以两个模式的三件套在这里一次改全，别在命令行上手工设。
export ORCH_IC_MODE="${ORCH_IC_MODE:-inprocess}"
if [ "$ORCH_IC_MODE" = "grpc" ]; then
  # 编排**自己**去连 IC（同机，localhost 没问题）
  export ORCH_IC_GRPC="${ORCH_IC_GRPC:-127.0.0.1:50051}"
  # 告诉 Agent 回连 IC 的地址 —— grpc 模式下 = **IC 的 gRPC 端口**
  export ORCH_IC_ADVERTISE="${ORCH_IC_ADVERTISE:-192.168.89.106:50051}"
  # 会话结束归还到本机 IC（grpc 模式每台机器一个，本机那个就是共用的）
  export ORCH_IC_RESTORE="${ORCH_IC_RESTORE:-192.168.89.106:50051}"
  # 回退态靠 Agent 的 `set_ic_target` 拿回连地址 ⇒ 必须开
  export ORCH_AGENT_SET_TARGET="${ORCH_AGENT_SET_TARGET:-1}"
else
  # 告诉 Agent 往哪写 apply_agent。必须**外部可回连** —— 写 127.0.0.1 的话
  # Agent 会去连它自己。
  export ORCH_IC_ADVERTISE="${ORCH_IC_ADVERTISE:-192.168.89.106:$PORT}"
  export ORCH_IC_CALLBACK_BASE="${ORCH_IC_CALLBACK_BASE:-https://192.168.89.106:$PORT/v1/ic}"
  # Agent 已能按 payload 里的 `callback_ic` 回写 ⇒ 关掉进程级全局的 set_target
  export ORCH_AGENT_SET_TARGET="${ORCH_AGENT_SET_TARGET:-0}"
fi
# IC 播报打的是**本机自己**的 /v1/speak；CA 用编排自己的自签证书
# （证书 SAN 必须覆盖 127.0.0.1，否则校验失败 —— 而失败只记日志，
#  表现为「IC 判 GREET 但没人播」，很难查）。
# ⚠️ grpc 模式下这两个变量**在编排侧不生效** —— IC 服务自己带
#    `--expression-url/--expression-ca`（见 `run_ic_106.sh`）。留着只为少一处 diff。
export ORCH_IC_EXPRESSION_URL="${ORCH_IC_EXPRESSION_URL:-https://127.0.0.1:$PORT}"
export ORCH_IC_EXPRESSION_CA="${ORCH_IC_EXPRESSION_CA:-$REPO/certs/cert.pem}"

# ---- Agent：106 用 102:8081（config.py 的默认值就是它，显式写出来自文档）----
# （`ORCH_AGENT_SET_TARGET` 随 IC 模式在上面定）
export ORCH_AGENT_URL="${ORCH_AGENT_URL:-http://192.168.89.102:8081}"

# ---- 人脸：走 105 的远端服务（按 session 隔离），**不加载本地 .so** ----
#
# ## 为什么人脸必须走远端（而不是「106 本地 .so 也能用」）
#
# 本地 provider 是 `ctypes.CDLL` 直连 `libsdk_stream.so`。CDLL 在**一个进程里
# 只加载一次**，.so 内部的跟踪/dwell/身份状态全是**进程级全局** —— 两路会话
# 各建一个 provider 也**共用同一份**状态，互相覆盖。这正是当初把 8767 单独拆
# 出来的原因，也是 105 本地 `.so` 路线被放弃的原因。改了进程内 IC 拿到真并发
# 之后，人脸这边不同步改，并发就只成了一半：IC 状态隔离了，人脸还在串。
#
# 服务端按 `session_id` 隔离（`G1_FACE_SESSIONS`），`/face/close` 释放。
#
# ## 代价（写清楚，别当没发生）
#
#   ① **106 从此依赖 105:8767** —— 105 挂了 106 就没人脸。fail-soft：无人脸
#      但不报错，表现为「人来了不迎宾」，静默。
#   ② **与 105 共用 `G1_FACE_MAX_SESSIONS=4` 的额度** —— 两边同时跑会话会打满
#      → 503 → 那一帧丢；打满时 106 的迎宾可能比 105 先坏。要真隔离得在 106
#      上再部署一份服务，或把额度提上去。
#   ③ 每帧多一跳网络（25 fps），`face_service_timeout_s` 是单帧上界。超时只丢
#      这一帧 —— `offer` 是 `put_nowait`，音频路径不受影响。
# ⚠️ 这里用 `-` 而不是 `:-` —— 只有 `-` 才让**显式给的空值**保留下来，
#    从而真的关掉远端模式、退回本地 `.so`。写成 `:-` 的话
#    `ORCH_FACE_SERVICE_URL= bash ...` 会被默认值顶掉，看着像「改了却没生效」。
export ORCH_ENABLE_FACE="${ORCH_ENABLE_FACE:-1}"
export ORCH_FACE_SERVICE_URL="${ORCH_FACE_SERVICE_URL-http://192.168.89.105:8767}"
export ORCH_FACE_SERVICE_TIMEOUT_S="${ORCH_FACE_SERVICE_TIMEOUT_S:-2.0}"
export G1_FACE_DEBUG="${G1_FACE_DEBUG:-0}"

# ---- 本地 .so 退路（只在 `ORCH_FACE_SERVICE_URL=` 显式置空时走）----
# 走 `G1FaceProvider`（`ctypes.CDLL` 直连 `libsdk_stream.so`，**单会话**）。
# `g1_root` 会自动探测「与本仓库同级的 board-face-and-cloud-infer/G1」，
# 模型目录默认 `<g1_root>/models` ⇒ 这两项通常不用给；**人脸库必须给**，
# 不给就只剩检测/唇动/唤醒（身份识别静默关闭，启动日志里一句 warning）：
#
#     ORCH_FACE_DB="$SHARED/faceidentification/data/face_db.npz" \
#     ORCH_FACE_SERVICE_URL= bash orchestrator/run_orch_106.sh
#
# 要强制用某一份 `.so` 才设 `ORCH_FACE_SO`（provider 会**实测**每份候选 + 验
# ABI，坏的不会被静默采纳 —— 106 上有几份产物，**不能按目录名猜**）。

# ---- 声学延迟 D：占位 250 等于不生效，真值要跑 tests/measure_delay.py ----
export ORCH_AEC_DEFAULT_DELAY_MS="${ORCH_AEC_DEFAULT_DELAY_MS:-250}"

export ORCH_LOG_LEVEL="${ORCH_LOG_LEVEL:-info}"
export ORCH_DUMP_AUDIO="${ORCH_DUMP_AUDIO:-$SHARED/orchdump/s}"

# ⚠️ 编排服务**不读** ORCH_LOG_FILE —— 日志重定向必须在这里做（exec）。
#    日志名带端口，影子实例和生产不会混在一个文件里。
LOG_FILE="${ORCH_LOG_FILE:-$REPO/orch106-$PORT.log}"
exec >>"$LOG_FILE" 2>&1
echo "=== $(date "+%F %T") 启动（pid $$，port $PORT）—— 日志 $LOG_FILE ===" >&2

SSL_CERT="${ORCH_SSL_CERT:-$REPO/certs/cert.pem}"
SSL_KEY="${ORCH_SSL_KEY:-$REPO/certs/key.pem}"
SSL_ARGS=()
if [ -f "$SSL_CERT" ] && [ -f "$SSL_KEY" ]; then
  SSL_ARGS=(--ssl-cert "$SSL_CERT" --ssl-key "$SSL_KEY")
else
  echo "⚠️ 未找到证书（$SSL_CERT）—— 将以 HTTP 启动，别的机器访问时采不到音视频" >&2
fi

echo "106 编排：IC 模式 = $ORCH_IC_MODE" >&2
echo "  端口            = $PORT" >&2
if [ "$ORCH_IC_MODE" = "grpc" ]; then
  echo "  IC 服务         = $ORCH_IC_GRPC（独立进程，回退态；别停它）" >&2
  echo "  告诉 Agent 的 IC = $ORCH_IC_ADVERTISE（**必须是 :50051**）" >&2
else
  echo "  IC 服务         = 无（进程内 Engine，每会话一份）" >&2
  echo "  apply_agent 回调 = $ORCH_IC_CALLBACK_BASE" >&2
fi
echo "  Agent           = $ORCH_AGENT_URL（set_target=$ORCH_AGENT_SET_TARGET）" >&2
echo "  人脸            = ${ORCH_FACE_SERVICE_URL:-本地 .so（未配远端服务）}" >&2
echo "  python          = $PY" >&2
# VLM 描述（两阶段）生效值 —— **106 上这行必须显示「关 / legacy」**。
# 编排代码与 105 共用一份（这个脚本跑的也是同一个 main.py），所以
# `ORCH_OMNI_DESCRIBE` / `ORCH_AGENT_TRANSCRIPT_MODE` 的**默认值**就是
# 106 重启后的行为 —— 默认开 = 替 106 决定「给 102 的 Agent 多发一个
# content 字段」，而那个 Agent 还没改好（见 config.py 同名开关的说明）。
echo "  VLM 描述        = ${ORCH_OMNI_DESCRIBE:-0}（0=关；106 保持 0）" >&2
echo "  Agent 载荷      = ${ORCH_AGENT_TRANSCRIPT_MODE:-legacy}" \
     "（legacy=与旧版逐字节一致；106 保持 legacy）" >&2

exec "$PY" -u -m orchestrator.main \
  --host "${ORCH_HOST:-0.0.0.0}" --port "$PORT" \
  --downstream-mode "${ORCH_DOWNSTREAM_MODE:-omni}" \
  --log-level "$ORCH_LOG_LEVEL" "${SSL_ARGS[@]}"
