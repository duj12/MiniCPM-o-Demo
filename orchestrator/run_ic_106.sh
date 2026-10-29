#!/bin/bash
# InteractionCore gRPC 服务 —— **106 专属**。
#
# ⚠️ 这**不是** 106 的常规运行路径。106 的编排已改成进程内 IC
#    （`run_orch_106.sh`，`ORCH_IC_MODE=inprocess`），不需要这个服务。
#    它现在的唯一用途是**回退路径**：编排侧 `ORCH_IC_MODE=grpc`
#    （`run_orch_106.sh` 的一条分支）会用 `127.0.0.1:50051` 连它。
#    ⇒ **别停它**，除非你确定再也不回退了。
#
# ## 这行命令是"照抄"来的，不是想出来的
#
# 2026-10-08 从**正在运行的进程**（pid 1286618，2026-09-29 15:53 启动，至今
# 没重启过）的 `/proc/<pid>/cmdline` + `environ` 逐字抄下来。**没有实跑验证**
# —— 不想为了验一条回退路径去重启它。抄下来是为了让回退可复现：在此之前它
# 是一行手工敲的命令，没有任何地方记着，进程一死就重建不出来。
#
# ## 三个必须原样保留的东西（都踩过）
#
# 1. `--agent-url http://192.168.89.102:8081` = **106 的 Agent**，必须显式给。
#    `scripts/run_ic.sh` 里有硬编码默认值，而 105 正是栽在这里：手工起 IC 时
#    漏了 `--agent-url`，IC 判了 ANSWER 却把 Action 送到**另一台机器的 Agent**，
#    那边不认识这个会话 ⇒ **没有回复，且完全不报错**。见 `run_ic_105.sh`。
# 2. `--expression-url https://192.168.89.106:8100` + `--expression-ca <编排的
#    自签证书>` —— IC 的 GREET/UTTER 要打回编排的 `/v1/speak`。CA 不匹配时校验
#    失败**只记 IC 侧日志**，现象是「IC 判了 GREET 但没人播」，极难排查。
# 3. `--port 50051` —— 编排侧 `ORCH_IC_GRPC` 指向的就是它。
#
# ## Agent 会话信封（session_id / callback_ic）—— 现在**必须开**
#
# Agent 端已把这两个字段改**必填**（缺任一字段 → 事件被 422 拒）。
# 不开信封这条回退路径上 Agent 一条事件都收不到（没有回复、没有播报）。
# 所以下面 set 段必须配齐三项：
#
#     INTERACTION_AGENT_SESSION_ENVELOPE=1
#     INTERACTION_CALLBACK_IC=https://192.168.89.106:8100/v1/ic
#       （HTTP 形态 → Agent 走 HTTP 回写；编排侧 grpc 模式的
#        `POST /v1/ic/apply_agent` 会把它转成 gRPC 投给本服务）
#     INTERACTION_SESSION_ID=<固定标识> —— 独立 IC 没有会话概念，
#       空着 payload 的 session_id 就是 null，照样 422
#
# ⚠️ 前置：本机 interactioncore 的版本要支持这三个 env（Pite 主线已支持；
#    老版本没有任何配置能兼容必填后的 Agent —— 先升代码）。
set -eu
REPO=/data/megastore/Projects/DuJing/code/MiniCPM-o-Demo
PY="${IC_PY:-/home/dujing/miniconda3/envs/py310/bin/python}"
cd /data/megastore/Projects/DuJing/code/interactioncore

export INTERACTION_AGENT_URL="${INTERACTION_AGENT_URL:-http://192.168.89.102:8081}"
export INTERACTION_EXPRESSION_URL="${INTERACTION_EXPRESSION_URL:-https://192.168.89.106:8100}"
export INTERACTION_EXPRESSION_CA="${INTERACTION_EXPRESSION_CA:-$REPO/certs/cert.pem}"

# ⚠️ Agent 端已把 session_id / callback_ic 改**必填** —— 信封必须开，且
#    标识 / 地址配齐（原因与形态说明见文件头「Agent 会话信封」段）。
export INTERACTION_AGENT_SESSION_ENVELOPE="${INTERACTION_AGENT_SESSION_ENVELOPE:-1}"
export INTERACTION_CALLBACK_IC="${INTERACTION_CALLBACK_IC:-https://192.168.89.106:8100/v1/ic}"
export INTERACTION_SESSION_ID="${INTERACTION_SESSION_ID:-ic106-standalone}"

LOG="${IC106_LOG:-/data/megastore/Projects/DuJing/code/orch-dumps/106/ic106.log}"
mkdir -p "$(dirname "$LOG")"
exec >>"$LOG" 2>&1
echo "=== $(date '+%F %T') 启动 IC gRPC（pid $$，port ${IC_PORT:-50051}）—— 日志 $LOG ===" >&2

exec "$PY" -u -m interaction.grpc_server \
  --port "${IC_PORT:-50051}" \
  --expression-url "$INTERACTION_EXPRESSION_URL" \
  --expression-ca "$INTERACTION_EXPRESSION_CA" \
  --agent-url "$INTERACTION_AGENT_URL"
