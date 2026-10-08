#!/bin/bash
# InteractionCore gRPC 服务 —— **106 专属**。
#
# ⚠️ 这**不是** 106 的常规运行路径。106 的编排已改成进程内 IC
#    （`run_orch_106.sh`，`ORCH_IC_MODE=inprocess`），不需要这个服务。
#    它现在的唯一用途是**回退路径**：`run_orch.sh`（grpc 模式）会用
#    `ORCH_IC_GRPC=localhost:50051` 连它。
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
# ## 回退后若发现 `agent.status` 写不进去
#
# 说明 Agent 已经不认 gRPC 的 `ApplyAgent` 了（只认 HTTP 回调）。那时补上：
#
#     --agent-envelope --callback-ic https://192.168.89.106:8100/v1/ic
#
# 让 IC 在 Action payload 里告诉 Agent 往哪写（不加的话 Agent 拿不到
# `callback_ic`，只能退回它自己配置的地址）。编排侧 grpc 模式的
# `POST /v1/ic/apply_agent` 会把它转成 gRPC 投给本服务。
#
# ⚠️ 当前（抄下来的）那份**没有**加这两个参数 —— 保持与"今天之前一直在跑的
#    状态"一致，别顺手加：改了协议就同时影响 Agent 和回退行为。
set -eu
REPO=/data/megastore/Projects/DuJing/code/MiniCPM-o-Demo
PY="${IC_PY:-/home/dujing/miniconda3/envs/py310/bin/python}"
cd /data/megastore/Projects/DuJing/code/interactioncore

export INTERACTION_AGENT_URL="${INTERACTION_AGENT_URL:-http://192.168.89.102:8081}"
export INTERACTION_EXPRESSION_URL="${INTERACTION_EXPRESSION_URL:-https://192.168.89.106:8100}"
export INTERACTION_EXPRESSION_CA="${INTERACTION_EXPRESSION_CA:-$REPO/certs/cert.pem}"
export INTERACTION_SESSION_ID="${INTERACTION_SESSION_ID:-}"

LOG="${IC106_LOG:-/data/megastore/Projects/DuJing/code/orch-dumps/106/ic106.log}"
mkdir -p "$(dirname "$LOG")"
exec >>"$LOG" 2>&1
echo "=== $(date '+%F %T') 启动 IC gRPC（pid $$，port ${IC_PORT:-50051}）—— 日志 $LOG ===" >&2

exec "$PY" -u -m interaction.grpc_server \
  --port "${IC_PORT:-50051}" \
  --expression-url "$INTERACTION_EXPRESSION_URL" \
  --expression-ca "$INTERACTION_EXPRESSION_CA" \
  --agent-url "$INTERACTION_AGENT_URL"
