#!/bin/bash
# 105 生产编排（共享树版）—— 不再用 code-inprocess-105 副本。
#
# IC 与编排代码都走**共享树** `/data/megastore/Projects/DuJing/code`：
#   · 好处：IC 更新只需改共享树/同步，不再有「副本漏同步」的隐患
#   · 代价：105 与 106 共用同一份编排代码（106 默认 ORCH_IC_MODE 未设 → grpc，
#           行为不变；106 无 --reload、无 systemd，不重启就不会变）
#
# 进程内 Engine 靠 `interaction` 包的 py310/orch105 环境（editable 指向共享树）。
set -eu
SHARED=/data/megastore/Projects/DuJing/code
REPO=$SHARED/MiniCPM-o-Demo
cd "$REPO"

export ORCH_PY="${ORCH_PY:-/home/dujing/miniconda3/envs/orch105/bin/python}"
export ORCH_LOG_FILE="${ORCH_LOG_FILE:-$SHARED/orch-dumps/105/orch105.log}"
mkdir -p "$(dirname "$ORCH_LOG_FILE")"

# ---- IC：进程内 Engine，每会话一份 ----
export ORCH_IC_MODE="${ORCH_IC_MODE:-inprocess}"
export ORCH_IC_ADVERTISE="${ORCH_IC_ADVERTISE:-192.168.89.105:8100}"
export ORCH_IC_CALLBACK_BASE="${ORCH_IC_CALLBACK_BASE:-https://192.168.89.105:8100/v1/ic}"
export ORCH_IC_EXPRESSION_URL="${ORCH_IC_EXPRESSION_URL:-https://127.0.0.1:8100}"
export ORCH_IC_EXPRESSION_CA="${ORCH_IC_EXPRESSION_CA:-$REPO/certs/cert.pem}"
export ORCH_IC_GRPC="${ORCH_IC_GRPC:-127.0.0.1:50051}"
export ORCH_IC_RESTORE="${ORCH_IC_RESTORE:-192.168.89.105:50051}"

# ---- Agent：105 专用（103:8082，独立副本含多会话支持）----
export ORCH_AGENT_URL="${ORCH_AGENT_URL:-http://192.168.89.103:8082}"
export ORCH_AGENT_SET_TARGET="${ORCH_AGENT_SET_TARGET:-0}"

# ---- Omni（105 本地没有，指 106）----
export ORCH_OMNI_URL="${ORCH_OMNI_URL:-wss://192.168.89.106:8006/v1/realtime?mode=video}"

# ---- 人脸：远端服务，不加载 .so ----
export ORCH_ENABLE_FACE="${ORCH_ENABLE_FACE:-1}"
export ORCH_FACE_SERVICE_URL="${ORCH_FACE_SERVICE_URL:-http://192.168.89.105:8767}"
export ORCH_FACE_SERVICE_TIMEOUT_S="${ORCH_FACE_SERVICE_TIMEOUT_S:-2.0}"
export G1_FACE_DEBUG="${G1_FACE_DEBUG:-0}"

# ⚠️ 编排服务**不读** ORCH_LOG_FILE —— 日志重定向必须在脚本里做（exec），
#    早先只设了变量，结果 8100 的 stdout 全进了 /dev/null（排查时无日志可看）。
LOG_FILE="${ORCH_LOG_FILE:-/data/megastore/Projects/DuJing/code/MiniCPM-o-Demo/orch.log}"
if [ -n "$LOG_FILE" ]; then
  exec >>"$LOG_FILE" 2>&1
  echo "=== $(date "+%F %T") 启动（pid $$）—— 日志 $LOG_FILE ===" >&2
fi

export ORCH_LOG_LEVEL="${ORCH_LOG_LEVEL:-info}"
export ORCH_DUMP_AUDIO="${ORCH_DUMP_AUDIO:-$SHARED/orch-dumps/105/s}"

SSL_CERT="${ORCH_SSL_CERT:-$REPO/certs/cert.pem}"
SSL_KEY="${ORCH_SSL_KEY:-$REPO/certs/key.pem}"
SSL_ARGS=()
[ -f "$SSL_CERT" ] && [ -f "$SSL_KEY" ] && SSL_ARGS=(--ssl-cert "$SSL_CERT" --ssl-key "$SSL_KEY")

exec "$ORCH_PY" -u -m orchestrator.main \
  --host "${ORCH_HOST:-0.0.0.0}" --port "${ORCH_PORT:-8100}" \
  --downstream-mode "${ORCH_DOWNSTREAM_MODE:-omni}" \
  --log-level "$ORCH_LOG_LEVEL" "${SSL_ARGS[@]}"
