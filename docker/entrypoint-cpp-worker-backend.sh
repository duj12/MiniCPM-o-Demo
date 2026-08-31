#!/usr/bin/env bash
set -euo pipefail

# Active model: "minicpm" (default, MiniCPM-o-4.5) or "qwen3omni" (Qwen3-Omni)
ACTIVE_MODEL="${ACTIVE_MODEL:-minicpm}"

# Backend ports and host
BACKEND_BIND_HOST="${BACKEND_BIND_HOST:-127.0.0.1}"
BACKEND_PORT="${BACKEND_PORT:-22500}"
WORKER_PORT="${WORKER_PORT:-22400}"
GPU_ID="${GPU_ID:-0}"
READY_TIMEOUT_S="${READY_TIMEOUT_S:-1200}"
BACKEND_URL="http://127.0.0.1:${BACKEND_PORT}"
LOG_DIR="${LOG_DIR:-/app/logs}"

cd /app
mkdir -p "$LOG_DIR"

require_path() {
    local path="$1"
    if [ ! -e "$path" ]; then
        echo "[entrypoint] missing required path: $path" >&2
        exit 1
    fi
}

cleanup() {
    echo "[entrypoint] stopping child processes..."
    wait 2>/dev/null || true
    exit 0
}
trap cleanup SIGTERM SIGINT

echo "=================================================="
echo "  C++ worker-backend bundle"
echo "  active_model  = ${ACTIVE_MODEL}"
echo "  backend       = ${BACKEND_BIND_HOST}:${BACKEND_PORT}"
echo "  worker        = 0.0.0.0:${WORKER_PORT} -> ${BACKEND_URL}"
echo "  gpu-id        = ${GPU_ID}"
echo "=================================================="

# ============================================================================
# MiniCPM-o-4.5 mode (default, full-duplex omni)
# ============================================================================
if [ "$ACTIVE_MODEL" = "minicpm" ]; then

LLAMA_SERVER_BIN="${LLAMA_SERVER_BIN:-/opt/llama.cpp-omni/bin/llama-omni-server}"
GGUF_MODEL="${GGUF_MODEL:-/models/MiniCPM-o-4_5-gguf/MiniCPM-o-4_5-Q4_K_M.gguf}"
N_GPU_LAYERS="${N_GPU_LAYERS:-99}"
LLAMA_LOG_FILE="${LLAMA_LOG_FILE:-${LOG_DIR}/llama-omni-server.log}"

require_path "$LLAMA_SERVER_BIN"
require_path "$GGUF_MODEL"

if [ "${CHECK_MODEL_LAYOUT:-1}" = "1" ]; then
    model_root="$(dirname "$GGUF_MODEL")"
    require_path "${model_root}/vision/MiniCPM-o-4_5-vision-F16.gguf"
    require_path "${model_root}/audio/MiniCPM-o-4_5-audio-F16.gguf"
    require_path "${model_root}/tts/MiniCPM-o-4_5-tts-F16.gguf"
    require_path "${model_root}/tts/MiniCPM-o-4_5-projector-F16.gguf"
    require_path "${model_root}/token2wav-gguf"
fi

echo "  llama server = $LLAMA_SERVER_BIN"
echo "  GGUF_MODEL   = $GGUF_MODEL"
echo "  n-gpu-layers = ${N_GPU_LAYERS}"
echo "=================================================="

: > "$LLAMA_LOG_FILE"
# 转发 backend 日志到 stdout，过滤调试噪声，突出关键事件。
# 丢弃: preprocess/[prof]/tensor 等；保留: session/duplex/decode/error/omni。
llama_args=(
    -m "$GGUF_MODEL"
    -ngl "$N_GPU_LAYERS"
    --host "$BACKEND_BIND_HOST"
    --port "$BACKEND_PORT"
)

if [ -n "${LLAMA_SERVER_EXTRA_ARGS:-}" ]; then
    # shellcheck disable=SC2206
    extra_args=( $LLAMA_SERVER_EXTRA_ARGS )
    llama_args+=( "${extra_args[@]}" )
fi

# 注意：llama-server 不再在此后台启动——由 supervisord 管理（见文件结尾）。
# 仅记录启动参数供 supervisor 配置使用；backend 健康检查交给 worker 的前置脚本。

# ============================================================================
# Qwen3-Omni mode (turn-based, text-only or multimodal via mmproj)
# ============================================================================
elif [ "$ACTIVE_MODEL" = "qwen3omni" ]; then

LLAMA_SERVER_BIN="${LLAMA_QWEN3_SERVER_BIN:-/opt/llama.cpp-omni/bin/llama-qwen3omni-server}"
GGUF_MODEL="${GGUF_MODEL:-/models/qwen3omni-gguf/Qwen3-Omni-30B-A3B-Instruct-Q4_K_S.gguf}"
MMPROJ_MODEL="${MMPROJ_MODEL:-/models/qwen3omni-gguf/mmproj-Qwen3-Omni-30B-A3B-Instruct-Q8_0.gguf}"
N_GPU_LAYERS="${N_GPU_LAYERS:-99}"
LLAMA_LOG_FILE="${LLAMA_LOG_FILE:-${LOG_DIR}/llama-qwen3omni-server.log}"

require_path "$LLAMA_SERVER_BIN"
require_path "$GGUF_MODEL"
require_path "$MMPROJ_MODEL"

echo "  llama server = $LLAMA_SERVER_BIN"
echo "  GGUF_MODEL   = $GGUF_MODEL"
echo "  MMPROJ_MODEL = $MMPROJ_MODEL"
echo "  n-gpu-layers = ${N_GPU_LAYERS}"
echo "=================================================="

: > "$LLAMA_LOG_FILE"
# 转发 backend 日志到 stdout，过滤调试噪声，突出关键事件。
# 丢弃: preprocess/[prof]/tensor 等；保留: session/duplex/decode/error/omni。
llama_args=(
    -m "$GGUF_MODEL"
    --mmproj "$MMPROJ_MODEL"
    -ngl "$N_GPU_LAYERS"
    --host "$BACKEND_BIND_HOST"
    --port "$BACKEND_PORT"
)

if [ -n "${LLAMA_SERVER_EXTRA_ARGS:-}" ]; then
    extra_args=( $LLAMA_SERVER_EXTRA_ARGS )
    llama_args+=( "${extra_args[@]}" )
fi

# 注意：llama-server 不再在此后台启动——由 supervisord 管理（见文件结尾）。
# 参数已在上面构造好，供 supervisor 配置使用；backend 健康检查交给 worker 的前置脚本。

else
    echo "[entrypoint] unknown ACTIVE_MODEL: ${ACTIVE_MODEL} (expected minicpm or qwen3omni)" >&2
    exit 1
fi

# ============================================================================
# 生成 supervisord.conf 并 exec supervisord 管理两个服务
#   - llama-server: C++ 推理后端（参数已按 ACTIVE_MODEL 构造）
#   - worker:       Python 转发（--backend-server-url 指向本容器 llama）
# supervisor 负责 autorestart + 日志落位 /app/logs；容器 HEALTHCHECK 走 worker /health。
# ============================================================================

# worker 启动前等 backend ready（llama-server 由 supervisor 拉起，可能稍后）
WAIT_BACKEND_SH="/app/docker/wait-backend.sh"
cat > "$WAIT_BACKEND_SH" <<EOF
#!/bin/bash
for i in \$(seq 1 $((READY_TIMEOUT_S / 2))); do
    if curl -sf "${BACKEND_URL}/health" >/dev/null 2>&1; then
        echo "[wait-backend] backend ready after ~\$((i * 2))s"
        exit 0
    fi
    sleep 2
done
echo "[wait-backend] backend not ready within ${READY_TIMEOUT_S}s" >&2
exit 1
EOF
chmod +x "$WAIT_BACKEND_SH"

cat > /etc/supervisor/conf.d/supervisord.conf <<EOF
[unix_http_server]
file=/var/run/supervisor.sock
chmod=0700

[supervisord]
nodaemon=true
logfile=/var/log/supervisor/supervisord.log
pidfile=/var/run/supervisord.pid
childlogdir=/var/log/supervisor

[rpcinterface:supervisor]
supervisor.rpcinterface_factory = supervisor.rpcinterface:make_main_rpcinterface

[supervisorctl]
serverurl=unix:///var/run/supervisor.sock

[program:llama-server]
command=${LLAMA_SERVER_BIN} ${llama_args[*]}
autostart=true
autorestart=true
startretries=5
priority=10
stdout_logfile=${LLAMA_LOG_FILE}
stdout_logfile_maxbytes=100MB
stdout_logfile_backups=3
stderr_logfile=${LLAMA_LOG_FILE}
stderr_logfile_maxbytes=100MB
stderr_logfile_backups=3

[program:worker]
command=/bin/bash -c "${WAIT_BACKEND_SH} && LLAMA_SERVER_BIN=${LLAMA_SERVER_BIN} ACTIVE_MODEL=${ACTIVE_MODEL} python worker.py --host 0.0.0.0 --port ${WORKER_PORT} --gpu-id ${GPU_ID} --backend-server-url ${BACKEND_URL}"
autostart=true
autorestart=true
startretries=5
priority=20
stdout_logfile=${LOG_DIR}/worker.log
stdout_logfile_maxbytes=100MB
stdout_logfile_backups=3
stderr_logfile=${LOG_DIR}/worker.log
stderr_logfile_maxbytes=100MB
stderr_logfile_backups=3
EOF

echo "[entrypoint] exec supervisord (llama-server + worker)"
exec /usr/bin/supervisord -c /etc/supervisor/conf.d/supervisord.conf
