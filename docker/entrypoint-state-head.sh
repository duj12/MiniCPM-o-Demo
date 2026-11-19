#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# state head 容器入口
#   - flide serve × STATE_REPLICAS   （127.0.0.1:22521,22522…，真正推理）
#   - uvicorn  state_head.app:app    （0.0.0.0:22520，异步门面）
# 两者都由 supervisord 管理（autorestart + 日志落 /app/logs，与 cpp 容器一致）。
# ============================================================================

STATE_MODEL_DIR="${STATE_MODEL_DIR:-onejev-1.6-0.8b}"
STATE_REPLICAS="${STATE_REPLICAS:-1}"
STATE_HEAD_PORT="${STATE_HEAD_PORT:-22520}"
STATE_FLIDE_BASE_PORT="${STATE_FLIDE_BASE_PORT:-22521}"
STATE_BATCH_WINDOW_MS="${STATE_BATCH_WINDOW_MS:-3}"
STATE_GATHER_MS="${STATE_GATHER_MS:-2}"
STATE_PLANNERS="${STATE_PLANNERS:-4}"
STATE_FRONTENDS="${STATE_FRONTENDS:-2}"
STATE_NO_CUDA_GRAPHS="${STATE_NO_CUDA_GRAPHS:-1}"
STATE_MAX_MEDIA_MB="${STATE_MAX_MEDIA_MB:-16}"
STATE_MAX_REQUEST_TOKENS="${STATE_MAX_REQUEST_TOKENS:-16384}"
STATE_DEVICE="${STATE_DEVICE:-cuda:0}"
# 0 = 所有副本共用同一张卡（默认：与 omni 共卡，compose 只 pin 了 GPU0）；
# 1 = 每个副本摊到不同卡（要求 compose 放开多张卡）。
STATE_REPLICA_SPREAD="${STATE_REPLICA_SPREAD:-0}"
LOG_DIR="${LOG_DIR:-/app/logs}"

cd /app
mkdir -p "$LOG_DIR"

# 模型根：优先镜像内 /app/checkpoints（build-image.sh 打包进来），
# 否则用运行时挂载的 /models。与 entrypoint-cpp-worker-backend.sh 同一口径。
if [ -e "/app/checkpoints/${STATE_MODEL_DIR}" ]; then
    MODEL_ROOT=/app/checkpoints
else
    MODEL_ROOT=/models
fi
MODEL_PATH="${MODEL_ROOT}/${STATE_MODEL_DIR}"
FLIDE_LOG="${LOG_DIR}/flide.log"
HEAD_LOG="${LOG_DIR}/state-head.log"

echo "=================================================="
echo "  state head"
echo "  model      = ${MODEL_PATH}"
echo "  replicas   = ${STATE_REPLICAS} (ports ${STATE_FLIDE_BASE_PORT}+i)"
echo "  facade     = 0.0.0.0:${STATE_HEAD_PORT}"
echo "  batch      = window ${STATE_BATCH_WINDOW_MS}ms / gather ${STATE_GATHER_MS}ms" \
     "/ planners ${STATE_PLANNERS} / frontends ${STATE_FRONTENDS}"
echo "  cuda graphs= $([ "$STATE_NO_CUDA_GRAPHS" = "1" ] && echo off || echo on)" \
     " / device ${STATE_DEVICE}"
echo "=================================================="

if [ ! -e "$MODEL_PATH" ]; then
    # 不 exit：容器照样起来，/health 会明确告诉调用方模型缺失，
    # 编排侧 probe 失败自动退回纯文本 E0，不阻塞其它镜像发布。
    echo "[entrypoint] ⚠️ 模型目录不存在: $MODEL_PATH" >&2
    echo "[entrypoint]    flide 会反复重启；/health 将返回 ok=false" >&2
fi

# ---- 生成 flide 副本命令 + 汇聚 STATE_BACKEND_URLS -------------------------
GRAPH_FLAG="on"
[ "$STATE_NO_CUDA_GRAPHS" = "1" ] && GRAPH_FLAG="off"

flide_programs=""
backend_urls=""
for ((i = 0; i < STATE_REPLICAS; i++)); do
    port=$((STATE_FLIDE_BASE_PORT + i))
    # ⚠️ 默认**不设** CUDA_VISIBLE_DEVICES：compose 已经用 device_ids 把容器
    #    钉在 GPU0 上，容器内只有一张卡；这时再写 CUDA_VISIBLE_DEVICES=1 会让
    #    CUDA 看不到任何设备、flide 直接崩。只有 STATE_REPLICA_SPREAD=1
    #    （显式放开多卡）时才按副本号切卡。
    if [ "$STATE_REPLICA_SPREAD" = "1" ]; then
        replica_env="environment=CUDA_VISIBLE_DEVICES=${i},HF_HUB_OFFLINE=1,TRANSFORMERS_OFFLINE=1"
    else
        replica_env="environment=HF_HUB_OFFLINE=1,TRANSFORMERS_OFFLINE=1"
    fi
    flide_programs+="
[program:flide-${i}]
command=/opt/venv/bin/flide serve --model ${MODEL_PATH} --device ${STATE_DEVICE} \\
    --host 127.0.0.1 --port ${port} \\
    --batch-window-ms ${STATE_BATCH_WINDOW_MS} --gather-ms ${STATE_GATHER_MS} \\
    --planners ${STATE_PLANNERS} --frontends ${STATE_FRONTENDS} \\
    --no-cuda-graphs ${GRAPH_FLAG} \\
    --max-media-mb ${STATE_MAX_MEDIA_MB} \\
    --max-request-tokens ${STATE_MAX_REQUEST_TOKENS}
${replica_env}
autostart=true
autorestart=true
startretries=5
startsecs=10
priority=10
stdout_logfile=${FLIDE_LOG}
stdout_logfile_maxbytes=100MB
stdout_logfile_backups=3
stderr_logfile=${FLIDE_LOG}
stderr_logfile_maxbytes=100MB
stderr_logfile_backups=3
"
    backend_urls+="${backend_urls:+,}http://127.0.0.1:${port}"
done

export STATE_BACKEND_URLS="${STATE_BACKEND_URLS:-$backend_urls}"
echo "[entrypoint] STATE_BACKEND_URLS=${STATE_BACKEND_URLS}"

# ---- supervisor ------------------------------------------------------------
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
${flide_programs}
[program:state-head]
command=/opt/venv/bin/python -m uvicorn state_head.app:app \\
    --host 0.0.0.0 --port ${STATE_HEAD_PORT} --log-level info
environment=STATE_BACKEND_URLS="${STATE_BACKEND_URLS}",STATE_MODEL_DIR="${STATE_MODEL_DIR}",HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1"
autostart=true
autorestart=true
startretries=5
startsecs=5
priority=20
stdout_logfile=${HEAD_LOG}
stdout_logfile_maxbytes=50MB
stdout_logfile_backups=3
stderr_logfile=${HEAD_LOG}
stderr_logfile_maxbytes=50MB
stderr_logfile_backups=3
EOF

echo "[entrypoint] exec supervisord (flide×${STATE_REPLICAS} + state-head)"
exec /usr/bin/supervisord -c /etc/supervisor/conf.d/supervisord.conf
