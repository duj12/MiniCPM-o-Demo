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
    [ -n "$worker_pid" ] && kill "$worker_pid" 2>/dev/null || true
    [ -n "$backend_pid" ] && kill "$backend_pid" 2>/dev/null || true
    [ -n "$tail_pid" ] && kill "$tail_pid" 2>/dev/null || true
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
tail -n +1 -F "$LLAMA_LOG_FILE" | \
    grep -vE "audition_audio_preprocess|vision_|clip_model_loader|load_tensors|\[prof\]|KV cache iter|Before incrementing|Final output|mel spectrogram|build_whisper|conv2d|tensor\[|audio slice|audio decoded" &
tail_pid=$!

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

echo "[entrypoint] starting llama-omni-server..."
"$LLAMA_SERVER_BIN" "${llama_args[@]}" >> "$LLAMA_LOG_FILE" 2>&1 &
backend_pid=$!

echo "[entrypoint] waiting for backend /health..."
max_retries=$((READY_TIMEOUT_S / 2))
if [ "$max_retries" -lt 1 ]; then max_retries=1; fi

for i in $(seq 1 "$max_retries"); do
    if ! kill -0 "$backend_pid" 2>/dev/null; then
        echo "[entrypoint] llama server exited while loading" >&2
        tail -50 "$LLAMA_LOG_FILE" >&2 || true
        cleanup
    fi
    if curl -sf "${BACKEND_URL}/health" >/dev/null 2>&1; then
        echo "[entrypoint] backend ready after ~$((i * 2))s"
        break
    fi
    if [ "$i" -eq "$max_retries" ]; then
        echo "[entrypoint] backend did not become ready within ${READY_TIMEOUT_S}s" >&2
        tail -80 "$LLAMA_LOG_FILE" >&2 || true
        cleanup
    fi
    sleep 2
done

# Preload omni models into VRAM
echo "[entrypoint] preloading omni models into VRAM..."
if curl -sf -X POST "${BACKEND_URL}/v1/stream/omni_init" \
    -H "Content-Type: application/json" \
    -d '{"msg_type":2,"use_tts":true}' >/dev/null 2>&1; then
    echo "[entrypoint] omni models loaded and pinned in VRAM"
else
    echo "[entrypoint] omni preload failed (will load on first request)" >&2
fi

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
tail -n +1 -F "$LLAMA_LOG_FILE" | \
    grep -vE "audition_audio_preprocess|vision_|clip_model_loader|load_tensors|\[prof\]|KV cache iter|Before incrementing|Final output|mel spectrogram|build_whisper|conv2d|tensor\[|audio slice|audio decoded" &
tail_pid=$!

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

echo "[entrypoint] starting llama-qwen3omni-server..."
"$LLAMA_SERVER_BIN" "${llama_args[@]}" >> "$LLAMA_LOG_FILE" 2>&1 &
backend_pid=$!

echo "[entrypoint] waiting for backend /health..."
max_retries=$((READY_TIMEOUT_S / 2))
if [ "$max_retries" -lt 1 ]; then max_retries=1; fi

for i in $(seq 1 "$max_retries"); do
    if ! kill -0 "$backend_pid" 2>/dev/null; then
        echo "[entrypoint] llama server exited while loading" >&2
        tail -50 "$LLAMA_LOG_FILE" >&2 || true
        cleanup
    fi
    if curl -sf "${BACKEND_URL}/health" >/dev/null 2>&1; then
        echo "[entrypoint] backend ready after ~$((i * 2))s"
        break
    fi
    if [ "$i" -eq "$max_retries" ]; then
        echo "[entrypoint] backend did not become ready within ${READY_TIMEOUT_S}s" >&2
        tail -80 "$LLAMA_LOG_FILE" >&2 || true
        cleanup
    fi
    sleep 2
done

echo "[entrypoint] Qwen3-Omni backend ready (no omni preload needed)"

else
    echo "[entrypoint] unknown ACTIVE_MODEL: ${ACTIVE_MODEL} (expected minicpm or qwen3omni)" >&2
    exit 1
fi

# ============================================================================
# Start Python worker (shared for both modes)
# ============================================================================
echo "[entrypoint] starting worker..."
LLAMA_SERVER_BIN="$LLAMA_SERVER_BIN" \
ACTIVE_MODEL="$ACTIVE_MODEL" \
python worker.py \
    --host 0.0.0.0 \
    --port "$WORKER_PORT" \
    --gpu-id "$GPU_ID" \
    --backend-server-url "$BACKEND_URL" &
worker_pid=$!

sleep 3
if curl -sf "http://127.0.0.1:${WORKER_PORT}/health" >/dev/null 2>&1; then
    echo "[entrypoint] worker ready"
else
    echo "[entrypoint] worker health check is not ready yet"
fi

echo "[entrypoint] running. backend pid=${backend_pid} worker pid=${worker_pid}"

while true; do
    if ! kill -0 "$backend_pid" 2>/dev/null; then
        echo "[entrypoint] llama server exited" >&2
        cleanup
    fi
    if ! kill -0 "$worker_pid" 2>/dev/null; then
        echo "[entrypoint] worker exited" >&2
        cleanup
    fi
    sleep 5
done
