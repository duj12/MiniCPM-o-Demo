#!/bin/bash
# 105 生产编排：每会话进程内 IC + 独立 Agent gRPC。
#
# 从哪个目录运行都可以；ORCH_PORT=8101 可启动影子实例。
# 更新代码后需重启服务；正在运行的进程不会自动重新导入。
# 重启时按监听端口确认 PID，再 kill -TERM <PID>；勿用 pkill -f 匹配共享命令行。
# IC 固定使用本仓库 vendor/interactioncore，不再启动独立 IC 服务。
# 先在仓库根目录初始化 submodule 并安装 orchestrator/requirements.txt。
set -eu
SHARED=/data/megastore/Projects/DuJing/code
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

export ORCH_PY="${ORCH_PY:-/home/dujing/miniconda3/envs/orch105/bin/python}"
if ! command -v "$ORCH_PY" >/dev/null 2>&1; then
  echo "找不到 python: $ORCH_PY（可用 ORCH_PY=... 覆盖）" >&2
  exit 1
fi
if [ ! -f "$REPO/vendor/interactioncore/interaction/runtime.py" ]; then
  echo "未初始化 InteractionCore：请先执行 git submodule update --init --recursive" >&2
  exit 1
fi
export PYTHONPATH="$REPO/vendor/interactioncore:$REPO${PYTHONPATH:+:$PYTHONPATH}"
# ⚠️ 环境里若装着旧的 editable `interaction`（老部署常见，指向共享树），
#    它的 `__editable__` finder 位于 sys.meta_path，**优先于**上面的
#    PYTHONPATH —— 不卸掉会静默跑成旧 IC。这里先拦下来。
IC_ORIGIN="$("$ORCH_PY" -c "import importlib.util as u; s=u.find_spec('interaction'); print(s.origin if s else '')" 2>/dev/null || true)"
case "$IC_ORIGIN" in
  "$REPO/vendor/interactioncore"/*|"") ;;
  *)
    echo "⚠️ 环境里有旧的 interaction 安装：$IC_ORIGIN" >&2
    echo "   它会盖过 $REPO/vendor/interactioncore；请先执行：" >&2
    echo "   $ORCH_PY -m pip uninstall -y interaction" >&2
    exit 1
    ;;
esac
PORT="${ORCH_PORT:-8100}"

# AgentService/Interact 是唯一 Agent 通道；每路会话建立独立双向 RPC。
# 与旧 IC gRPC 的 :50051 不同，这个端口现在属于独立 Agent 服务。
export ORCH_IC_MODE=inprocess
export ORCH_AGENT_GRPC="${ORCH_AGENT_GRPC:-192.168.89.103:50051}"
export ORCH_OMNI_URL="${ORCH_OMNI_URL:-wss://192.168.89.106:8006/v1/realtime?mode=video}"

# P05 barge 状态机的打断判据由 DecisionProbe 提供。
# DECISION_BASE_URL 未设时使用 IC 包默认 http://192.168.89.101:8094。
# ORCH_IC_DECISION_PROBE=0 仅用于排障：关闭后缺少探针判据。
export ORCH_IC_DECISION_PROBE="${ORCH_IC_DECISION_PROBE:-1}"
# 暂停/续播的浏览器音量渐变，0 表示关闭渐变。
export ORCH_TTS_PAUSE_FADE_MS="${ORCH_TTS_PAUSE_FADE_MS:-1000}"
export ORCH_TTS_RESUME_FADE_MS="${ORCH_TTS_RESUME_FADE_MS:-300}"

# 人脸走按 session 隔离的远端服务，避免本地 .so 的进程级状态串扰。
# 105/106 共用 105:8767 的服务额度；超时或额度打满会丢帧。
# 必须用 "-" 保留显式空值；ORCH_FACE_SERVICE_URL= 才会回到本地 .so。
# 105 当前没有可用本地 .so；置空前还需 ORCH_ENABLE_FACE=0，避免误以为能回退。
# 106 的本地 .so 退路仅适合单会话，见 run_orch_106.sh。
export ORCH_ENABLE_FACE="${ORCH_ENABLE_FACE:-1}"
export ORCH_FACE_SERVICE_URL="${ORCH_FACE_SERVICE_URL-http://192.168.89.105:8767}"
export ORCH_FACE_SERVICE_TIMEOUT_S="${ORCH_FACE_SERVICE_TIMEOUT_S:-2.0}"
export G1_FACE_DEBUG="${G1_FACE_DEBUG:-0}"

# 算法 AEC 的 D 必须按设备实测；250ms 只是占位，默认仍为浏览器 AEC。
export ORCH_AEC_DEFAULT_DELAY_MS="${ORCH_AEC_DEFAULT_DELAY_MS:-250}"
# 两阶段描述：起点全量 + 后续断句触发/心跳兜底；经 AgentAction.vlm 发送。
export ORCH_OMNI_DESCRIBE="${ORCH_OMNI_DESCRIBE:-1}"
# VLM 描述刷新间隔（秒）。主路径是 VAD 断句点触发（ORCH_OMNI_DESC_ON_TURN_BOUNDARY=1），
# 这个值管两件事：① 两次生成之间的最小间隔（不重叠，防止请求排队、延迟单调增长）
# ② 没人说话、也没有断句时的兜底心跳。调小不会让断句触发变快。
export ORCH_OMNI_DELTA_INTERVAL_S="${ORCH_OMNI_DELTA_INTERVAL_S:-10}"

# 保留各机现有日志与音视频排障素材的默认目录。
export ORCH_LOG_LEVEL="${ORCH_LOG_LEVEL:-info}"
export ORCH_DUMP_AUDIO="${ORCH_DUMP_AUDIO:-$SHARED/orch-dumps/105/s}"
export ORCH_LOG_FILE="${ORCH_LOG_FILE:-$SHARED/orch-dumps/105/orch105.log}"
LOG_FILE="$ORCH_LOG_FILE"
mkdir -p "$(dirname "$LOG_FILE")"
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

echo "105 编排：IC = 进程内 vendor/interactioncore（每会话一份 Engine）" >&2
echo "  端口            = $PORT" >&2
echo "  Agent gRPC      = $ORCH_AGENT_GRPC（每会话独立 Interact）" >&2
echo "  Omni            = $ORCH_OMNI_URL" >&2
echo "  人脸            = ${ORCH_FACE_SERVICE_URL:-本地 .so（仅单会话）}" >&2
echo "  python          = $ORCH_PY" >&2
echo "  音视频转储      = $ORCH_DUMP_AUDIO" >&2
echo "  VLM 描述        = $ORCH_OMNI_DESCRIBE（0=关，omni 当对话方）" >&2
echo "  描述阶段        = 起点 full + 每轮 turn，AgentAction.vlm 单独传递" >&2
echo "  描述刷新间隔    = ${ORCH_OMNI_DELTA_INTERVAL_S}s（断句点触发为主，此为最小间隔/静默兜底）" >&2
echo "  暂停/续播渐变   = 淡出 ${ORCH_TTS_PAUSE_FADE_MS}ms / 淡入 ${ORCH_TTS_RESUME_FADE_MS}ms" >&2
echo "  打断探针        = $ORCH_IC_DECISION_PROBE" >&2
echo "  HTTPS 证书      = $SSL_CERT" >&2

exec "$ORCH_PY" -u -m orchestrator.main \
  --host "${ORCH_HOST:-0.0.0.0}" --port "$PORT" \
  --downstream-mode "${ORCH_DOWNSTREAM_MODE:-omni}" \
  --log-level "$ORCH_LOG_LEVEL" "${SSL_ARGS[@]}"
