#!/bin/bash
# 106 生产编排：每会话进程内 IC + 独立 Agent gRPC。
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

PY="${ORCH_PY:-/home/dujing/miniconda3/envs/py310/bin/python}"
if ! command -v "$PY" >/dev/null 2>&1; then
  echo "找不到 python: $PY（可用 ORCH_PY=... 覆盖）" >&2
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
IC_ORIGIN="$("$PY" -c "import importlib.util as u; s=u.find_spec('interaction'); print(s.origin if s else '')" 2>/dev/null || true)"
case "$IC_ORIGIN" in
  "$REPO/vendor/interactioncore"/*|"") ;;
  *)
    echo "⚠️ 环境里有旧的 interaction 安装：$IC_ORIGIN" >&2
    echo "   它会盖过 $REPO/vendor/interactioncore；请先执行：" >&2
    echo "   $PY -m pip uninstall -y interaction" >&2
    exit 1
    ;;
esac
PORT="${ORCH_PORT:-8100}"

# AgentService/Interact 是唯一 Agent 通道；每路会话建立独立双向 RPC。
# 与旧 IC gRPC 的 :50051 不同，这个端口现在属于独立 Agent 服务。
export ORCH_IC_MODE=inprocess
export ORCH_AGENT_GRPC="${ORCH_AGENT_GRPC:-192.168.89.102:50051}"
export ORCH_OMNI_URL="${ORCH_OMNI_URL:-wss://127.0.0.1:8006/v1/realtime?mode=video}"

# P05 barge 状态机的打断判据由 DecisionProbe 提供。
# DECISION_BASE_URL 未设时使用 IC 包默认 http://192.168.89.101:8094。
# ORCH_IC_DECISION_PROBE=0 仅用于排障：关闭后缺少探针判据。
export ORCH_IC_DECISION_PROBE="${ORCH_IC_DECISION_PROBE:-1}"
# 暂停/续播的浏览器音量渐变，0 表示关闭渐变。
export ORCH_TTS_PAUSE_FADE_MS="${ORCH_TTS_PAUSE_FADE_MS:-1000}"
export ORCH_TTS_RESUME_FADE_MS="${ORCH_TTS_RESUME_FADE_MS:-300}"

# 人脸走按 session 隔离的远端服务，避免本地 .so 的进程级状态串扰。
# 服务已迁到 106:8767（105 那台已坏）；超时或额度打满会丢帧。
# 必须用 "-" 保留显式空值；ORCH_FACE_SERVICE_URL= 才会回到本地 .so。
# 本地 .so 仅适合单会话；需要时同时指定 ORCH_FACE_DB / ORCH_FACE_MODELS。
export ORCH_ENABLE_FACE="${ORCH_ENABLE_FACE:-1}"
export ORCH_FACE_SERVICE_URL="${ORCH_FACE_SERVICE_URL-http://192.168.89.106:8767}"
export ORCH_FACE_SERVICE_TIMEOUT_S="${ORCH_FACE_SERVICE_TIMEOUT_S:-2.0}"
# 8PCM 门控（8768 变体）：每次 detect 必须带客户端送来的 8 声道 PCM。
# 客户端音频是 100ms 一块，但**不能**因此把 detect 也压到 10Hz —— 编排
# 每帧从缓冲里切 40ms（640 采样）发一次，视频 25fps ⇒ 人脸 25Hz，
# 同时保持「相邻 PCM 首尾相接」。开的时候 URL 必须指 8768。
#   8767 = 只看图片帧（默认）；8768 = 图片 + 连续 8 声道 PCM 门控。
export ORCH_FACE_PCM8="${ORCH_FACE_PCM8:-0}"
export G1_FACE_DEBUG="${G1_FACE_DEBUG:-0}"

# 算法 AEC 的 D 必须按设备实测；250ms 只是占位，默认仍为浏览器 AEC。
export ORCH_AEC_DEFAULT_DELAY_MS="${ORCH_AEC_DEFAULT_DELAY_MS:-250}"
# ASR 新语音感知（语种/情绪/事件/噪声/交叠/SNR）总开关。
# 0 = 服务端不跑感知推理，state 回到原来的 5 个基础字段（省算力）。
export ORCH_ASR_PERCEPTION="${ORCH_ASR_PERCEPTION:-1}"
# 视觉对 Omni 的开关（1=按 1s 推帧，0=**一个视频帧都不发**给 Omni）。
# 纯语音策略设 0：音频照旧推给 Omni（仍可当听觉理解方），只是不发视频。
# ⚠️ 与 ORCH_OMNI_DESCRIBE=1 冲突：不发帧 ⇒ 帧新鲜度闸永久压住描述。
export ORCH_OMNI_VISION="${ORCH_OMNI_VISION:-1}"
# 两阶段描述：起点全量 + 后续断句触发 / 定时触发；经 AgentAction.vlm 发送。
export ORCH_OMNI_DESCRIBE="${ORCH_OMNI_DESCRIBE:-1}"
# state head（带置信度的 E2 状态）—— **旁挂容器** omnillm-state-head:22520。
#   0（默认）= 完全不调，行为与今天逐字节一致（纯文本 E0）。
#   1        = 调；schema §3 的枚举槽位全部由它答（独立 1s 节拍），
#              文本只出两行描述。见下方 ORCH_TEXT_DESC_ONLY。
# 打开前先确认那边确实活着：curl -s http://<host>:22520/health
export ORCH_STATE_HEAD="${ORCH_STATE_HEAD:-0}"
# desc-only 模式下文本只是"人 / 环境"两行，10s 心跳足够；没开
# state head 时保持 1s（老行为：文本自己答全量 8 键）。
if [ "$ORCH_STATE_HEAD" = "1" ]; then
  _DEF_DELTA_INTERVAL=10; _DEF_DELTA_ALWAYS=0
else
  _DEF_DELTA_INTERVAL=1;  _DEF_DELTA_ALWAYS=1
fi
# VLM 描述定时触发间隔（秒）。主路径是 VAD 断句点触发
# （ORCH_OMNI_DESC_ON_TURN_BOUNDARY=1），这个值管两件事：
# ① 两次生成之间的最小间隔（不重叠，防止请求排队、延迟单调增长）
# ② 定时触发周期 —— 从上一轮完成起算。调小不会让断句触发变快。
export ORCH_OMNI_DELTA_INTERVAL_S="${ORCH_OMNI_DELTA_INTERVAL_S:-$_DEF_DELTA_INTERVAL}"
# 定时触发是否**不看语音活动**（1=开）：用户正在说话也按上面的间隔准时
# 触发一次（与断句点触发叠加）。设 0 只在静默时补心跳 —— desc-only 模式的
# 默认（文本不再负责动作/姿态，说话期间不需要额外 decode）。
export ORCH_OMNI_DELTA_ALWAYS="${ORCH_OMNI_DELTA_ALWAYS:-$_DEF_DELTA_ALWAYS}"

export ORCH_STATE_URL="${ORCH_STATE_URL:-http://127.0.0.1:22520}"
# Jev 的**独立定时器**（秒；0 = 关，退回"跟随文本触发"的老行为）。
# 帧历史本身就是 1Hz（帧跟着 1s 音频块发），1s 已是上限。
export ORCH_STATE_INTERVAL_S="${ORCH_STATE_INTERVAL_S:-1}"
# 每拍请求的 stage：full 才含 env.scene（schema §3 的 ENV.scene 唯一入口）。
export ORCH_STATE_STAGE="${ORCH_STATE_STAGE:-full}"
# 超时即丢弃本轮 state（文本那条不受影响）。0.8s ≈ 目标 P95 的 1.6 倍。
export ORCH_STATE_TIMEOUT_S="${ORCH_STATE_TIMEOUT_S:-0.8}"
# 送给 state head 的帧数（2 = 当前 + 1s 前，显著改善动作/运动/duration）
export ORCH_STATE_N_FRAMES="${ORCH_STATE_N_FRAMES:-2}"
# 发前把帧长边缩到多少像素（显存/延迟的主要旋钮之一）
export ORCH_STATE_MAX_PX="${ORCH_STATE_MAX_PX:-640}"
# 渲染成 `label(0.82)` 形态（0 = 只出标签）
export ORCH_STATE_CONF_IN_TEXT="${ORCH_STATE_CONF_IN_TEXT:-1}"
# Qwen3-Omni 只出自由文本槽位（FEATURE / ENV / POSITION + 一行 DESCRIPTION）。
# **仅在 ORCH_STATE_HEAD=1 时生效** —— 没开 state head 还砍掉 5 个枚举键
# 就什么都不剩了，所以代码里两个条件是"与"关系。
export ORCH_TEXT_SHORT="${ORCH_TEXT_SHORT:-1}"
# **文本只出两行描述**（FEATURE / ENV，每行 ≤10 字）：schema §3 的
# 枚举槽位**全部**交给 state head。同样要求 ORCH_STATE_HEAD=1，且
# ORCH_STATE_SLOTS 必须为空（代码里有护栏，配错会告警并按全量处理）。
export ORCH_TEXT_DESC_ONLY="${ORCH_TEXT_DESC_ONLY:-1}"
# desc-only 的硬截断（两行关键词约 20~30 token）。
export ORCH_OMNI_DESC_ONLY_MAX_NEW_TOKENS="${ORCH_OMNI_DESC_ONLY_MAX_NEW_TOKENS:-32}"
# 槽位白名单：**默认空 = 问全套 16 题**（schema 对齐后的形态）。
# 空 / all / * = 全量；非空只作回滚用，例如
#   ORCH_STATE_SLOTS=attn.gaze,beh.activity,rel.person_count,rel.interaction
# ⚠️ 必须与 state head 容器一致（不一致 = 有槽位没人答）。
export ORCH_STATE_SLOTS="${ORCH_STATE_SLOTS:-}"

# 保留各机现有日志与音视频排障素材的默认目录。
export ORCH_LOG_LEVEL="${ORCH_LOG_LEVEL:-info}"
export ORCH_DUMP_AUDIO="${ORCH_DUMP_AUDIO:-$SHARED/orchdump/s}"
LOG_FILE="${ORCH_LOG_FILE:-$REPO/orch106-$PORT.log}"
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

echo "106 编排：IC = 进程内 vendor/interactioncore（每会话一份 Engine）" >&2
echo "  端口            = $PORT" >&2
echo "  Agent gRPC      = $ORCH_AGENT_GRPC（每会话独立 Interact）" >&2
echo "  Omni            = $ORCH_OMNI_URL" >&2
echo "  人脸            = ${ORCH_FACE_SERVICE_URL:-本地 .so（仅单会话）}（8PCM = $ORCH_FACE_PCM8）" >&2
echo "  python          = $PY" >&2
echo "  音视频转储      = $ORCH_DUMP_AUDIO" >&2
echo "  Omni 视觉       = $ORCH_OMNI_VISION（0=不发视频帧，音频照旧）" >&2
echo "  VLM 描述        = $ORCH_OMNI_DESCRIBE（0=关，omni 当对话方）" >&2
echo "  描述阶段        = 起点 full + 每轮 turn，AgentAction.vlm 单独传递" >&2
if [ "$ORCH_OMNI_DELTA_ALWAYS" = "0" ]; then _ALWAYS_TXT="仅静默时"; else _ALWAYS_TXT="不看语音活动"; fi
echo "  描述定时触发    = 每 ${ORCH_OMNI_DELTA_INTERVAL_S}s（${_ALWAYS_TXT}）+ 断句点各一次" >&2
echo "  ASR 感知        = $ORCH_ASR_PERCEPTION（0=只出基础 5 键）" >&2
if [ "$ORCH_STATE_HEAD" = "1" ]; then
  echo "  state head      = $ORCH_STATE_URL（超时 ${ORCH_STATE_TIMEOUT_S}s，${ORCH_STATE_N_FRAMES} 帧，节拍 ${ORCH_STATE_INTERVAL_S}s，stage=$ORCH_STATE_STAGE）" >&2
  echo "  E2 槽位白名单   = ${ORCH_STATE_SLOTS:-（空 = 全量 16 题）}" >&2
  echo "  文本侧          = desc-only=$ORCH_TEXT_DESC_ONLY（两行描述，上限 ${ORCH_OMNI_DESC_ONLY_MAX_NEW_TOKENS} token）" >&2
else
  echo "  state head      = 关（纯文本 E0 状态）" >&2
fi
echo "  暂停/续播渐变   = 淡出 ${ORCH_TTS_PAUSE_FADE_MS}ms / 淡入 ${ORCH_TTS_RESUME_FADE_MS}ms" >&2
echo "  打断探针        = $ORCH_IC_DECISION_PROBE" >&2
echo "  HTTPS 证书      = $SSL_CERT" >&2

exec "$PY" -u -m orchestrator.main \
  --host "${ORCH_HOST:-0.0.0.0}" --port "$PORT" \
  --downstream-mode "${ORCH_DOWNSTREAM_MODE:-omni}" \
  --log-level "$ORCH_LOG_LEVEL" "${SSL_ARGS[@]}"
