#!/bin/bash
# 105 生产编排 —— 与 `run_orch_106.sh` 对称（同一套开关，只有地址/环境不同）。
#
# IC 与编排代码都走**共享树** `/data/megastore/Projects/DuJing/code`：
#   · 好处：IC 更新只需改共享树/同步，不再有「副本漏同步」的隐患
#   · 代价：105 与 106 共用同一份代码。106 跑的也是并发版
#           （`run_orch_106.sh`，同样 inprocess）—— 两边靠**各自的 Agent**
#           隔离（105→103，106→102），不是靠 IC 模式隔离。
#           改了共享树**不影响在跑的进程**（Python 启动时就 import 完了），
#           只影响下次重启 ⇒「改了代码」≠「服务更新了」。
#
# 进程内 Engine 靠 `interaction` 包的 orch105 环境（editable 指向共享树）。
#
# ⚠️ 曾经为了绕开共享树而走的**独立副本**（`code-inprocess-105`）已废弃 ——
#    那条路踩的坑（pip editable 的 `sys.meta_path` 钩子硬编码指向共享树，
#    `PYTHONPATH` 盖不过它）记在 docs/多会话并发与云服务化-问题与方案.md。
#
# ## 回退到 grpc 版（同一脚本，只翻模式）
#
#     # 1) 停掉在跑的进程 —— **按 PID**。别用 `pkill -f orchestrator.main`：
#     #    那个模式会匹配到 ssh 自己的命令行（踩过）
#     ss -ltnp | grep ':8100' | sed -n 's/.*pid=\([0-9]*\).*/\1/p' | xargs -r kill -TERM
#     # 2) 起 grpc 版
#     ORCH_IC_MODE=grpc nohup setsid bash .../orchestrator/run_orch_105.sh </dev/null >/dev/null 2>&1 &
#
#     ⚠️ :50051 上那个独立 IC 服务（pid 101057）**别停** —— 回退的依赖。
#        它由 `run_ic_105.sh` 起。
#     （Agent 两种模式都是 **103:8081**，翻模式不用跟着改 —— 见下面那节。）
#
# ## 影子实例
#
#     ORCH_PORT=8101 bash .../orchestrator/run_orch_105.sh
# 回调地址与 IC 播报地址都跟着 `ORCH_PORT` 走（日志文件不跟 —— 影子实例会
# 追加到同一个 orch105.log，看的时候注意分辨启动横幅）。
set -eu
SHARED=/data/megastore/Projects/DuJing/code
REPO=$SHARED/MiniCPM-o-Demo
cd "$REPO"

export ORCH_PY="${ORCH_PY:-/home/dujing/miniconda3/envs/orch105/bin/python}"
export ORCH_LOG_FILE="${ORCH_LOG_FILE:-$SHARED/orch-dumps/105/orch105.log}"
mkdir -p "$(dirname "$ORCH_LOG_FILE")"

# 端口先定下来 —— Agent 回调地址、IC 播报地址都跟着它走。
# ⚠️ 早先这几处是**写死的 8100**：起 8101 影子实例时 advertise 仍指回生产，
#    属于「看着起来了、其实把 Agent 的写入引到另一个进程上」的那种错。
PORT="${ORCH_PORT:-8100}"

# ---- IC：两种模式，`ORCH_IC_MODE` 就是回退开关 ----
#
#   `inprocess`（默认）每会话独占一份 Engine —— 真并发，**不需要** IC 服务。
#   `grpc`             回退到 :50051 上的独立 IC 服务（`run_ic_105.sh` 起的那个）。
#
# ⚠️⚠️ **只翻 `ORCH_IC_MODE`、却把 `ORCH_IC_ADVERTISE` 留在 `:8100` 是最坏的
#       一种半切换**：8100 是 HTTP(S) 端口，而 Agent 会拿它当 **gRPC 目标**去
#       回连 IC → 连不上，而**编排侧看不出异常**（回连的不是它）。一次改全。
export ORCH_IC_MODE="${ORCH_IC_MODE:-inprocess}"
if [ "$ORCH_IC_MODE" = "grpc" ]; then
  export ORCH_IC_GRPC="${ORCH_IC_GRPC:-127.0.0.1:50051}"
  export ORCH_IC_ADVERTISE="${ORCH_IC_ADVERTISE:-192.168.89.105:50051}"
  export ORCH_IC_RESTORE="${ORCH_IC_RESTORE:-192.168.89.105:50051}"
  export ORCH_AGENT_SET_TARGET="${ORCH_AGENT_SET_TARGET:-1}"
else
  # 必须**外部可回连** —— 写 127.0.0.1 的话 Agent 会去连它自己
  export ORCH_IC_ADVERTISE="${ORCH_IC_ADVERTISE:-192.168.89.105:$PORT}"
  export ORCH_IC_CALLBACK_BASE="${ORCH_IC_CALLBACK_BASE:-https://192.168.89.105:$PORT/v1/ic}"
  export ORCH_AGENT_SET_TARGET="${ORCH_AGENT_SET_TARGET:-0}"
fi
# 进程内模式下 IC 打的是**本机自己**的 /v1/speak / /v1/stop。
# ⚠️ grpc 模式下这两个变量在编排侧不生效 —— IC 服务自己带
#    `--expression-url/--expression-ca`（它是 `https://127.0.0.1:8100`，见 run_ic_105.sh）。
export ORCH_IC_EXPRESSION_URL="${ORCH_IC_EXPRESSION_URL:-https://127.0.0.1:$PORT}"
export ORCH_IC_EXPRESSION_CA="${ORCH_IC_EXPRESSION_CA:-$REPO/certs/cert.pem}"

# ---- Agent：105 专用（103）----
#
# **两种 IC 模式都走 8081**（2026-10-10 起）。
#
#   grpc      → 必须 8081：跑着的 IC 服务（pid 101057）是
#               `--agent-url http://192.168.89.103:8081` 起的，IC 的四个
#               `on_*` payload 送到 8081。编排若走别处，就等于「IC 判了
#               ANSWER、回复却送到另一个 Agent」—— 症状是**没有回复也不报错**
#               （`run_ic_105.sh` 记的正是这个坑的另一半）。
#   inprocess → 也是 8081：8081 上的 `openai-agent-demo-try` 已经是多会话版
#               （`BridgePool` 每会话一桥接 + LRU 回收），编排的每会话
#               `callback_ic` 会让它把播报出口切到本会话 —— 与 8082 等价。
#
# ⚠️ **8082（dujing 的 `~/agent-test` 手工副本）已退役**。它存在的唯一理由
#    是「播报出口是进程级全局」（`OrchestratorClient.target` 只在
#    `VoiceSession.__init__` 读一次）⇒ 一份实例只能服务一台编排。该限制已按
#    「会话级覆盖 + 全局兜底」改掉并合进 pite 的 try ⇒ 8081 一份就够，
#    无需再维护一条同步链（见 memory `agent-103-8082-vlm`）。
#    别把这里改回 8082：那个端口上已经没有服务了，症状同样是「没有回复也不
#    报错」——与上面 grpc 那半句一模一样。
export ORCH_AGENT_URL="${ORCH_AGENT_URL:-http://192.168.89.103:8081}"

# ---- Omni（105 本地没有，指 106）----
export ORCH_OMNI_URL="${ORCH_OMNI_URL:-wss://192.168.89.106:8006/v1/realtime?mode=video}"

# ---- 人脸：远端服务（8767），不加载 .so ----
#
# ⚠️ 用 `-` 而不是 `:-` —— 只有 `-` 才让**显式给的空值**保留下来。105 上
#    **没有可用的 `.so`**（编不出来：opencv/libstdc++/libffi 一连串符号问题，
#    见「已废弃部署路线」那节），所以置空不是「换回本地」而是「关掉人脸」：
#        ORCH_FACE_SERVICE_URL= ORCH_ENABLE_FACE=0 bash orchestrator/run_orch_105.sh
export ORCH_ENABLE_FACE="${ORCH_ENABLE_FACE:-1}"
export ORCH_FACE_SERVICE_URL="${ORCH_FACE_SERVICE_URL-http://192.168.89.105:8767}"
export ORCH_FACE_SERVICE_TIMEOUT_S="${ORCH_FACE_SERVICE_TIMEOUT_S:-2.0}"
export G1_FACE_DEBUG="${G1_FACE_DEBUG:-0}"

# ⚠️ 编排服务**不读** ORCH_LOG_FILE —— 日志重定向必须在脚本里做（exec），
#    早先只设了变量，结果 8100 的 stdout 全进了 /dev/null（排查时无日志可看）。
LOG_FILE="${ORCH_LOG_FILE:-$SHARED/orch-dumps/105/orch105.log}"
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

# ---- VLM 描述（两阶段）----
#
# ⚠️ **只有 105 这一台开**。`run_orch_106.sh` 保持关（0 / legacy）—— 那边的
#    102 Agent 还没确认能接 `content` 字段，而 `orchestrator/` 是两台共用的
#    一份代码，`config.py` 里的**默认值就是 106 下次重启后的行为**。
#    ⇒ 要开只能落在这个**本机专属**脚本里，`config.py` 的默认不许动；
#      这样「哪台开了」在启动横幅上一眼可见（下两行 echo 的就是这里的值）。
#
#   `dual`   = **纯增量**：`transcript` 仍是字符串，只多一个
#              `content={ASR,VLM}` 键 ⇒ 对没改过的老 Agent 向后兼容
#              （多出来的键被忽略）。105 走 103 的 Agent，2026-10-08 起用这个。
#   `dict`   = `transcript` 变 dict，**要 Agent 侧先改好**才能翻。
#   `legacy` = 连子类都不装，payload 逐字节等于旧版（回退用）。
#
# 回退：`ORCH_AGENT_TRANSCRIPT_MODE=legacy ORCH_OMNI_DESCRIBE=0 bash orchestrator/run_orch_105.sh`
export ORCH_OMNI_DESCRIBE="${ORCH_OMNI_DESCRIBE:-1}"
export ORCH_AGENT_TRANSCRIPT_MODE="${ORCH_AGENT_TRANSCRIPT_MODE:-dual}"

# ---- 增量换人设（全量做完后关掉重连、换「只报变化」短人设）----
#
# 为什么非这样不可：每轮 prompt（`input.append.text`）**压不住** system
# prompt 里的八类清单 —— 实测三版全失败，唯一有效的是让它没有八类清单，
# 而那要换 system prompt，服务端又不支持一条连接上第二次 init ⇒ 只能关掉
# 重连一次。证据与推演在 `orchestrator/omni/describe.py` 的模块文档
# 「为什么增量轮压不短」。代价：整个会话多一次重连（~1~3s）。
#
# 默认**开着**（只在这台脚本里）—— 与 `ORCH_OMNI_DESCRIBE` 同款：105 是
# 唯一跑描述链路的一台，106 的 `run_orch_106.sh` 两个都不设。
# 关了仍能跑，只是增量长（八类分条、每次 3~4s）。
export ORCH_OMNI_DELTA_PERSONA="${ORCH_OMNI_DELTA_PERSONA:-1}"

# 生效配置落日志 —— 回退/切换后**第一眼看这里**，别靠猜
echo "105 编排：IC 模式 = $ORCH_IC_MODE" >&2
if [ "$ORCH_IC_MODE" = "grpc" ]; then
  echo "  IC 服务         = $ORCH_IC_GRPC（独立进程，回退态；别停它）" >&2
  echo "  告诉 Agent 的 IC = $ORCH_IC_ADVERTISE（**必须是 :50051**）" >&2
else
  echo "  IC 服务         = 无（进程内 Engine，每会话一份）" >&2
  echo "  apply_agent 回调 = $ORCH_IC_CALLBACK_BASE" >&2
fi
echo "  Agent           = $ORCH_AGENT_URL（set_target=$ORCH_AGENT_SET_TARGET）" >&2
echo "  人脸            = ${ORCH_FACE_SERVICE_URL:-本地 .so（未配远端服务）}" >&2
echo "  python          = $ORCH_PY" >&2
# VLM 描述（两阶段）生效值 —— **这行是「有没有碰过本机 Agent 协议」的唯一
# 硬证据**。105 期望看到「1 / dual」（见上面 export 块）；**106 期望看到
# 「0 / legacy」**，哪天在 106 的日志里看到 dual，就是有人改了共享的默认值。
echo "  VLM 描述        = $ORCH_OMNI_DESCRIBE（0=关，omni 当对话方）" >&2
echo "  Agent 载荷      = $ORCH_AGENT_TRANSCRIPT_MODE" \
     "（legacy=与旧版逐字节一致，dual=多一个 content 键）" >&2
# 「视觉背景轮」没有独立开关：它与 `content` **同开同关**（判据 = sink 是否被
# 包装，而那恰好等于「描述开 且 载荷非 legacy」）。加第三个开关会造出
# 「背景轮开、content 关」这种半开状态 —— sink 根本没包装，静默不发、无日志。
# 开了之后日志里应能看到 `视觉背景已交给 Agent（stage=full，N 字）`。
echo "  视觉背景轮      = 随上面两行（$ORCH_OMNI_DESCRIBE / $ORCH_AGENT_TRANSCRIPT_MODE）" \
     "（全量描述额外投一发 ASR 空的 on_answer，只存不答）" >&2
# ⚠️ 与上两行同理：这行是「本机跑的是哪一版描述链路」的硬证据。
# 开了之后日志里应能看到 `已换成「只报变化」短人设（关掉重连一次）`。
echo "  增量人设        = $ORCH_OMNI_DELTA_PERSONA" \
     "（1=全量做完后关掉重连、换短人设跑增量）" >&2

exec "$ORCH_PY" -u -m orchestrator.main \
  --host "${ORCH_HOST:-0.0.0.0}" --port "$PORT" \
  --downstream-mode "${ORCH_DOWNSTREAM_MODE:-omni}" \
  --log-level "$ORCH_LOG_LEVEL" "${SSL_ARGS[@]}"
