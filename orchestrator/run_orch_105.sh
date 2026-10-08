#!/usr/bin/env bash
# ⚠️ **已废弃（2026-10-08）** —— 105 现在跑 `run_orch_105_shared.sh`：
#    共享树 + 进程内 IC + 远端人脸（105:8767），**不再在 105 本地编 `.so`、
#    不再需要 LD_PRELOAD**（105 那份 `.so` 本来就编不出来）。
#    保留本文件是因为下面那些坑仍然成立、仍然有人踩：libffi/libstdc++ 冲突、
#    以及**绝不能在 NFS 共享目录里跑 `G1/build.sh`**（会覆盖 106 正在用的
#    `.so`）。但照它描述的本地 `.so` 部署**已经跑不起来**。
#
# 编排服务 —— **105（内部测试机）专属启动脚本**。
#
# 用法（105 上）：
#     bash orchestrator/run_orch_105.sh
#     nohup setsid bash orchestrator/run_orch_105.sh </dev/null >/dev/null 2>&1 &
#
# ⚠️ **不要再自己重定向到 `orch.log`** —— 脚本内部已经落到 `orch105.log`
#    （见下面第 5 条：NFS 共享目录，两边同名会打架）。前台调试想直接看输出：
#     ORCH_LOG_FILE= bash orchestrator/run_orch_105.sh
#
# 为什么不直接用 run_orch.sh —— 105 和 106 有几处**环境差异**，写在这里
# 比散在命令行里可靠（少一次"漏了某个 env 导致排查半天"）。
#
# ## 与 106 的四处差异（每处都踩过）
#
# 1. **python 环境**：106 用 `envs/py310`，105 没有它。
#    ⚠️ 用 `dj_py310` 不行 —— 它的 `libstdc++.so.6` 是 6.0.29，
#    而 G1 的 `.so` 要更新的 GLIBCXX（报错形如 `libstdc++.s`，被截断的
#    `libstdc++.so.6`）。新建的 `orch105` 是 6.0.34，与 106 一致。
#
# 2. **libffi**：`orch105` 自带的 `libffi.so.8` **缺 `LIBFFI_BASE_7.0`
#    符号**，而系统的 `libp11-kit` 需要它 → 加载 G1 的 `.so` 时报
#    `libp11-kit.so.0: undefined symbol: ffi_type_pointer`。
#    解药是让系统 `libffi.so.7` 先加载（见下面的 LD_PRELOAD）。
#
# 3. **G1 的 .so 必须在 105 本地编**：105 是 Ubuntu 20.04（opencv 4.2），
#    106 是 22.04（opencv 4.5），ABI 不通用。
#    ⚠️⚠️ **绝不能在共享目录里跑 `G1/build.sh`** —— `/data/megastore` 是
#    NFS，105 和 106 挂的是**同一份**（同 inode 已验证）。build.sh 结尾是
#    `cp -f` 到 `lib/x86_64/`，会**直接覆盖 106 正在用的那个 .so**，
#    把 106 的服务砸掉。所以源码拷到 105 **本地磁盘**再编：
#        rsync -a <共享>/G1/ /home/dujing/g1-105/
#        cd /home/dujing/g1-105 && bash build.sh
#
# 4. **Omni 在 106 上**（105 没有）。它监听 `0.0.0.0:8006`，所以可以跨机连
#    —— **不修改 106 的任何服务**，只是调用它已有的端口。
#    ⚠️ 代价：105 依赖 106 活着。
#
# 5. **日志文件名**：`/data/megastore` 是 NFS，105 和 106 看到的是**同一份**。
#    两边都叫 `orch.log` 会互相覆盖/错乱（105 上现存的那批 `orch.log.*`
#    其实全是 **106** 写的）。所以这里用 `orch105.log`，且由**本脚本**
#    负责重定向 —— 调用方不用再操心。
#
# 其余下游（ASR / AEC / TTS / Agent）都是**共享服务**，105、106 用同一套，
# 不需要改。
set -eu

ORCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$ORCH_DIR/.." && pwd)"
cd "$REPO"

# ---- 日志（见文件头第 5 条）----
# 落 REPO 根下，与 106 的 orch.log 区分开。
# `ORCH_LOG_FILE` 可覆盖；设成空串则输出到 stdout（前台调试用）。
LOG_FILE="${ORCH_LOG_FILE-$REPO/orch105.log}"
if [ -n "$LOG_FILE" ]; then
  # 追加而不是截断：重启一次不该把上一轮的线索冲掉
  exec >>"$LOG_FILE" 2>&1
  echo "=== $(date '+%F %T') 启动（pid $$）—— 日志 $LOG_FILE ===" >&2
fi

# ---- 1. python 环境（105 专用）----
export ORCH_PY="${ORCH_PY:-/home/dujing/miniconda3/envs/orch105/bin/python}"

# ---- 2. G1 根目录（105 本地编译的那份，不是 NFS 上 106 的）----
export ORCH_G1_ROOT="${ORCH_G1_ROOT:-/home/dujing/g1-105}"

# ---- 3. Omni 指向 106（105 本地没有）----
export ORCH_OMNI_URL="${ORCH_OMNI_URL:-wss://192.168.89.106:8006/v1/realtime?mode=video}"

# ---- 4. IC：**进程内 Engine，每会话一份**（不再是本机的 gRPC 服务）----
#
# 2026-09-29 切换：105 从「远端 gRPC IC + 单会话」改为「进程内 Engine + 多会话」。
#
# 旧方案的问题：远端 IC 的 `InteractionState` 是**进程级全局一份**，两个会话
# 同时驱动会互相覆盖 `mode` / `barge_hold_ms` / `turn`，现象是「ASR 识别完美却
# 零决策」。编排侧只好用「单一驱动者」闸门兜着（后来者接管、先来的挂起）。
#
# 新方案：每路 `OrchestratorSession` 自己持有一个 `interaction.runtime.Engine`
# —— 并发问题由构造消解，不需要 session 路由，也不需要单独跑 IC 服务。
# 那个 `:50051` 的 `interaction.grpc_server` 对 105 来说**已经没有用了**。
export ORCH_IC_MODE="${ORCH_IC_MODE:-inprocess}"

# `ORCH_IC_ADVERTISE` = **告诉 Agent 的会话回调根地址**。
# ⚠️ 进程内模式下**不再是 IC 的 gRPC 端口** —— 那个端口没了。要给的是**编排
#    服务自己的 HTTP 端点**（`{这个}/v1/ic/apply_agent`），Agent 把
#    `agent.status` / `session_end_pending` 写到这里（见 IC 的 AgentSink payload
#    里的 `callback_ic`）。
# ⚠️⚠️ **必须**是 105 的对外可回连地址，**不能**是 127.0.0.1 —— 从 Agent 视角看
#     127.0.0.1 是它自己，回写会派到错误的地方。
export ORCH_IC_ADVERTISE="${ORCH_IC_ADVERTISE:-192.168.89.105:8100}"
export ORCH_IC_CALLBACK_BASE="${ORCH_IC_CALLBACK_BASE:-https://192.168.89.105:8100/v1/ic}"

# 进程内模式下 IC 的 GREET/UTTER 与 YIELD/END 打的是**编排服务自己**的
# `/v1/speak` / `/v1/stop`（同机回环）。CA 用编排自己的自签证书。
export ORCH_IC_EXPRESSION_URL="${ORCH_IC_EXPRESSION_URL:-https://127.0.0.1:8100}"
export ORCH_IC_EXPRESSION_CA="${ORCH_IC_EXPRESSION_CA:-$REPO/certs/cert.pem}"

# `ORCH_IC_GRPC` / `ORCH_IC_RESTORE` 在进程内模式下**不再使用**（保留 export 只为
# 少一处 diff；置空也行）。
export ORCH_IC_GRPC="${ORCH_IC_GRPC:-127.0.0.1:50051}"
export ORCH_IC_RESTORE="${ORCH_IC_RESTORE:-192.168.89.105:50051}"

# ---- 4b. Agent：**105 专用那一套**（192.168.89.103），不是共享的 102 ----
#
# ⚠️⚠️ 这是本机与 106 **隔离的关键**。Agent 的 `interaction_core_target` 是
#    进程级全局单例（只有一个值）。两台机器共用一个 Agent 时，谁最后开会话
#    谁赢，另一方的 IC Action 必然派错（实测：12:36 抢了一轮、12:38 又抢）。
#
#    105 用 103 的 Agent、106 用 102 的，**两套完全独立**。
#
# ⚠️ 2026-09-29 起 105 走 **8082**（`103:/home/dujing/agent-test`）而不是 8081：
#    8081 跑的是 NFS 共享那份代码（属主 lijiahui），只有「方案 A」，
#    **没有**多会话并发所需的「每会话一个桥接」与「IC 回写 HTTP 分流」。
#    8082 是独立进程 + 独立代码副本，与 8081 / 102 完全隔离。
#    起停：`ssh 192.168.89.103 'bash ~/agent-test/scripts/run_agent_105.sh start'`
export ORCH_AGENT_URL="${ORCH_AGENT_URL:-http://192.168.89.103:8082}"

# 105 的 Agent 已能按 payload 里的 `callback_ic` 回写（方案 A），
# 不再需要 `set_target` 这个进程级全局兜底 —— 关掉它，少一条互相覆盖的路径。
export ORCH_AGENT_SET_TARGET="${ORCH_AGENT_SET_TARGET:-0}"

# ---- 5. 人脸：**走 105 本机已部署的服务**（服务化，不加载 .so）----
CODE="$(cd "$REPO/.." && pwd)"
# ⚠️ **必须显式开**，否则 `face=False`、会话降级为无人脸。
#    `run_orch.sh` 里有这一行，本脚本漏抄过一次 —— 现象是"推理都正常、
#    就是没有人脸框"，而日志只有一行 `能力: ... face=False`，很容易看漏。
export ORCH_ENABLE_FACE="${ORCH_ENABLE_FACE:-1}"

# 2026-09-29 起 105 走**远端人脸服务**（HTTP），不再 CDLL 加载 `libsdk_stream.so`：
# 那个 .so 绑定编译机的 OpenCV（105 是 4.2 / 106 是 4.5，二进制不通用），
# 换机器就得重编，且 105 上还踩过 libstdc++ / libffi 符号冲突。
# 服务化后调用方只需一个 URL。
#
# ⚠️ `ORCH_G1_ROOT` / `ORCH_FACE_DB` 在远端模式下**不再需要**（资产在服务端）。
export ORCH_FACE_SERVICE_URL="${ORCH_FACE_SERVICE_URL:-http://192.168.89.105:8767}"
export ORCH_FACE_SERVICE_TIMEOUT_S="${ORCH_FACE_SERVICE_TIMEOUT_S:-2.0}"
# G1 的 debug 落盘必须关：开了 create 时会写最多约 4GB 视频
# （远端模式下这个变量由服务端自己管，留着不影响）
export G1_FACE_DEBUG="${G1_FACE_DEBUG:-0}"

# ---- 日志与转储 ----
export ORCH_LOG_LEVEL="${ORCH_LOG_LEVEL:-info}"
export ORCH_DUMP_AUDIO="${ORCH_DUMP_AUDIO:-$CODE/orchdump/s}"

# ---- libffi 修复（见文件头第 2 条）----
# ⚠️ 只 preload **系统 libffi**，不要 preload libstdc++ —— 那会引入
#    `libp11-kit` 的另一个符号冲突（实测过）。
if [ -e /usr/lib/x86_64-linux-gnu/libffi.so.7 ]; then
  export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libffi.so.7${LD_PRELOAD:+:$LD_PRELOAD}"
fi

SSL_CERT="${ORCH_SSL_CERT:-$REPO/certs/cert.pem}"
SSL_KEY="${ORCH_SSL_KEY:-$REPO/certs/key.pem}"
SSL_ARGS=()
if [ -f "$SSL_CERT" ] && [ -f "$SSL_KEY" ]; then
  SSL_ARGS=(--ssl-cert "$SSL_CERT" --ssl-key "$SSL_KEY")
else
  echo "⚠️ 未找到证书（$SSL_CERT / $SSL_KEY）—— 将以 HTTP 启动；" >&2
  echo "   从别的机器用 http:// 打开时浏览器不会给麦克风/摄像头权限。" >&2
fi

echo "105 编排服务启动：py=$ORCH_PY"
echo "  G1   = $ORCH_G1_ROOT（105 本地编译）"
echo "  Omni = $ORCH_OMNI_URL"
echo "  IC   = $ORCH_IC_GRPC"

exec "$ORCH_PY" -u -m orchestrator.main \
  --host "${ORCH_HOST:-0.0.0.0}" --port "${ORCH_PORT:-8100}" \
  --downstream-mode "${ORCH_DOWNSTREAM_MODE:-omni}" \
  --log-level "${ORCH_LOG_LEVEL}" \
  "${SSL_ARGS[@]}"
