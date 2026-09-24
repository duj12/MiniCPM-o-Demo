#!/usr/bin/env bash
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

# ---- 4. IC 在 105 本机 ----
# `ORCH_IC_GRPC` = **编排服务自己去连** IC（本机，127.0.0.1 没问题）
export ORCH_IC_GRPC="${ORCH_IC_GRPC:-127.0.0.1:50051}"
# `ORCH_IC_ADVERTISE` = **告诉 Agent 回连哪个地址**（Agent 在 192.168.89.102）
# ⚠️⚠️ 这个**必须**是 105 的对外可回连地址，**不能**是 127.0.0.1 ——
#     从 Agent 视角看 127.0.0.1 是它自己，Action 会派到错误的地方；
#     而且会话结束时若把这个错地址"归还"出去，**106 的会话也会跟着错**。
#     这是 105/106 互相影响的真实通道之一（实测踩过）。
export ORCH_IC_ADVERTISE="${ORCH_IC_ADVERTISE:-192.168.89.105:50051}"
# `ORCH_IC_RESTORE` = 会话结束后把 Agent 的 IC 目标**归还到哪**。
# ⚠️ 是**大家共用的**那个 IC（106），**不是**本机（105）。归还成自己的话，
#    106 的下一个会话又得抢一次、每次都打冲突告警，抢的窗口里 106 是错的。
export ORCH_IC_RESTORE="${ORCH_IC_RESTORE:-192.168.89.106:50051}"

# ---- 5. 人脸（共享存储，链接有效）----
CODE="$(cd "$REPO/.." && pwd)"
# ⚠️ **必须显式开**，否则 `face=False`、会话降级为无人脸。
#    `run_orch.sh` 里有这一行，本脚本漏抄过一次 —— 现象是"推理都正常、
#    就是没有人脸框"，而日志只有一行 `能力: ... face=False`，很容易看漏。
export ORCH_ENABLE_FACE="${ORCH_ENABLE_FACE:-1}"
export ORCH_FACE_DB="${ORCH_FACE_DB:-$CODE/faceidentification/data/face_db.npz}"
# G1 的 debug 落盘必须关：开了 create 时会写最多约 4GB 视频
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
