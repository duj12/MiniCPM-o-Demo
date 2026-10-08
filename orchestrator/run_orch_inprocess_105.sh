#!/bin/bash
# ⚠️ **已废弃（2026-10-08）** —— 它走**独立副本** `/data/.../code-inprocess-105/`，
#    那份副本已经删掉，105 现在用 `run_orch_105_shared.sh`（直接读共享树）。
#    `docs/105-inprocess-deploy.md` 描述的正是这套副本流程，**同样过期**。
#
# 105 **进程内 IC + 远端人脸** 部署脚本（隔离路径，不动 106）。
#
# 用法（105 上，路径任意）：
#     bash /data/megastore/Projects/DuJing/code-inprocess-105/MiniCPM-o-Demo/orchestrator/run_orch_inprocess_105.sh
#     nohup setsid bash .../run_orch_inprocess_105.sh </dev/null >orch.log 2>&1 &
#
# ## 为什么是一个**独立路径**的独立脚本
#
# 105 与 106 读的是**同一份 NFS 代码**（`/data/megastore/Projects/DuJing/code`），
# 而且两边的 conda env 都把 `interactioncore` 装成了**指向那份共享目录的
# editable install**：
#
#     orch105 (105)  -> /data/.../DuJing/code/interactioncore
#     py310   (106)  -> /data/.../DuJing/code/interactioncore
#
# 所以在共享目录里改一行 IC 代码，**106 下次重启就会跟着变**。git 分支挡不住
# 这件事 —— 远程根本没有 git 仓库，只是同步过来的文件树。
#
# 因此本部署走**另一份拷贝**：
#     /data/megastore/Projects/DuJing/code-inprocess-105
# 它与共享目录是**不同的路径、不同的文件**，105 只从这里读，106 完全不受影响。
# （G1 人脸服务在 105 上早就是这么做的：`.../tianxiaowen/code/g1-face-service-105`。）
#
# ## 与 105 现有部署的差异
#
#   · `ORCH_IC_MODE=inprocess` —— 每路会话在本进程内独占一份 IC Engine，
#     **不需要单独跑 IC 服务**（原来的 :50051 那份可以停掉，本脚本不起它）。
#   · `ORCH_FACE_SERVICE_URL` —— 人脸改走 HTTP 调本机 8767 的服务，
#     **不加载 libsdk_stream.so**（105 上那份 .so 的 OpenCV/libstdc++/libffi
#     问题就是这么绕开的）。
#   · Agent 仍是 **103**（105 专用），不是 106 的 102。
#
# ## ⚠️ 切换前置条件（别忘了）
#
# Agent Platform 必须能把 `apply_agent`（写 agent.status / session_end_pending）
# 投到本服务的 `POST /v1/ic/apply_agent`。它原来直连 IC 的 gRPC `ApplyAgent`，
# 而进程内模式下**没有那个 gRPC 端口**了。Agent 不改，SOP 39 与
# PENDING_ANNOUNCE 两条分支会失效。
#
# 兜底：`ORCH_AGENT_SET_TARGET=1`（默认）时会继续用 set_target 告诉 Agent
# 回连地址 —— 但那只解决「Action 往哪派」，不解决 apply_agent 往哪写。
set -eu

NEW="/data/megastore/Projects/DuJing/code-inprocess-105"
REPO="$NEW/MiniCPM-o-Demo"
SHARED="/data/megastore/Projects/DuJing/code"

cd "$REPO"

PY="${ORCH_PY:-/home/dujing/miniconda3/envs/orch105/bin/python}"
if [ ! -x "$PY" ]; then
  echo "找不到 python: $PY（可用 ORCH_PY=... 覆盖）" >&2
  exit 1
fi

# ---- 让 `import interaction` 指向**隔离路径**的那份 ----
#
# ⚠️⚠️ **光设 PYTHONPATH 没用**（实测）。orch105 里的 `interaction` 是
#     现代 pip 的 editable install，装的是一个 **`sys.meta_path` 钩子**：
#
#         __editable___interaction_0_1_0_finder.py
#         MAPPING = {'interaction': '/data/.../DuJing/code/interactioncore/interaction'}
#
#     `meta_path` 的查找**优先于 `sys.path`**，所以无论把隔离路径放在
#     PYTHONPATH 还是 `sys.path[0]`，`import interaction` 都会命中那份
#     **硬编码的共享路径** —— 也就是说会静默地跑到 106 正在用的代码上去。
#     这个坑没有任何报错，只会表现为「改了代码却不生效」。
#
#     所以这里：① 把 meta_path 里的 editable 钩子摘掉；② 再把隔离路径
#     插到 sys.path 最前。用 `sitecustomize.py` 是因为它在**解释器启动时**
#     就自动执行，早于任何业务 import。
SITECUSTOMIZE="$NEW/sitecustomize.py"
cat > "$SITECUSTOMIZE" <<'PYEOF'
"""让本部署只从**隔离路径**读 `interaction` 包（不碰共享目录）。

见启动脚本里的说明：pip 的 editable install 用 `sys.meta_path` 钩子把
`interaction` 硬编码到共享目录，PYTHONPATH 盖不过它。
"""
import sys

_ISOLATED = "/data/megastore/Projects/DuJing/code-inprocess-105/interactioncore"

# 1) 摘掉 editable 的 meta_path 钩子（它优先于 sys.path）
sys.meta_path = [f for f in sys.meta_path
                 if type(f).__name__ != "_EditableFinder"]
# 2) 任何已被 editable 钩子抢先加载的 interaction.* 都作废，避免半新半旧
for _name in [n for n in sys.modules if n == "interaction"
              or n.startswith("interaction.")]:
    del sys.modules[_name]
# 3) 隔离路径置顶
if _ISOLATED not in sys.path:
    sys.path.insert(0, _ISOLATED)
PYEOF
export PYTHONPATH="$NEW${PYTHONPATH:+:$PYTHONPATH}"

# ---- IC：进程内模式 ----
# 不再需要 ORCH_IC_GRPC / ORCH_IC_ADVERTISE / ORCH_IC_RESTORE（那是远端模式
# 的三件套）。但 ic_advertise 仍被用来推导 Agent 的回调根地址，所以留着 ——
# 它必须是**外部可回连**的地址（不能是 127.0.0.1，否则 Agent 会去连它自己）。
export ORCH_IC_MODE="${ORCH_IC_MODE:-inprocess}"
export ORCH_IC_ADVERTISE="${ORCH_IC_ADVERTISE:-192.168.89.105:8100}"
# 告诉 Agent 往哪写 apply_agent（空则由 ic_advertise 推导）
export ORCH_IC_CALLBACK_BASE="${ORCH_IC_CALLBACK_BASE:-https://192.168.89.105:8100/v1/ic}"
# IC 播报打的是**本机自己**的 /v1/speak
export ORCH_IC_EXPRESSION_URL="${ORCH_IC_EXPRESSION_URL:-https://127.0.0.1:8100}"

# ---- Agent：105 专用（**不是** 106 的 102）----
export ORCH_AGENT_URL="${ORCH_AGENT_URL:-http://192.168.89.103:8081}"

# ---- 人脸：走远端服务（本机 8767），不加载 .so ----
export ORCH_ENABLE_FACE="${ORCH_ENABLE_FACE:-1}"
export ORCH_FACE_SERVICE_URL="${ORCH_FACE_SERVICE_URL:-http://192.168.89.105:8767}"
export ORCH_FACE_SERVICE_TIMEOUT_S="${ORCH_FACE_SERVICE_TIMEOUT_S:-2.0}"

# ---- 声学延迟 D：105 的实测值（占位 250 等于不生效，见 run_orch.sh）----
export ORCH_AEC_DEFAULT_DELAY_MS="${ORCH_AEC_DEFAULT_DELAY_MS:-250}"

export ORCH_LOG_LEVEL="${ORCH_LOG_LEVEL:-info}"
# 转储落到隔离路径下，别污染共享的 orchdump/
export ORCH_DUMP_AUDIO="${ORCH_DUMP_AUDIO:-$NEW/orchdump/s}"

# ---- HTTPS（沿用共享路径那份证书；105/106 同证书）----
SSL_CERT="${ORCH_SSL_CERT:-$SHARED/MiniCPM-o-Demo/certs/cert.pem}"
SSL_KEY="${ORCH_SSL_KEY:-$SHARED/MiniCPM-o-Demo/certs/key.pem}"
SSL_ARGS=()
if [ -f "$SSL_CERT" ] && [ -f "$SSL_KEY" ]; then
  SSL_ARGS=(--ssl-cert "$SSL_CERT" --ssl-key "$SSL_KEY")
else
  echo "⚠️ 未找到证书（$SSL_CERT）—— 将以 HTTP 启动，别的机器访问时采不到音视频" >&2
fi

echo "105 进程内 IC 部署："
echo "  代码路径   = $NEW"
echo "  IC 模式    = $ORCH_IC_MODE（不需要单独的 IC 服务）"
echo "  人脸       = $ORCH_FACE_SERVICE_URL（远端 HTTP，不加载 .so）"
echo "  Agent      = $ORCH_AGENT_URL"
echo "  apply_agent 回调 = $ORCH_IC_CALLBACK_BASE"
echo "  python     = $PY"

exec "$PY" -u -m orchestrator.main \
  --host "${ORCH_HOST:-0.0.0.0}" --port "${ORCH_PORT:-8100}" \
  --downstream-mode "${ORCH_DOWNSTREAM_MODE:-omni}" \
  --log-level "${ORCH_LOG_LEVEL:-info}" \
  "${SSL_ARGS[@]}"
