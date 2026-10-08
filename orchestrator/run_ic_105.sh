#!/usr/bin/env bash
# InteractionCore —— **105（内部测试机）专属启动脚本**。
#
# 用法（105 上）：
#     bash orchestrator/run_ic_105.sh
#     nohup setsid bash orchestrator/run_ic_105.sh </dev/null >/dev/null 2>&1 &
#
# 为什么要单独一个脚本 —— 105 的 IC 早先是**手工敲命令行**起的，结果漏了
# `--agent-url`，用上了 `scripts/run_ic.sh` 里硬编码的默认值
# `http://192.168.89.102:8081`（那是 **106** 的 Agent）。后果很隐蔽：
#
#     IC 判 ANSWER → 送到 106 的 Agent → 它不认识 105 的会话 → 没有回复
#     → 没有 `/v1/speak` → **TTS 没声音**
#     而 GREET 有声音（它是 IC 直接调 `/v1/speak`，不经过 Agent）
#
# 现象是「Agent 像是回复了、但没声音」，实际是回复**根本没产生**。
# 固化到脚本里，就不会再漏。
#
# ## 与 106 的差异
#
#   · **Agent 地址**：105 用 **103** 的（专用），106 用 102 的（共享）。
#     两台共用 102 时，Agent 的 `interaction_core_target` 是进程级全局单例，
#     谁最后开会话谁赢 —— 必须分开。
#   · **python 环境**：105 是 `ic105`，106 是 `py310`。
#   · **expression-url**：105 连本机 `127.0.0.1:8100`。
#
# ⚠️ `scripts/run_ic.sh` 是 **NFS 共享文件**（105/106 同 inode），
#    **不要改它的默认值** —— 改了会连 106 一起影响。所以这里用环境变量覆盖。
set -eu

# 本文件在 <code>/MiniCPM-o-Demo/orchestrator/ 下，interactioncore 是**同级仓库**：
#   <code>/MiniCPM-o-Demo/orchestrator/run_ic_105.sh
#   <code>/interactioncore/
# 所以是 `../../../interactioncore`（orchestrator → MiniCPM-o-Demo → code）。
#
# ⚠️ 别写成 `../interactioncore` —— 那是 `MiniCPM-o-Demo/interactioncore`，
#    不存在，脚本会在 `cd` 时直接失败（实测踩过）。
IC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../interactioncore" && pwd)"

# ---- Agent：105 专用的那个（**不是** 102）----
export INTERACTION_AGENT_URL="${INTERACTION_AGENT_URL:-http://192.168.89.103:8081}"

# ---- 表达层（本机编排服务）----
export INTERACTION_EXPRESSION_URL="${INTERACTION_EXPRESSION_URL:-https://127.0.0.1:8100}"

# ---- python 环境（105 专用；106 的 py310 在 105 上不存在）----
export IC_PY="${IC_PY:-/home/dujing/miniconda3/envs/ic105/bin/python}"

echo "105 IC 启动：agent=$INTERACTION_AGENT_URL  expression=$INTERACTION_EXPRESSION_URL"
echo "  python=$IC_PY"

cd "$IC_DIR"
exec bash scripts/run_ic.sh
