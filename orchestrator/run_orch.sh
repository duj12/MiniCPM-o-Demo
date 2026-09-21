#!/bin/bash
# Orchestrator 启动脚本 —— 集中管理所有环境变量。
#
# 用法（**从哪跑都行**，路径按脚本自身位置推导）：
#     bash orchestrator/run_orch.sh
#     nohup setsid bash orchestrator/run_orch.sh </dev/null >orch.log 2>&1 &
#
# ⚠️ 这些环境变量**必须在这里**：早先手工 nohup 启动时漏了人脸相关的，
#    表现为"人脸模块突然没了"（face=False），排查了半天。
set -eu

# ---- 路径：按脚本位置推导，不写死 ----
# 本文件在 <repo>/orchestrator/ 下，repo 根是它的上一级。
ORCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$ORCH_DIR/.." && pwd)"
# CODE = 与 MiniCPM-o-Demo **同级**的仓库目录（board-face-and-cloud-infer、
# faceidentification 都在那里）。
CODE="$(cd "$REPO/.." && pwd)"

cd "$REPO"

PY="${ORCH_PY:-/home/dujing/miniconda3/envs/py310/bin/python}"
if [ ! -x "$PY" ]; then
  echo "找不到 python: $PY（可用 ORCH_PY=... 覆盖）" >&2
  exit 1
fi

# ---- 声学延迟 D（毫秒）—— 每台设备一个固定常量，**离线测一次** ----
# 运行时自适应已从代码里整个撤下（真机上会被噪声假峰钉死并永久失效，
# 见 orchestrator/README.md），所以 ORCH_AEC_ADAPTIVE 这个开关已不存在。
#
# ⚠️ 250 是**占位值，不是可用值** —— 实测抑制只有 0.3dB，等于不工作。
#    测本机的真实 D（ORCH_DUMP_AUDIO 已开）：
#      1) 跑一轮真实会话：页面选「算法服务 AEC」、外放、别戴耳机、说两三句
#      2) python -m orchestrator.tests.measure_delay \
#           --mic "$CODE/orchdump/s-<sid>-mic.wav" \
#           --raw "$CODE/orchdump/s-<sid>-raw.wav" \
#           --verify-url ws://192.168.88.253:30255/ws/asr_frontend
#      3) 把打印出来的值填到下面这一行，然后重启。
#
# 注意：AEC 模式默认已是 browser（浏览器原生，不需要 D）；只有显式选
# 「算法服务 AEC」时才用到这个值。
export ORCH_AEC_DEFAULT_DELAY_MS="${ORCH_AEC_DEFAULT_DELAY_MS:-250}"

# ---- 日志级别（排障用）----
# debug 会逐条打出服务端回来的**每一条** ASR 原始消息（mode / text / is_final /
# confidence / turnsense 判决）。「UI 上没有 ASR 文字」有两种完全不同的原因
# —— 服务端没收到结果 vs 收到了但没下发 UI —— 这条日志把两者分开。
# ⚠️ 必须在这里透传：orchestrator.main 的 --log-level 默认是 info，
#    只设 ORCH_LOG_LEVEL 环境变量**不会**生效。
export ORCH_LOG_LEVEL="${ORCH_LOG_LEVEL:-info}"

# ---- 音视频转储（排障用；不设则关闭，零开销）----
# 会话结束时落盘：mic/ref/raw/aec 四路 wav + face/omni 的 mjpeg+tsv。
# 文件名里的 <sid> 就是网页状态栏显示的那个会话 id。
# ⚠️ 会话**进行中不会有文件**，是 close() 时才写。
export ORCH_DUMP_AUDIO="${ORCH_DUMP_AUDIO:-$CODE/orchdump/s}"

# ---- 人脸模块 ----
export ORCH_ENABLE_FACE="${ORCH_ENABLE_FACE:-1}"
# ⚠️ **默认不指定 .so 路径 —— 让代码自己挑。**
#
# G1 仓库里同时躺着几份 x86_64 产物，**它们不通用**：`lib/<arch>/` 那份是
# `build.sh` 的正规产物，但它链接的 opencv 取决于**编译那台机器**（在
# opencv 4.10 上编的，拿到只有 4.5 的 106 上就是 `imgcodecs.so.410 not found`）；
# `lib/` 根下那份是旧约定遗留，**ABI 太老**（没有 g1_face_abi_version）。
#
# 按目录名挑（"分架构 = 新的"）已经踩过一次：编排服务人脸模块整个装配失败，
# 而 g1face 自己的 `_find_lib()` 也优先挑 `lib/<arch>/`，同样中招。
#
# 现在 provider 会**实测**每份候选（真的 CDLL 一次 + 验 ABI 版本），
# 选中第一份能用的，并把它钉给 g1face（见 `face/g1face_provider.py` 的
# `pick_g1_lib`）。所以在 106 上只要 G1 仓库在、模型在，就能直接跑。
#
# 想强制用某一份时才设它（同样会过校验，坏的不会被静默采纳）：
# export ORCH_FACE_SO="/path/to/libsdk_stream.so"
export ORCH_FACE_MODELS="${ORCH_FACE_MODELS:-$CODE/board-face-and-cloud-infer/G1/models}"
export ORCH_FACE_DB="${ORCH_FACE_DB:-$CODE/faceidentification/data/face_db.npz}"
# ⚠️ 必须为 0：否则 create 时就会写最多约 4GB 视频
export G1_FACE_DEBUG="${G1_FACE_DEBUG:-0}"

# ---- HTTPS ----
# 浏览器只在安全上下文（https / localhost）给麦克风+摄像头权限：
# 用 http://<局域网IP>:8100 打开时 navigator.mediaDevices 是 undefined，
# 页面根本采不到音视频。
# 证书是自签的，**SAN 里必须含实际访问用的地址**（Edge/Chrome 只看 SAN、
# 忽略 CN）。换机器/换 IP 要重新签。首次访问浏览器拦一次，点「继续访问」。
SSL_CERT="${ORCH_SSL_CERT:-$REPO/certs/cert.pem}"
SSL_KEY="${ORCH_SSL_KEY:-$REPO/certs/key.pem}"
SSL_ARGS=()
if [ -f "$SSL_CERT" ] && [ -f "$SSL_KEY" ]; then
  SSL_ARGS=(--ssl-cert "$SSL_CERT" --ssl-key "$SSL_KEY")
else
  # 缺证书不致命（只是别的机器访问时采不到音视频），但要说清楚
  echo "⚠️ 未找到证书（$SSL_CERT / $SSL_KEY）—— 将以 HTTP 启动；" >&2
  echo "   从别的机器用 http:// 打开时浏览器不会给麦克风/摄像头权限。" >&2
fi

exec "$PY" -u -m orchestrator.main \
  --host "${ORCH_HOST:-0.0.0.0}" --port "${ORCH_PORT:-8100}" \
  --downstream-mode "${ORCH_DOWNSTREAM_MODE:-omni}" \
  --log-level "${ORCH_LOG_LEVEL:-info}" \
  "${SSL_ARGS[@]}"
