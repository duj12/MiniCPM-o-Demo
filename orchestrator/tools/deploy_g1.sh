#!/bin/bash
# 把 board-face-and-cloud-infer/G1 的最新代码同步到 106 **并重编 .so**。
#
# 用法（**必须从 106 上跑**，因为要调用那边的 g++/opencv）：
#
#     ssh 192.168.89.106
#     cd /data/megastore/Projects/DuJing/code/MiniCPM-o-Demo
#     bash orchestrator/tools/deploy_g1.sh
#
# 也可以从本地一条命令跑完：
#     ssh 192.168.89.106 'bash /data/megastore/Projects/DuJing/code/MiniCPM-o-Demo/orchestrator/tools/deploy_g1.sh'
#
# ────────────────────────────────────────────────────────────────────── #
# 为什么需要这个脚本（每一步都是踩出来的）
#
# ① **绝不能同步 .so**。G1 仓库里那份 `lib/<arch>/libsdk_stream.so` 是在
#    **编译者那台机器**上编的，链接的 opencv 版本与 glibc 都编死在里面，
#    换一台机器就加载不了。实测：仓库里那份要 opencv 4.10 + glibc 2.38，
#    而 106 只有 4.5.4 + glibc 2.35 →
#        OSError: libopencv_imgcodecs.so.410: cannot open shared object file
#    所以流程是「同步**源码** → 在**目标机器上重编**」，不是拷 .so。
#
# ② **不能按目录名挑 .so**。「lib/<arch>/ 是新的」这个直觉是错的：106 上
#    同时存在几份产物，加载不了的那份可能恰好在 lib/<arch>/ 下。编排服务
#    的 `face/g1face_provider.py::pick_g1_lib()` 会**实测**每份候选
#    （真 CDLL 一次 + 验 ABI），取第一份能用的 —— 这个脚本沿用同样的判据。
#
# ③ **重编前先备份**。G1 是同步目录、不是 git 仓库，编坏了没得回退。
#
# ④ **改完必须重启编排服务**。.so 是进程启动时加载的；不重启则仍用旧库，
#    表现为「改了阈值但行为没变」（实测踩过）。
# ────────────────────────────────────────────────────────────────────── #
set -eu

ORCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"       # .../orchestrator
REPO="$(cd "$ORCH_DIR/.." && pwd)"                                # .../MiniCPM-o-Demo
CODE="$(cd "$REPO/.." && pwd)"                                    # code/ 层
G1="${G1_ROOT:-$CODE/board-face-and-cloud-infer/G1}"

PY="${ORCH_PY:-/home/dujing/miniconda3/envs/py310/bin/python}"

say() { echo "[deploy_g1] $*"; }
die() { echo "[deploy_g1] 错误: $*" >&2; exit 1; }

[ -d "$G1" ] || die "找不到 G1 仓库: $G1（可用 G1_ROOT=... 覆盖）"
[ -x "$PY" ] || die "找不到 python: $PY（可用 ORCH_PY=... 覆盖）"

# ---- 1) 备份（G1 不是 git 仓库，编坏了没得回退）----
if [ "${SKIP_BACKUP:-0}" != "1" ]; then
  BAK="$CODE/board-face-and-cloud-infer/G1.bak.$(date +%Y%m%d_%H%M%S)"
  cp -a "$G1" "$BAK"
  say "已备份 → $BAK"
fi

# ---- 2) 同步源码（**排除 .so**）----
# 本地 → 远程这一步由同步工具负责；这里只校验「源码是否比 .so 新」，
# 避免「改了源码忘了重编」——那是本流程最常见的错误。
SRC_NEWEST="$(find "$G1/src" "$G1/include" -name '*.cpp' -o -name '*.h' \
              | xargs ls -t 2>/dev/null | head -1)"
SO="$G1/lib/x86_64/libsdk_stream.so"
if [ -n "$SRC_NEWEST" ] && [ -f "$SO" ] && [ "$SRC_NEWEST" -nt "$SO" ]; then
  say "源码比 .so 新，需要重编（$(basename "$SRC_NEWEST")）"
else
  say "警告：源码不比 .so 新 —— 若刚同步过代码却没生效，检查同步是否真的落地"
fi

# ---- 3) 重编（产物落 lib/<arch>/，ORT 一并拷过去）----
say "编译中（当前机器: $(uname -m)）..."
( cd "$G1" && bash build.sh ) || die "编译失败（看上面的 g++ 输出）"

# ---- 4) 验证：**实测加载**，不按目录名猜 ----
# 与 face/g1face_provider.py 的 pick_g1_lib() 同口径：真的 CDLL 一次 + 验 ABI。
say "验证新产物..."
"$PY" - "$G1" <<'PYEOF' || die "新编的 .so 不可用"
import ctypes, os, sys
g1 = sys.argv[1]
cands = [os.path.join(g1, "lib", a, "libsdk_stream.so")
         for a in ("x86_64", "aarch64")]
cands += [os.path.join(g1, "src", "libsdk_stream.so"),
          os.path.join(g1, "lib", "libsdk_stream.so")]
ok = []
for p in cands:
    if not os.path.isfile(p):
        continue
    try:
        lib = ctypes.CDLL(p)
    except OSError as e:
        print(f"  跳过 {p}：{str(e)[:60]}")
        continue
    fn = getattr(lib, "g1_face_abi_version", None)
    if fn is None:
        print(f"  跳过 {p}：没有 g1_face_abi_version（旧构建）")
        continue
    fn.restype = ctypes.c_int; fn.argtypes = []
    ok.append((p, fn()))
if not ok:
    sys.exit(1)
for p, v in ok:
    print(f"  可用: {p}（ABI={v}）")
PYEOF
say "验证通过"

# ---- 5) 重启编排服务（.so 是启动时加载的，不重启不生效）----
if [ "${SKIP_RESTART:-0}" = "1" ]; then
  say "已跳过重启（SKIP_RESTART=1）"
  exit 0
fi
say "重启编排服务..."
pkill -9 -f 'python -u -m orchestrator.main' 2>/dev/null || true
sleep 2
cd "$REPO"
setsid bash orchestrator/run_orch.sh </dev/null >orch.log 2>&1 &
sleep 14
if curl -sk --max-time 5 https://127.0.0.1:"${ORCH_PORT:-8100}"/healthz; then
  echo
  say "完成。注意：**浏览器要 Ctrl+Shift+R 强制刷新**（HTML 会被缓存）"
else
  die "服务未起来 —— 看 $REPO/orch.log"
fi
