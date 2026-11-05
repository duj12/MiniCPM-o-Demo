#!/usr/bin/env bash
# ============================================================================
# 构建 worker 镜像（可选把模型打包进镜像）
#
# 用法：
#   ./docker/build-image.sh                 # 默认打包模型进镜像（qwen3omni+VAD+TurnSense）
#   ./docker/build-image.sh --no-bundle     # 不打包（运行时挂载 /models）
#   ./docker/build-image.sh --push          # 打包并推送（需 REGISTRY/IMAGE_TAG 环境变量）
#
# 模型打包原理：
#   checkpoints/ 可为实体目录或外部目录的软链。docker build 的 COPY
#   不会解析软链，所以这里先将模型暂存到 .bundle-checkpoints/checkpoints/。
#   优先跟随软链创建硬链接，不支持时回退为普通复制。Dockerfile
#   直接 COPY 到 /app/checkpoints，模型在镜像内只保留一份。
#
#   .bundle-checkpoints/ 是构建临时目录（.dockerignore 未排除以便 COPY），
#   构建后自动清理。CI 保留 checkpoints/，通过 rsync 增量更新模型。
# ============================================================================
set -euo pipefail

cd "$(dirname "$0")/.."   # 项目根

BUNDLE=1    # 默认打包模型进镜像；--no-bundle 关闭（开发/调试用挂载）
PUSH=0
for arg in "$@"; do
    case "$arg" in
        --no-bundle) BUNDLE=0 ;;
        --push)      PUSH=1 ;;
    esac
done

BUNDLE_DIR=".bundle-checkpoints"
LLAMA_SRC_DIR=".llama-src"
# 默认源码目录（宿主机 clone 的 duj12/dev，含最新代码）；可用 --llama-src 覆盖
DEFAULT_LLAMA_SRC="../llama.cpp-omni-dev"
LLAMA_SRC="${LLAMA_SRC:-$DEFAULT_LLAMA_SRC}"

cleanup() {
    if [ -d "$BUNDLE_DIR" ]; then
        echo "[build-image] cleaning $BUNDLE_DIR"
        rm -rf "$BUNDLE_DIR"
    fi
    if [ -d "$LLAMA_SRC_DIR" ]; then
        echo "[build-image] cleaning $LLAMA_SRC_DIR"
        rm -rf "$LLAMA_SRC_DIR"
    fi
}
trap cleanup EXIT

# 源码进构建上下文（.llama-src/，Dockerfile COPY 它）。不在容器内 git clone
# —— 部分网络到 github 不稳，容器内 clone 常失败。宿主机 clone 后传进来。
if [ ! -d "$LLAMA_SRC" ]; then
    echo "[build-image] ERROR: llama source not found at $LLAMA_SRC (host clone duj12/dev first)" >&2
    exit 1
fi
echo "[build-image] staging llama source from $LLAMA_SRC ..."
mkdir -p "$LLAMA_SRC_DIR"
cp -r "$LLAMA_SRC"/. "$LLAMA_SRC_DIR"/
echo "[build-image]   llama source staged ($(du -sh "$LLAMA_SRC_DIR" | cut -f1))"

if [ "$BUNDLE" = "1" ]; then
    echo "[build-image] bundling models into image..."
    if [ ! -d checkpoints ]; then
        echo "[build-image] ERROR: checkpoints/ not found (softlink dir required)" >&2
        exit 1
    fi
    # 只打包三个必需子目录；每次重建硬链接，避免残留旧模型。
    rm -rf "$BUNDLE_DIR"
    mkdir -p "$BUNDLE_DIR/checkpoints"
    for sub in qwen3omni-gguf fsmn-vad-onnx TurnSense; do
        if [ -d "checkpoints/$sub" ]; then
            echo "[build-image]   linking $sub -> $BUNDLE_DIR/checkpoints/$sub"
            if ! cp -Lrl "checkpoints/$sub" "$BUNDLE_DIR/checkpoints/$sub"; then
                # 跨文件系统等场景无法硬链接，清理不完整结果后普通复制。
                echo "[build-image]   hard links unavailable for $sub, copying instead"
                rm -rf "$BUNDLE_DIR/checkpoints/$sub"
                cp -Lr "checkpoints/$sub" "$BUNDLE_DIR/checkpoints/$sub"
            fi
        else
            echo "[build-image] WARNING: checkpoints/$sub not found, skipped" >&2
        fi
    done
else
    echo "[build-image] no --bundle: model NOT baked into image (mount /models at runtime)"
fi

echo "[build-image] docker build (bundle_models=${BUNDLE})..."
docker compose -f docker-compose.cpp.yml build \
    --build-arg "BUNDLE_MODELS=${BUNDLE}" \
    cpp-worker-backend

if [ "$PUSH" = "1" ]; then
    : "${REGISTRY:?REGISTRY required for --push}"
    : "${IMAGE_TAG:?IMAGE_TAG required for --push}"
    echo "[build-image] pushing ${REGISTRY}/omnillm-cpp-backend:${IMAGE_TAG} ..."
    docker tag omnillm-cpp-backend:dev "${REGISTRY}/omnillm-cpp-backend:${IMAGE_TAG}"
    docker push "${REGISTRY}/omnillm-cpp-backend:${IMAGE_TAG}"
fi

echo "[build-image] done"
