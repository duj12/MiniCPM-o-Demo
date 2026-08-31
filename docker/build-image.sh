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
#   checkpoints/ 是指向外部目录的软链（gitignore，不入库）。docker build 的
#   COPY 不会解析软链（拷成悬空链），所以这里先把软链解析成实体文件到
#   .bundle-checkpoints/checkpoints/（-L 跟随软链拷贝），Dockerfile 在
#   BUNDLE_MODELS=1 时把它 COPY 进 /app/checkpoints。
#
#   .bundle-checkpoints/ 是构建临时目录（.dockerignore 未排除以便 COPY），
#   构建后自动清理。
# ============================================================================
set -euo pipefail

cd "$(dirname "$0")/.."   # 项目根

BUNDLE=1    # 默认打包模型进镜像；--no-bundle 关闭（开发/调试用挂载）
PUSH=0
CACHE_BUST="$(date +%s)"
for arg in "$@"; do
    case "$arg" in
        --no-bundle) BUNDLE=0 ;;
        --push)      PUSH=1 ;;
    esac
done

BUNDLE_DIR=".bundle-checkpoints"

cleanup() {
    if [ -d "$BUNDLE_DIR" ]; then
        echo "[build-image] cleaning $BUNDLE_DIR"
        rm -rf "$BUNDLE_DIR"
    fi
}
trap cleanup EXIT

build_args=(
    --build-arg "CACHE_BUST=${CACHE_BUST}"
)

if [ "$BUNDLE" = "1" ]; then
    echo "[build-image] bundling models into image..."
    if [ ! -d checkpoints ]; then
        echo "[build-image] ERROR: checkpoints/ not found (softlink dir required)" >&2
        exit 1
    fi
    # 只打包三个必需子目录，软链实体化
    mkdir -p "$BUNDLE_DIR/checkpoints"
    for sub in qwen3omni-gguf fsmn-vad-onnx TurnSense; do
        if [ -e "checkpoints/$sub" ]; then
            echo "[build-image]   copying $sub -> $BUNDLE_DIR/checkpoints/$sub"
            cp -Lr "checkpoints/$sub" "$BUNDLE_DIR/checkpoints/$sub"
        else
            echo "[build-image] WARNING: checkpoints/$sub not found, skipped" >&2
        fi
    done
    build_args+=(--build-arg "BUNDLE_MODELS=1")
else
    echo "[build-image] no --bundle: model NOT baked into image (mount /models at runtime)"
fi

echo "[build-image] docker build (cache_bust=${CACHE_BUST})..."
# compose build 会把 build.args 默认值（LLAMA_OMNI_REPO/REFSPEC）传给 Dockerfile，
# 这里用 --build-arg 追加 CACHE_BUST 和可选的 BUNDLE_MODELS。
docker compose -f docker-compose.cpp.yml build \
    --build-arg "CACHE_BUST=${CACHE_BUST}" \
    "${build_args[@]}" \
    cpp-worker-backend

if [ "$PUSH" = "1" ]; then
    : "${REGISTRY:?REGISTRY required for --push}"
    : "${IMAGE_TAG:?IMAGE_TAG required for --push}"
    echo "[build-image] pushing ${REGISTRY}/omnillm-cpp-backend:${IMAGE_TAG} ..."
    docker tag omnillm-cpp-backend:dev "${REGISTRY}/omnillm-cpp-backend:${IMAGE_TAG}"
    docker push "${REGISTRY}/omnillm-cpp-backend:${IMAGE_TAG}"
fi

echo "[build-image] done"
