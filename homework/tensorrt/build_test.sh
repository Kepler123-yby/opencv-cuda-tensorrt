#!/usr/bin/env bash
# 编译并运行三种推理方式的对拍测试
set -euo pipefail

cd "$(dirname "$0")"

TRT_ROOT="${TRT_ROOT:-/home/kepler/tensorrt/TensorRT-10.10.0.31}"
CUDA_ROOT="${CUDA_ROOT:-/usr/local/cuda}"
ENGINE="${1:-model/model.engine}"

g++ -std=c++17 -O2 -Wall -Wextra \
    -Wno-deprecated-declarations -Wno-unused-parameter \
    -o ./test_three_mode_infer \
    three_mode_infer.cpp test_three_mode_infer.cpp \
    -I. -I"$TRT_ROOT/include" -I"$CUDA_ROOT/include" \
    -L"$TRT_ROOT/targets/x86_64-linux-gnu/lib" -lnvinfer \
    -L"$CUDA_ROOT/lib64" -lcudart -lpthread

export LD_LIBRARY_PATH="$TRT_ROOT/targets/x86_64-linux-gnu/lib:$CUDA_ROOT/lib64:${LD_LIBRARY_PATH:-}"
exec ./test_three_mode_infer "$ENGINE"
