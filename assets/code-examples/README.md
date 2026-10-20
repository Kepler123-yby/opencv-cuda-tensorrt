# Code Example Checklist: OpenCV-CUDA 与 TensorRT 推理部署实战

**Generated from:** `04-readiness-assessment.md` + `03-lesson-type-classification.md`
**Primary language:** C++（CUDA kernel 为 `.cu`）
**Key libraries:** OpenCV（含 CUDA 模块）、TensorRT、CUDA Runtime、yaml-cpp
**Last updated:** 2026-10-10

## Environment Requirements

- **编译环境**：C++20，CMake ≥ 3.22，Ninja 或 Make
- **CUDA**：与 TensorRT 匹配的 CUDA Toolkit（`nvcc` 可用）
- **TensorRT**：提供 `NvInfer.h` / `nvonnxparser`；**版本已确认 10.10.0.31**
- **OpenCV**：带 `cudaarithm` / `cudaimgproc` / `cudawarping` 的源码编译版本
- **cuDNN**：与 CUDA 匹配（OpenCV DNN CUDA 后端需要）
- **测试环境**：单张 NVIDIA GPU，离线即可（无需相机/串口）

## Naming Convention

```
[slt-number]-[example-name].[ext]
```

示例：`3.1-build-engine.cpp`、`4.1-fused-preprocess.cu`

## Checklist

### SLT 1.2: 编写 CMake 配置自动检测推理后端与 CUDA 能力，并编译进同一个库

| # | Filename | What to Demonstrate | Status |
|---|----------|---------------------|--------|
| 1 | `1.2-backend-detect/CMakeLists.txt` | `find_package(OpenCV/TensorRT)` + target 检测 + 生成式宏 | [ ] |

**Code requirements:**
- [ ] `find_package(OpenCV QUIET)`，检测 `opencv_cudawarping` target
- [ ] `find_package(TensorRT QUIET)`，回退到 `find_path`/`find_library`
- [ ] 检测结果通过 `target_compile_definitions` 传入源码
- [ ] 配置阶段打印检测到的后端

**API references needed:**
- [ ] 本机 `OpenCV_DIR` / `TENSORRT_ROOT` 实际路径

**Verification:**
- [ ] `cmake -S . -B build` 输出后端检测列表

---

### SLT 2.1: 用 GpuMat 在主机与设备间上传和下载图像数据

| # | Filename | What to Demonstrate | Status |
|---|----------|---------------------|--------|
| 1 | `2.1-gpumat-transfer.cpp` | `GpuMat::upload` / `download`、H2D/D2H 往返 | [ ] |

**Code requirements:**
- [ ] 读入一张图 → `upload` → `download` → 比对尺寸
- [ ] 打印上传/下载耗时，体现往返成本
- [ ] 注释说明 H2D / D2H 方向

**Verification:**
- [ ] 输出上传/下载后尺寸与耗时

---

### SLT 2.2: 用 cv::cuda 的缩放与 padding 算子在 GPU 上完成预处理

| # | Filename | What to Demonstrate | Status |
|---|----------|---------------------|--------|
| 1 | `2.2-cuda-preprocess.cpp` | `cv::cuda::resize` + `cv::cuda::copyMakeBorder`（letterbox） | [ ] |

**Code requirements:**
- [ ] upload → resize → copyMakeBorder（BORDER_CONSTANT）→ download
- [ ] 计算 letterbox 的缩放比与 padding
- [ ] 与 CPU 版本 `cv::resize`/`cv::copyMakeBorder` 视觉比对

**Verification:**
- [ ] GPU 预处理图与 CPU 图视觉一致，打印尺寸

---

### SLT 3.1: 从 ONNX 构建并缓存 TensorRT engine，并反序列化用于推理

| # | Filename | What to Demonstrate | Status |
|---|----------|---------------------|--------|
| 1 | `3.1-build-engine.cpp` | Builder + ONNX Parser + serialize + 缓存 + deserialize | [ ] |

**Code requirements:**
- [ ] `createInferBuilder` / `createNetworkV2` / `createParser`
- [ ] `buildSerializedNetwork` → 写 `.engine` 缓存
- [ ] 缓存存在则直接 `deserializeCudaEngine`
- [ ] Logger 实现

**API references needed:**
- [ ] TensorRT 10.10.0.31 API：`enqueueV3` / `setTensorAddress` / `setMemoryPoolLimit`（✅ 已核对）
- [ ] `nvinfer1::NetworkDefinitionCreationFlag`（注：`kEXPLICIT_BATCH` 已废弃）

**Verification:**
- [ ] 首次构建生成 `.engine`，二次运行走缓存，打印张量信息

---

### SLT 3.2: 分配输入输出显存与页锁定内存，并在 CUDA 流上执行推理

| # | Filename | What to Demonstrate | Status |
|---|----------|---------------------|--------|
| 1 | `3.2-buffers-and-stream.cpp` | `cudaMalloc` / `cudaMallocHost` / `setTensorAddress` / `enqueueV3` | [ ] |

**Code requirements:**
- [ ] 分配 device 输入输出 + pinned 主机缓冲
- [ ] `setInputShape` + `setTensorAddress` 绑定
- [ ] `cudaStreamCreate` + `enqueueV3` + `cudaMemcpyAsync` + `cudaStreamSynchronize`
- [ ] 析构释放全部资源

**Verification:**
- [ ] 一次推理输出维度正确，打印输出张量形状

---

### SLT 3.3: 实现同步、异步与多路异步三种推理模式

| # | Filename | What to Demonstrate | Status |
|---|----------|---------------------|--------|
| 1 | `3.3-infer-modes.cpp` | sync / 单路 async / N 路并行（多 context + 多 stream，保序回收） | [x] 已生成、编译通过；GPU 验证待完成 |

**Code requirements:**
- [x] 三种模式共用同一 engine、同一组按编号变化的输入
- [x] 多路并行每路独立 context + buffer + stream + event，复用前回收，尾部排空
- [x] 单独正确性验证，对照同步参考，检查全输出容差与非有限值
- [x] 打印每模式 FPS 与逐请求 mean / p50 / p95，包含背压与 fill/drain
- [x] 提供拒绝忙槽位复用、错帧回收和重复回收的检查；不把竞争必现当作验收条件

**Verification:**
- [ ] 正确性通过后报告三模式 FPS 与延迟；解释并行收益或无收益，不要求并行一定更快

> **参考实现**：课文 [`3.3-sync-async-parallel.md`](../../lessons/module-3/3.3-sync-async-parallel.md)、完整程序 [`3.3-infer-modes.cpp`](3.3-infer-modes.cpp)、CPU 调度检查 [`tests/verify-module3.py`](tests/verify-module3.py)、验证记录 [`module-3-verification.md`](module-3-verification.md)。示例限定静态一进一出 FP32/LINEAR/DEVICE；编译和 FakeSlot 调度检查不等于 GPU 正确性或性能验证。

---

### SLT 4.1: 编写融合 letterbox、通道转换与归一化的 CUDA kernel，直接写入设备输入缓冲区

| # | Filename | What to Demonstrate | Status |
|---|----------|---------------------|--------|
| 1 | `4.1-fused-preprocess.cu` | `__global__` kernel：letterbox + BGR→RGB + /255 + HWC→CHW | [ ] |

**Code requirements:**
- [ ] kernel 处理 GpuMat 行步长对齐（`src_step`）
- [ ] 直接写 TensorRT 输入显存，无 D2H→H2D 往返
- [ ] 与 CPU 预处理数值比对

**Verification:**
- [ ] 融合结果与 CPU `blobFromImage` 数值一致（容差内）

---

### SLT 4.2: 编写并行解码检测候选的 CUDA kernel，并用原子操作压缩结果

| # | Filename | What to Demonstrate | Status |
|---|----------|---------------------|--------|
| 1 | `4.2-decode-kernel.cu` | `__global__` 解码候选 + `atomicAdd` 压缩 | [ ] |

**Code requirements:**
- [ ] 每个线程处理一个候选，阈值过滤
- [ ] `atomicAdd` 写入紧凑数组
- [ ] NMS 留在 CPU（说明语义一致性）

**Verification:**
- [ ] GPU 解码候选数量/坐标与 CPU 路径一致

---

### SLT 5.3: 独立编写并调试一个 TensorRT + OpenCV-CUDA 项目，复用部署库的模式

| # | Filename | What to Demonstrate | Status |
|---|----------|---------------------|--------|
| 1 | `5.3-capstone/` | 可运行最小项目：读视频 → GPU 预处理 → TensorRT 推理 → GPU 后处理 → 画框输出 | [ ] |

**Code requirements:**
- [ ] 完整 CMake 工程
- [ ] 输入视频 → 输出视频
- [ ] 可切换 CPU / CUDA 路径
- [ ] 含 README 与运行命令

**Verification:**
- [ ] 独立跑通，输出视频有检测框，打印 FPS

---

## Summary

| SLT | Examples Needed | Verified | Notes |
|-----|-----------------|----------|-------|
| 1.2 | 1 | 0 | CMake 检测 |
| 2.1 | 1 | 0 | GpuMat 往返 |
| 2.2 | 1 | 0 | CUDA 预处理 |
| 3.1 | 1 | 0 | engine 构建/缓存 |
| 3.2 | 1 | 0 | buffer + stream |
| 3.3 | 1 | 0 | 完整样例已编译，CPU 调度检查通过；GPU 验证待完成 |
| 4.1 | 1 | 0 | 课文 kernel 片段已真机验证；完整独立样例待补 |
| 4.2 | 1 | 0 | 课文 kernel/NMS 片段已真机验证；完整独立样例待补 |
| 5.3 | 1（多文件） | 0 | 课文、接口与调度片段已补；完整可运行 capstone 待实现 |

**Total:** 9 code examples（capstone 含多文件；上表 Verified 指完整独立样例）。

## Module 4 已生成的辅助素材

| 文件 | 作用 | 状态 |
|---|---|---|
| `4.4-compare-tensors.py` | 比较小端 float32 张量导出；检查尺寸、NaN/Inf、容差，提供明确退出码 | 7 项 CLI 测试通过 |
| `tests/verify-module4.py` | 从课文提取 CUDA kernel，运行合成数据与 CPU 参考，支持 memcheck | 预处理 7 项、解码 7 项通过 |
| `module-4-verification.md` | 环境、复现命令、误差与验证范围 | 已记录 |

本轮不使用业务仓库或真实 TensorRT engine。kernel 测试与完整推理集成是两项不同的验收；素材收集状态仍未完成。

## Module 5 已生成的辅助素材

| 文件 | 作用 | 状态 |
|---|---|---|
| `../configs/5.1-deploy.yaml` | 完整配置 schema、显式 backend/pre/post/mode | 8 个合法组合验证通过 |
| `5.1-validate-config.cpp` | yaml-cpp 配置关系与字段校验，不探测设备或加载模型 | 共 26 项通过，含 18 项非法配置拒绝 |
| `5.2-summarize-benchmark.py` | 从已消费请求的 CPU 时间戳算延迟/FPS，含 fill/drain 与背压等待 | 15 项人工数据/负向测试通过；不是性能实测 |
| `5.3-pipeline-contract.hpp` | 原图/frame_id/token/slot 的接口契约 | 接口与课文 shape/调度片段共 3 项语法检查通过 |
| `tests/verify-module5.py` | 配置/统计回归，可选 C++ 语法检查 | 可复现，不调用真实推理 |
| `module-5-verification.md` | 环境、命令、覆盖与未验证范围 | 已记录 |

Module 5 三节课时已出稿。完整 detector 是 5.3 的综合实现作业，不把接口、CMake 参考和未来 CLI 当成附带的可运行工程。

## 依赖分层

| Tier | SLT | 需要 | 预计工作量 |
|------|-----|------|-----------|
| 1：独立 | 2.1, 2.2 | OpenCV CUDA + CUDA（无需模型） | 低 |
| 2：需模型/后端 | 1.2, 3.1, 3.2, 3.3, 4.1, 4.2 | TensorRT + ONNX / 已构建 Engine；3.3 的 CPU 调度检查无需 GPU | 中 |
| 3：综合项目 | 5.3 | 视频 + 模型 + 完整工程 | 较高 |

## API Documentation Links

| Library | Doc URL | Version | Status |
|---------|---------|---------|--------|
| TensorRT | https://docs.nvidia.com/deeplearning/tensorrt/ | 10.10.0.31 | [x] 已核对头文件 |
| OpenCV CUDA | https://docs.opencv.org/4.x/ | 4.x | [ ] |
| CUDA Runtime | https://docs.nvidia.com/cuda/cuda-runtime-api/ | 待确认 | [ ] |

## Gathering Session Notes

_记录实测观察：_

- Date:
- Tester:
- Environment version:
- Issues encountered:
