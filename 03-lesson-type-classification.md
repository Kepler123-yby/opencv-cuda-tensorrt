# Lesson Type Classification: OpenCV-CUDA 与 TensorRT 推理部署实战

## Summary

| Lesson Type | Count | SLTs |
|-------------|-------|------|
| Product Demo | 0 | — |
| Developer Documentation | 9 | 1.2, 2.1, 2.2, 3.1, 3.2, 3.3, 4.1, 4.2, 5.3 |
| How To Guide | 5 | 1.1, 1.3, 4.4, 5.1, 5.2 |
| Organization Onboarding | 0 | — |
| Exploration | 3 | 2.3, 3.4, 4.3 |

> 本课程为**代码密集型、无 UI**，故不涉及 Product Demo 与 Organization Onboarding。

### Classification Insight（来自以往课程）

- 「实现 / 编写 / 构建」类动词 → Developer Documentation（代码是主要产物）
- 「安装 / 编译 / 运行 / 配置」类动词、终端或配置文件为主 → How To Guide
- 「解释 / 比较 / 评估」类动词、无代码产物 → Exploration
- 边界规则：**主要界面决定类型**。终端/配置文件驱动的实操 → How To Guide；代码产物驱动的技能 → Developer Documentation。

## Classifications

### Module 1: 环境与工具链

#### SLT 1.1: "我能从源码编译启用 CUDA 与 cuDNN 的 OpenCV，并安装到独立前缀与系统 CPU 版共存。"
- **Lesson Type**: How To Guide
- **Key Factors**: 终端编译安装流程；产物是「可用的环境」而非代码文件
- **Edge Notes**: 也可写成 Developer Documentation（CMake 配置），但主线是步骤流程
- **Inputs Needed**: 编译命令与 flags 清单、cuDNN 获取步骤、环境脚本、验证命令

#### SLT 1.2: "我能编写 CMake 配置，自动检测推理后端与 CUDA 能力并将其编译进同一库。"
- **Lesson Type**: Developer Documentation
- **Key Factors**: 产物是 `CMakeLists.txt`（代码文件）；涉及 `find_package` / target 检测
- **Edge Notes**: 无
- **Inputs Needed**: `CMakeLists.txt` 代码、`OPENVINO_DIR` / `TENSORRT_ROOT` / `OpenCV_DIR` 说明

#### SLT 1.3: "我能运行一个 GPU 图像处理示例，验证 CUDA 设备与算子已正确启用。"
- **Lesson Type**: How To Guide
- **Key Factors**: 「运行 + 验证」的检查流程
- **Edge Notes**: 无
- **Inputs Needed**: 示例命令、期望输出、常见报错与处理

### Module 2: GPU 挂载与数据搬运

#### SLT 2.1: "我能用 GpuMat 在主机与设备间上传和下载图像数据。"
- **Lesson Type**: Developer Documentation
- **Key Factors**: OpenCV CUDA API 用法（`GpuMat::upload/download`）
- **Edge Notes**: 无
- **Inputs Needed**: `cv::cuda::GpuMat` 最小代码示例、H2D/D2H 语义说明

#### SLT 2.2: "我能用 cv::cuda 的缩放与 padding 算子在 GPU 上完成预处理。"
- **Lesson Type**: Developer Documentation
- **Key Factors**: 库 API 集成（`cv::cuda::resize` / `copyMakeBorder`）
- **Edge Notes**: 无
- **Inputs Needed**: 完整预处理代码参照 `network_preprocess_cuda.cpp`

#### SLT 2.3: "我能解释 host↔device 往返如何成为 GPU 预处理的性能瓶颈。"
- **Lesson Type**: Exploration
- **Key Factors**: 纯概念理解，无代码产物；解释「为什么」
- **Edge Notes**: 是 4.3 的问题铺垫
- **Inputs Needed**: 引导问题（往返开销、pinned memory、带宽 vs 延迟）

### Module 3: TensorRT 推理

#### SLT 3.1: "我能从 ONNX 构建并缓存 TensorRT engine，并反序列化用于推理。"
- **Lesson Type**: Developer Documentation
- **Key Factors**: TensorRT C++ API 用法（Builder / Parser / Runtime / Engine）
- **Edge Notes**: 无
- **Inputs Needed**: 参照 `network_deployment_tensorrt.cpp` 的构建/缓存代码

#### SLT 3.2: "我能分配输入输出显存与页锁定内存，并在 CUDA 流上执行推理。"
- **Lesson Type**: Developer Documentation
- **Key Factors**: CUDA/TensorRT API 用法（`cudaMalloc`、`cudaMallocHost`、`enqueueV3`）
- **Edge Notes**: 无
- **Inputs Needed**: 缓冲区分配与绑定的代码、流同步说明

#### SLT 3.3: "我能实现同步、异步与多路异步三种推理模式。"
- **Lesson Type**: Developer Documentation
- **Key Factors**: 代码实现三种模式
- **Edge Notes**: ✅ 已确认 TensorRT **10.10.0.31** API（`enqueueV3` / `setTensorAddress` / 多 context）。仓库的多路异步示例是 OpenVINO 专属，但 TensorRT 的多 context + 多 stream 模式是 10.x 标准能力，API 已核对
- **Inputs Needed**: `network_deployment_openvino.cpp` 的 async4 实现（作对照）+ TensorRT 多 context 参考（课时 `3.3-sync-async-parallel.md` 已含）

#### SLT 3.4: "我能比较串行与多路异步流水线在延迟与吞吐上的差异。"
- **Lesson Type**: Exploration
- **Key Factors**: 权衡与选型判断，无代码产物
- **Edge Notes**: 与 3.3 配对（3.3 实现，3.4 比较）
- **Inputs Needed**: 引导问题（吞吐 vs 延迟、流水线填充、帧对齐）

### Module 4: 前后处理 GPU 化

#### SLT 4.1: "我能编写融合 letterbox、通道转换与归一化的 CUDA kernel，直接写入设备输入缓冲区。"
- **Lesson Type**: Developer Documentation
- **Key Factors**: 编写 CUDA 代码（`__global__` kernel）
- **Edge Notes**: 无
- **Inputs Needed**: `letterboxBgrToRgbNchwKernel` 代码、步长对齐说明

#### SLT 4.2: "我能编写并行解码检测候选的 CUDA kernel，并用原子操作压缩结果。"
- **Lesson Type**: Developer Documentation
- **Key Factors**: 编写 CUDA 代码（kernel + `atomicAdd`）
- **Edge Notes**: 无
- **Inputs Needed**: 参照 `network_postprocess_cuda.cpp` 的解码 kernel

#### SLT 4.3: "我能评估零拷贝设计如何消除 host↔device 往返。"
- **Lesson Type**: Exploration
- **Key Factors**: 设计评估，判断「为什么」；是 2.3 的解法
- **Edge Notes**: 无
- **Inputs Needed**: 引导问题（融合预处理、输出留显存、端到端收益）

#### SLT 4.4: "我能验证 GPU 前后处理与 CPU 路径的结果一致性。"
- **Lesson Type**: How To Guide
- **Key Factors**: 运行一致性测试并解读结果的流程（`cuda_parity_test`）
- **Edge Notes**: 亦可强调写校验代码 → Developer Documentation；此处按**验证流程**归类
- **Inputs Needed**: 一致性测试命令、容差阈值（2px / 0.1）

### Module 5: 部署、调试与性能

#### SLT 5.1: "我能通过 YAML 配置在运行时切换推理后端与 CUDA 开关。"
- **Lesson Type**: How To Guide
- **Key Factors**: 配置文件操作，无代码产物
- **Edge Notes**: 无
- **Inputs Needed**: YAML 字段说明、`detect_*_cuda.yaml` 示例

#### SLT 5.2: "我能测量端到端延迟与吞吐，对比 CPU、CUDA 与 TensorRT 配置。"
- **Lesson Type**: How To Guide
- **Key Factors**: 基准测试流程（`detector_speed_bench` / `benchmark_app`）
- **Edge Notes**: 亦带 Exploration（选型判断）成分；主线按**测量流程**归类
- **Inputs Needed**: 测速命令、参数（iterations / key）、结果表模板

#### SLT 5.3: "我能独立编写并调试一个 TensorRT + OpenCV-CUDA 项目，复用部署库的模式。"
- **Lesson Type**: Developer Documentation
- **Key Factors**: 综合代码产物（capstone）
- **Edge Notes**: 综合了 M2–M5 的模式，作为结课项目
- **Inputs Needed**: 接口 `network_deployment_interface.hpp`、最小可运行项目骨架、调试清单

## Heuristics Developed

### Verb Patterns
- 「编写 / 实现 / 构建」→ Developer Documentation（代码是产物）
- 「安装 / 编译 / 运行 / 配置」→ How To Guide（终端/配置流程）
- 「解释 / 比较 / 评估」→ Exploration（无代码产物的概念）

### Subject Matter Patterns
- CUDA kernel、TensorRT/CUDA C++ API、OpenCV CUDA API → Developer Documentation
- 环境编译、配置切换、基准测试、一致性验证 → How To Guide
- 性能瓶颈、流水线权衡、零拷贝设计 → Exploration

### Ambiguity Patterns
- 当 SLT 同时含「测量」与「对比」时，按**主线动作**归类（测量为流程 → How To Guide；对比为判断 → 可归 Exploration）。
- 当「验证」既是流程又涉及写代码时，用**主要产物**判别：产物是测试结果 → How To Guide；产物是校验代码 → Developer Documentation。

## Context Shopping List

### Code Examples Needed (Developer Documentation)

- [ ] 1.2 — `CMakeLists.txt` 后端自动检测片段
- [ ] 2.1 — `GpuMat` upload/download 最小示例
- [ ] 2.2 — `cv::cuda::resize` + `copyMakeBorder` 预处理（参照 `network_preprocess_cuda.cpp`）
- [ ] 3.1 — ONNX→engine 构建与缓存（参照 `network_deployment_tensorrt.cpp`）
- [ ] 3.2 — 显存/pinned 分配与 `enqueueV3`（同上）
- [ ] 3.3 — sync/async 实现（TensorRT）+ async4 流水线（OpenVINO，参照 `network_deployment_openvino.cpp`）
- [ ] 4.1 — 融合预处理 kernel `letterboxBgrToRgbNchwKernel`
- [ ] 4.2 — 解码 kernel + `atomicAdd`（参照 `network_postprocess_cuda.cpp`）
- [ ] 5.3 — capstone 项目骨架 + `network_deployment_interface.hpp`

### Framing Questions Needed (Exploration)

- [ ] 2.3 — 往返开销从何而来？pinned memory 为什么更快？
- [ ] 3.4 — 吞吐与延迟如何取舍？流水线填充为何重要？
- [ ] 4.3 — 零拷贝在哪些环节消除往返？端到端收益为何有限？

### Supplementary Docs Needed (How To Guide)

- [ ] 1.1 — 编译 flags、cuDNN 获取、环境脚本
- [ ] 1.3 — 验证命令与期望输出
- [ ] 4.4 — 一致性测试与容差
- [ ] 5.1 — YAML 字段说明
- [ ] 5.2 — 测速命令与结果模板

### Screenshots Needed (Product Demo)
- （无）
