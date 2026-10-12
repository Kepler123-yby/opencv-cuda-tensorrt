---
course: OpenCV-CUDA 与 TensorRT 推理部署实战
status: building
lesson_drafts: complete
integration_validation: pending
last_updated: 2026-10-10
audience: 有 OpenCV / Python 基础、用过 CUDA 与 TensorRT 但未深入原理的开发者
reference_repo: rm_2027/other_auto_aim_project/26_NNDeployment_Lib_and_Detection_Models
tensorrt: 10.10.0.31
---

# OpenCV-CUDA 与 TensorRT 推理部署实战

## 课程目标

学完后，学员能**独立编写并调试 TensorRT + OpenCV-CUDA 项目**，覆盖：
环境搭建、GPU 挂载（GpuMat）、预处理/后处理 GPU 化、TensorRT 串/并行推理、部署与性能评估。

## 模块与 SLT

### Module 1: 环境与工具链（How To Guide / Developer Documentation）
- 1.1 我能从源码编译启用 CUDA 与 cuDNN 的 OpenCV，并安装到独立前缀与系统 CPU 版共存。
- 1.2 我能编写 CMake 配置自动检测推理后端与 CUDA 能力，并编译进同一个库。
- 1.3 我能运行一个 GPU 图像处理示例，验证 CUDA 设备与算子已正确启用。

### Module 2: GPU 挂载与数据搬运（Developer Documentation / Exploration）
- 2.1 我能用 GpuMat 在主机与设备间上传和下载图像数据。
- 2.2 我能用 cv::cuda 的缩放与 padding 算子在 GPU 上完成预处理。
- 2.3 我能解释 host↔device 往返如何成为 GPU 预处理的性能瓶颈。

### Module 3: TensorRT 推理（Developer Documentation / Exploration）
- 3.1 我能从 ONNX 构建并缓存 TensorRT engine，并反序列化用于推理。
- 3.2 我能分配输入输出显存与页锁定内存，并在 CUDA 流上执行推理。
- 3.3 我能实现同步、异步与多路异步三种推理模式。
- 3.4 我能比较串行与多路异步流水线在延迟与吞吐上的差异。

### Module 4: 前后处理 GPU 化（Developer Documentation / Exploration / How To Guide）
- 4.1 我能编写融合 letterbox、通道转换与归一化的 CUDA kernel，直接写入设备输入缓冲区。
- 4.2 我能编写并行解码检测候选的 CUDA kernel，并用原子操作压缩结果。
- 4.3 我能评估零拷贝设计如何消除 host↔device 往返。
- 4.4 我能验证 GPU 前后处理与 CPU 路径的结果一致性。

**模块作业**：提交融合预处理、并行解码与 CPU NMS 的实验，附数据流图、边界一致性测试和性能 A/B；详细标准见 `lessons/module-4/4.4-cpu-gpu-parity.md`。

### Module 5: 部署、调试与性能（How To Guide / Developer Documentation）
- 5.1 我能通过 YAML 配置在运行时切换推理后端与 CUDA 开关。
- 5.2 我能测量端到端延迟与吞吐，对比 CPU、CUDA 与 TensorRT 配置。
- 5.3 我能独立编写并调试一个 TensorRT + OpenCV-CUDA 项目，复用部署库的模式。

**综合作业**：独立实现可配置的视频检测工程，验证单路/多路的结果对齐、背压/drain 与资源生命周期，提交正确性、排错和性能证据；详细范围、接口、CMake 参考及评分见 `lessons/module-5/5.3-capstone-project.md`。

## 已出稿课时

| 课时 | 文件 | 类型 |
|---|---|---|
| 1.1 从源码编译启用 CUDA 的 OpenCV | `lessons/module-1/1.1-build-opencv-cuda.md` | How To Guide |
| 1.2 用 CMake 自动检测推理后端 | `lessons/module-1/1.2-cmake-backend-detect.md` | Developer Documentation |
| 1.3 验证 GPU 链路：跑一个示例 | `lessons/module-1/1.3-verify-gpu.md` | How To Guide |
| 2.1 GpuMat 与主机↔设备数据搬运 | `lessons/module-2/2.1-gpumat-transfer.md` | Developer Documentation |
| 2.2 在 GPU 上做预处理：缩放与 letterbox padding | `lessons/module-2/2.2-cuda-preprocess.md` | Developer Documentation |
| 2.3 往返的代价：为什么 GPU 预处理不总是更快 | `lessons/module-2/2.3-roundtrip-cost.md` | Exploration |
| 3.1 从 ONNX 构建并缓存 TensorRT engine | `lessons/module-3/3.1-build-engine.md` | Developer Documentation |
| 3.2 分配缓冲区，在 CUDA 流上执行推理 | `lessons/module-3/3.2-buffers-and-stream.md` | Developer Documentation |
| 3.3 同步、异步与多路并行推理 | `lessons/module-3/3.3-sync-async-parallel.md` | Developer Documentation |
| 3.4 串行与多路异步流水线：延迟还是吞吐 | `lessons/module-3/3.4-serial-vs-parallel.md` | Exploration |
| 4.1 融合预处理 kernel：直接写入 TensorRT 输入显存 | `lessons/module-4/4.1-fused-preprocess-kernel.md` | Developer Documentation |
| 4.2 GPU 并行解码：候选筛选与 atomicAdd 压缩 | `lessons/module-4/4.2-parallel-decode-kernel.md` | Developer Documentation |
| 4.3 「零拷贝」的工程含义：让中间数据留在 GPU | `lessons/module-4/4.3-device-resident-pipeline.md` | Exploration |
| 4.4 验证 CPU/GPU 一致性：先正确，再提速 | `lessons/module-4/4.4-cpu-gpu-parity.md` | How To Guide |
| 5.1 运行时配置：切换后端，不让 CUDA 开关含糊 | `lessons/module-5/5.1-runtime-config.md` | How To Guide |
| 5.2 测量端到端延迟与吞吐：计时必须覆盖完成 | `lessons/module-5/5.2-benchmark-pipeline.md` | How To Guide |
| 5.3 综合项目：独立组织、调试和交付 GPU 推理流水线 | `lessons/module-5/5.3-capstone-project.md` | Developer Documentation |

**全部 17 个课时初稿已齐全。** `status: building` 保留，表示完整独立样例、集成验证与整课审校尚未完成，不等于已发布或学员作业已验收。

## Module 4 验证状态

独立 kernel 与比较脚本已在本机运行；预处理 7 项、解码 7 项、比较脚本 7 项通过，memcheck 零错误。完整 TensorRT 集成、异步压力测试与端到端性能尚未验证，课时仍为草稿。

- 记录：`assets/code-examples/module-4-verification.md`
- 复现：`assets/code-examples/tests/verify-module4.py`
- 辅助比较：`assets/code-examples/4.4-compare-tensors.py`

## Module 5 验证状态

辅助工具已本机验证：配置 26 项、人工计时统计 15 项、C++ 接口/调度/shape 片段语法检查 3 项通过。已修复 YAML null 被转成字符串的问题。没有加载真实 ONNX/engine，未把统计示例当性能实测，也未宣称完整 detector 可直接运行。

- 配置：`assets/configs/5.1-deploy.yaml`
- 工具与接口：`assets/code-examples/5.1-validate-config.cpp`、`assets/code-examples/5.2-summarize-benchmark.py`、`assets/code-examples/5.3-pipeline-contract.hpp`
- 复现：`assets/code-examples/tests/verify-module5.py`
- 范围与结果：`assets/code-examples/module-5-verification.md`

## 工作流状态

| 阶段 | 状态 | 产物 |
|---|---|---|
| 1 SLT 起草 | ✅ | `01-slts.md` |
| 2 质量校准 | ✅ | `02-slts-quality-review.md` |
| 3 课时类型分类 | ✅ | `03-lesson-type-classification.md` |
| 4 备课就绪度 | ✅ | `04-readiness-assessment.md` |
| 5 委派映射 | ✅ | `05-delegation-map.md` |
| 6 素材收集 | ⏳ | kernel 测试、比较/配置/统计工具与接口已补；完整独立工程样例仍待完成 |
| 7 课时编写 | ✅ 初稿 | Module 1–5 已出稿（17/17）；整课审校与集成验证仍待完成 |
| 8 编译导入 | — | — |
| 9 回顾与沉淀 | — | — |
