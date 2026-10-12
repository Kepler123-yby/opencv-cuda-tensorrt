# Coaching Readiness Assessment

## Summary

- **Total SLTs assessed**: 17
- **Distribution**: 17 Ready / 0 Needs Context / 0 Needs Human
- **Overview**: 主题为**主流、长期稳定**的库（OpenCV CUDA / TensorRT / CUDA Runtime），且用户提供了**完整可运行的参考代码库**作为 ground truth。TensorRT 版本已确认为 **10.10.0.31** 并已核对头文件 API，原先唯一的缺口（3.3 多路异步）已解除，全部 SLT 就绪。

### Calibration Note

基于以往评估数据的调整已应用：
- **Exploration SLT 在文档完善的概念上可靠地自评为 Ready** → 2.3、3.4、4.3 直接判 Ready。
- **Developer Documentation 涉及 API 端点是高幻觉区**，必须对照真实代码库审计 → 本课程**已持有参考仓库**，所有代码类 SLT 均以仓库实现为锚点。
- **代码库/模板访问可替代「去取文档」** → 参考仓库即最高杠杆的 context。

## Readiness by Module

| Module | Ready | Needs Context | Needs Human | Notes |
|--------|-------|---------------|-------------|-------|
| 1 环境与工具链 | 3 | 0 | 0 | 构建 flags 稳定；仓库记录了精确命令 |
| 2 GPU 挂载与数据搬运 | 3 | 0 | 0 | OpenCV CUDA API 稳定，仓库有实现 |
| 3 TensorRT 推理 | 4 | 0 | 0 | TensorRT 10.10.0.31 API 已核对，多 context 并行可落地 |
| 4 前后处理 GPU 化 | 4 | 0 | 0 | CUDA kernel 模式由仓库完整锚定 |
| 5 部署、调试与性能 | 3 | 0 | 0 | 配置/测速/接口均由仓库覆盖 |

## Per-SLT Assessment

### Module 1: 环境与工具链

| # | SLT | Conceptual | Code Demo | Assessment | Currency | Overall |
|---|-----|-----------|-----------|------------|----------|---------|
| 1.1 | 从源码编译启用 CUDA 与 cuDNN 的 OpenCV，并安装到独立前缀与系统 CPU 版共存 | Strong | Strong | Strong | Current | **Ready** |
| 1.2 | 编写 CMake 配置自动检测推理后端与 CUDA 能力，并编译进同一个库 | Strong | Strong | Strong | Current | **Ready** |
| 1.3 | 运行一个 GPU 图像处理示例，验证 CUDA 设备与算子已正确启用 | Strong | Strong | Strong | Current | **Ready** |

### Module 2: GPU 挂载与数据搬运

| # | SLT | Conceptual | Code Demo | Assessment | Currency | Overall |
|---|-----|-----------|-----------|------------|----------|---------|
| 2.1 | 用 GpuMat 在主机与设备间上传和下载图像数据 | Strong | Strong | Strong | Current | **Ready** |
| 2.2 | 用 cv::cuda 的缩放与 padding 算子在 GPU 上完成预处理 | Strong | Strong | Strong | Current | **Ready** |
| 2.3 | 解释 host↔device 往返如何成为 GPU 预处理的性能瓶颈 | Strong | N/A | Strong | Current | **Ready** |

### Module 3: TensorRT 推理

| # | SLT | Conceptual | Code Demo | Assessment | Currency | Overall |
|---|-----|-----------|-----------|------------|----------|---------|
| 3.1 | 从 ONNX 构建并缓存 TensorRT engine，并反序列化用于推理 | Strong | Strong | Strong | Current* | **Ready** |
| 3.2 | 分配输入输出显存与页锁定内存，并在 CUDA 流上执行推理 | Strong | Strong | Strong | Current* | **Ready** |
| 3.3 | 实现同步、异步与多路异步三种推理模式 | Strong | Strong | Strong | Current | **Ready** |
| 3.4 | 比较串行与多路异步流水线在延迟与吞吐上的差异 | Strong | N/A | Strong | Current | **Ready** |

> \* TensorRT C++ API 跨版本变动较大（`enqueueV2`→`enqueueV3`、`setBinding`→`setTensorAddress`）。已确认本环境为 **TensorRT 10.10.0.31**，并核对了 `~/tensorrt/TensorRT-10.10.0.31/include` 中的头文件（`enqueueV3` / `setTensorAddress` / `MemoryPoolType::kWORKSPACE` 等），因此 Currency 判为 Current。

#### Gap Analysis

**SLT 3.3: "我能实现同步、异步与多路异步三种推理模式。"**
- ✅ **已解决**。TensorRT 版本确认为 **10.10.0.31**，多 context + 多 stream 并行是 10.x 标准能力，API（`enqueueV3` / `setTensorAddress` / `createExecutionContext`）已对照头文件核对。
- 课时 `lessons/module-3/3.3-sync-async-parallel.md` 已出稿，含三模式代码骨架与踩坑清单。
- 剩余动作仅为**真机验证**（buffer 隔离、保序、同步），属课时构建/验证环节，不是知识缺口。

### Module 4: 前后处理 GPU 化

| # | SLT | Conceptual | Code Demo | Assessment | Currency | Overall |
|---|-----|-----------|-----------|------------|----------|---------|
| 4.1 | 编写融合 letterbox、通道转换与归一化的 CUDA kernel，直接写入设备输入缓冲区 | Strong | Strong | Strong | Current | **Ready** |
| 4.2 | 编写并行解码检测候选的 CUDA kernel，并用原子操作压缩结果 | Strong | Strong | Strong | Current | **Ready** |
| 4.3 | 评估零拷贝设计如何消除 host↔device 往返 | Strong | N/A | Strong | Current | **Ready** |
| 4.4 | 验证 GPU 前后处理与 CPU 路径的结果一致性 | Strong | Strong | Strong | Current | **Ready** |

### Module 5: 部署、调试与性能

| # | SLT | Conceptual | Code Demo | Assessment | Currency | Overall |
|---|-----|-----------|-----------|------------|----------|---------|
| 5.1 | 通过 YAML 配置在运行时切换推理后端与 CUDA 开关 | Strong | Strong | Strong | Current | **Ready** |
| 5.2 | 测量端到端延迟与吞吐，对比 CPU、CUDA 与 TensorRT 配置 | Strong | Strong | Strong | Current | **Ready** |
| 5.3 | 独立编写并调试一个 TensorRT + OpenCV-CUDA 项目，复用部署库的模式 | Strong | Strong | Strong | Current | **Ready** |

## Lesson Prioritization

### Tier 1: Build First（Ready）

| # | SLT | Module | Rationale |
|---|-----|--------|-----------|
| 1.2, 2.1, 2.2, 3.1, 3.2, 4.1, 4.2, 5.3 | — | 1–5 | Developer Documentation，全部有仓库代码锚定 |
| 1.1, 1.3, 4.4, 5.1, 5.2 | — | 1/4/5 | How To Guide，仓库 README/命令可直接引用 |
| 2.3, 3.4, 4.3 | — | 2–4 | Exploration，文档完善的概念 |

### Tier 2: Build with Context（Needs Context）

（无）

### Tier 3: Defer（Needs Human）
- （无）

## Context Shopping List

| Resource | Type | SLTs Unlocked | Priority |
|----------|------|---------------|----------|
| 参考仓库源码（`26_NNDeployment_Lib_and_Detection_Models`，已通过文件系统获取） | Example Code | 1.2, 2.1, 2.2, 3.1, 3.2, 4.1, 4.2, 5.3（及大部分 SLT） | High（**已获取**） |
| TensorRT / CUDA / cuDNN / OpenCV 版本矩阵 | Version Confirmation | 1.1, 3.1, 3.2, 3.3 | Medium（**TensorRT 10.10.0.31 已确认**） |
| 仓库 README + `docs/`（环境、配置、性能结论） | Docs | 1.1, 1.3, 4.4, 5.1, 5.2 | Medium（**已获取**） |

## Set-Level Observations

- **Strength clusters**：Module 2 与 Module 4 最稳——OpenCV CUDA API 长期稳定，且仓库提供了从 `GpuMat` 到融合 kernel 的完整实现。
- **Weakness cluster**：原先唯一弱点 **TensorRT 并行推理（3.3）** 已随 TensorRT 10.10.0.31 版本确认 + 头文件 API 核对而解除。
- **Critical path**：无阻塞项。可直接进入课时编写。
- **Sequencing implications**：建议按 M1→M2→M4→M3→M5 编写（先建立 GpuMat 与 kernel 直觉，再切入 TensorRT 的并行复杂度），或保持 M1→M5 原顺序。
- **Confabulation risk zones**：TensorRT API 版本细节已通过头文件核对消除；剩余风险仅在于**真机上的 buffer 隔离与流同步正确性**，建议每课包含可运行的验证步骤。
