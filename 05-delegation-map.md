# 委派映射：OpenCV-CUDA 与 TensorRT 推理部署实战

> Phase 5 产物。定义每个课时「谁来做」：Agent（我）负责结构化、讲解、代码走读骨架与文字；Human（你）负责真实环境验证、版本确认、领域纠错与关键代码背书。

## 角色定义

| 角色 | 职责 |
|---|---|
| **Agent** | 课程结构、讲解文字、示例代码骨架、引导问题、一致性检查、文档整理 |
| **Human** | 在真实 GPU 环境运行、确认版本矩阵、核对 API 签名、领域纠错、最终审校 |
| **Joint** | 先由 Agent 出稿，Human 在真实环境验证后共同定稿 |

## 总览

| # | SLT | 课时类型 | 主责 | 你需要投入 |
|---|-----|---------|------|-----------|
| 1.1 | 编译 CUDA 版 OpenCV | How To Guide | Joint | 确认本机/目标机版本与编译命令 |
| 1.2 | CMake 后端检测 | Developer Documentation | Joint | 核对 `find_package` 路径 |
| 1.3 | 运行 GPU 示例验证 | How To Guide | Human | 在真实 GPU 跑通并给输出 |
| 2.1 | GpuMat 上传下载 | Developer Documentation | Agent | 复核示例可编译 |
| 2.2 | cv::cuda 预处理算子 | Developer Documentation | Joint | 对照真实 API 签名 |
| 2.3 | 往返瓶颈原理 | Exploration | Agent | 审阅引导问题 |
| 3.1 | ONNX→engine 构建/缓存 | Developer Documentation | Joint | 确认 TensorRT 版本 + API（✅ 已确认 10.10.0.31） |
| 3.2 | 显存/pinned/流执行 | Developer Documentation | Joint | 核对 CUDA 版本与签名 |
| 3.3 | 同步/异步/多路并行 | Developer Documentation | Joint | 真实多流跑通、验证保序 |
| 3.4 | 串行 vs 并行对比 | Exploration | Agent | 审阅取舍判断 |
| 4.1 | 融合预处理 kernel | Developer Documentation | Joint | 真机编译并验证输出 |
| 4.2 | 并行解码 kernel + atomicAdd | Developer Documentation | Joint | 真机编译并验证结果 |
| 4.3 | 零拷贝设计评估 | Exploration | Agent | 审阅结论 |
| 4.4 | CPU/GPU 一致性验证 | How To Guide | Human | 运行一致性测试并给容差 |
| 5.1 | YAML 切换后端 | How To Guide | Agent | 确认字段命名 |
| 5.2 | 延迟/吞吐测量 | How To Guide | Joint | 真实测速并给结果表 |
| 5.3 | 独立编写调试完整项目 | Developer Documentation | Joint | 作为结课项目验收 |

## 逐课时委派

### Module 1 · 环境与工具链

**1.1 编译 CUDA 版 OpenCV（Joint）**
- Agent：整理编译 flags、cuDNN 获取步骤、独立前缀策略、环境脚本骨架。
- Human：确认目标机的 CUDA / cuDNN / OpenCV 版本组合；在真机跑通编译。
- 验收：`pkg-config --modversion opencv4` 与 `cv2.cuda.getCudaEnabledDeviceCount()` 有输出。

**1.2 CMake 后端检测（Joint）**
- Agent：写 `find_package` / target 检测片段与说明。
- Human：核对 `OpenCV_DIR` / `TENSORRT_ROOT` / `OPENVINO_DIR` 在本机的实际路径。
- 验收：配置阶段能打印出检测到的后端列表。

**1.3 运行 GPU 示例验证（Human 主责）**
- Agent：给出示例命令与期望输出模板。
- Human：真机运行，回填实际输出与常见报错。
- 验收：设备查询 + 一个 GPU 算子跑通。

### Module 2 · GPU 挂载与数据搬运

**2.1 GpuMat 上传下载（Agent 主责）**
- Agent：写最小 `GpuMat::upload/download` 示例与 H2D/D2H 说明。
- Human：复核可编译。
- 验收：示例输出上传/下载后的尺寸与耗时。

**2.2 cv::cuda 预处理算子（Joint）**
- Agent：写 `cv::cuda::resize` + `copyMakeBorder` 预处理示例。
- Human：对照真实 OpenCV API 签名复核。
- 验收：GPU 预处理输出与 CPU 版本视觉一致。

**2.3 往返瓶颈原理（Agent 主责）**
- Agent：写原理讲解与引导问题。
- Human：审阅问题是否切中。
- 验收：无需代码产物。

### Module 3 · TensorRT 推理

**3.1 ONNX→engine 构建/缓存（Joint · 高风险）**
- Agent：写 Builder/Parser/Runtime/Engine 代码骨架与缓存逻辑。
- Human：**确认 TensorRT 版本与 API 名称**（如 `enqueueV3` / `setTensorAddress`），核对签名。✅ 已确认：**TensorRT 10.10.0.31**，头文件 API 已核对。
- 验收：ONNX 构建出 engine 并落盘缓存，二次加载走缓存。

**3.2 显存/pinned/流（Joint）**
- Agent：写 `cudaMalloc` / `cudaMallocHost` / `enqueueV3` 示例。
- Human：核对 CUDA 版本与 `enqueueV3` 签名。
- 验收：一次推理输出与参考一致。

**3.3 同步/异步/多路并行（Joint · 已出稿，见 `lessons/module-3/`）**
- Agent：已完成三模式讲解、代码骨架、踩坑清单、验证步骤。
- Human：真机跑多路并行，确认吞吐上升且无错帧。
- 验收：300 帧跑出 同步 / 异步 / 并行 的 FPS 对比。

**3.4 串行 vs 并行对比（Agent 主责）**
- Agent：已完成对比、阶段分析、引导问题。
- Human：审阅取舍判断是否符合你的场景。
- 验收：无需代码产物。

### Module 4 · 前后处理 GPU 化

**4.1 融合预处理 kernel（Joint）**
- Agent：写融合 letterbox + 通道转换 + 归一化的 kernel 骨架。
- Human：真机编译、验证写入设备缓冲区正确。
- 验收：融合结果与 CPU 预处理数值一致（容差内）。

**4.2 并行解码 kernel + atomicAdd（Joint）**
- Agent：写解码 kernel 与 `atomicAdd` 压缩骨架。
- Human：真机编译并验证候选数量与坐标正确。
- 验收：GPU 解码与 CPU 解码结果一致。

**4.3 零拷贝设计评估（Agent 主责）**
- Agent：写零拷贝收益与边界的讲解。
- Human：审阅结论。
- 验收：无需代码产物。

**4.4 CPU/GPU 一致性验证（Human 主责）**
- Agent：给出一致性测试命令与容差模板。
- Human：真机运行并回填实际容差。
- 验收：一致性测试通过，输出容差值。

### Module 5 · 部署、调试与性能

**5.1 YAML 切换后端（Agent 主责）**
- Agent：写 YAML 字段说明与示例。
- Human：确认字段命名与实际一致。
- 验收：改 YAML 无需重编译即可切后端。

**5.2 延迟/吞吐测量（Joint）**
- Agent：写测速命令、参数、结果表模板。
- Human：真机跑出 CPU / CUDA / TensorRT 三组数据。
- 验收：结果表填写完整、有 FPS 与平均耗时。

**5.3 独立编写调试完整项目（Joint · capstone）**
- Agent：写项目骨架、接口说明、调试清单。
- Human：作为结课项目验收——学员能独立跑通。
- 验收：一个可运行的 TensorRT + OpenCV-CUDA 最小项目。

## 需要你优先提供的三件事

| # | 事项 | 解锁 | 优先级 |
|---|------|------|--------|
| 1 | ~~TensorRT 版本号 + 关键 API 确认~~ ✅ **已确认 10.10.0.31** | 3.1, 3.2, 3.3 | ~~High~~ 完成 |
| 2 | **TensorRT 多流并行真机验证** | 3.3 | High |
| 3 | **真机 CUDA / cuDNN / OpenCV 版本矩阵** | 1.1, 2.2, 3.x | Medium |

> 其余部分（讲解、骨架、引导问题、对比分析）我可以直接产出；你主要负责「在真机上说不说得通」。
