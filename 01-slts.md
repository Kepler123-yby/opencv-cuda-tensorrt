# SLTs: OpenCV-CUDA 与 TensorRT 推理部署实战

> 学员画像：有 OpenCV / Python 使用基础，用过 CUDA 与 TensorRT，但未深入其原理；
> 目标：能独立编写并调试 TensorRT + OpenCV-CUDA 项目。
> 参考代码库：`rm_2027/other_auto_aim_project/26_NNDeployment_Lib_and_Detection_Models`

## Module 1: 环境与工具链

- 1.1: 我能从源码编译启用 CUDA 与 cuDNN 的 OpenCV，并安装到独立前缀与系统 CPU 版共存。
- 1.2: 我能编写 CMake 配置自动检测推理后端与 CUDA 能力，并编译进同一个库。
- 1.3: 我能运行一个 GPU 图像处理示例，验证 CUDA 设备与算子已正确启用。

## Module 2: GPU 挂载与数据搬运

- 2.1: 我能用 GpuMat 在主机与设备间上传和下载图像数据。
- 2.2: 我能用 cv::cuda 的缩放与 padding 算子在 GPU 上完成预处理。
- 2.3: 我能解释 host↔device 往返如何成为 GPU 预处理的性能瓶颈。

## Module 3: TensorRT 推理

- 3.1: 我能从 ONNX 构建并缓存 TensorRT engine，并反序列化用于推理。
- 3.2: 我能分配输入输出显存与页锁定内存，并在 CUDA 流上执行推理。
- 3.3: 我能实现同步、异步与多路异步三种推理模式。
- 3.4: 我能比较串行与多路异步流水线在延迟与吞吐上的差异。

## Module 4: 前后处理 GPU 化

- 4.1: 我能编写融合 letterbox、通道转换与归一化的 CUDA kernel，直接写入设备输入缓冲区。
- 4.2: 我能编写并行解码检测候选的 CUDA kernel，并用原子操作压缩结果。
- 4.3: 我能评估零拷贝设计如何消除 host↔device 往返。
- 4.4: 我能验证 GPU 前后处理与 CPU 路径的结果一致性。

## Module 5: 部署、调试与性能

- 5.1: 我能通过 YAML 配置在运行时切换推理后端与 CUDA 开关。
- 5.2: 我能测量端到端延迟与吞吐，对比 CPU、CUDA 与 TensorRT 配置。
- 5.3: 我能独立编写并调试一个 TensorRT + OpenCV-CUDA 项目，复用部署库的模式。
