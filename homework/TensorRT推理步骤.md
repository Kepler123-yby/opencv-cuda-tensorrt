# TensorRT 推理步骤与原理

本文结合 `run_infer.cpp` 示例，介绍 TensorRT 推理流程及各步骤的作用。

## 流程概览

**读取 Engine → 创建运行时对象 → 检查输入输出 → 分配资源 → 准备输入 → 执行推理 → 读取输出**

## 1. 准备序列化 Engine

推理使用已经构建好的 `.engine` 文件。该文件保存 TensorRT 构建阶段生成的优化结果，例如网络结构、权重和执行实现。

示例通过 `read_engine()` 以二进制方式读取文件，再调用：

```cpp
runtime_->deserializeCudaEngine(engine_date_.data(), engine_date_.size());
```

将序列化数据恢复为可执行的 Engine。Engine 通常与构建它时的 TensorRT、CUDA、GPU 等环境相关，部署时要注意版本和硬件兼容性。

## 2. 创建 Runtime、Engine 和 ExecutionContext

示例依次创建三个 TensorRT 对象：

```cpp
runtime_.reset(nvinfer1::createInferRuntime(logger_));
engine_.reset(runtime_->deserializeCudaEngine(...));
context_.reset(engine_->createExecutionContext());
```

它们的职责不同：

- **Runtime**：提供反序列化 Engine 所需的运行环境。
- **Engine**：保存优化后的模型和执行计划，可理解为模型的可执行模板。
- **ExecutionContext**：承载一次推理所需的执行状态，例如张量地址、运行时形状和执行过程。

如果同一 Engine 需要并发推理，通常应为每个并发执行准备独立的 ExecutionContext 和相关资源。当前示例复用单个 context 和缓冲区，因此同一个 `TRTInfer` 实例不能并发调用。

## 3. 检查 Engine 的输入输出

`Information::getInformation()` 遍历 Engine 的 I/O 张量，查询张量名称、输入/输出类型、形状、数据类型、格式和所在位置。

示例限定只支持**一个输入、一个输出**，且要求张量为 **FP32、LINEAR、DEVICE**，并排除形状推导 I/O。这样的检查使后续的数据大小计算、内存分配和拷贝方式保持明确一致；不符合条件时立即报错。

## 4. 确认形状并计算缓冲区大小

代码通过 Engine 查询输入维度，并拒绝未确定的维度；再从 ExecutionContext 获取输入输出维度，通过 `get_elements()` 计算元素数量。

例如，形状 `[1, 3, 640, 640]` 的元素数是 `1 × 3 × 640 × 640`。FP32 每个元素占 `sizeof(float)` 字节，因此缓冲区大小为：

```text
元素数量 × sizeof(float)
```

准确计算缓冲区大小很重要：空间不足会造成内存错误；维度未解析或乘法溢出则可能导致错误分配。

## 5. 创建 CUDA Stream 和输入输出缓冲区

示例创建非阻塞 CUDA stream，并为输入、输出各分配一份 GPU 缓冲区和一份页锁定主机缓冲区：

```cpp
cudaStreamCreateWithFlags(&buffer_.stream, cudaStreamNonBlocking);
cudaMalloc(&buffer_.d_in, in_bytes);
cudaMalloc(&buffer_.d_out, out_bytes);
cudaMallocHost(&buffer_.h_in, in_bytes);
cudaMallocHost(&buffer_.h_out, out_bytes);
```

- `d_in`、`d_out` 位于 GPU 上，供 TensorRT 执行使用。
- `h_in`、`h_out` 是页锁定（pinned）主机内存，适合 CUDA 异步传输。

这些缓冲区在构造时分配、推理时重复使用，避免每次推理都重新申请 GPU 内存。

## 6. 准备并传入输入数据

测试入口用 `std::fill_n` 构造连续的 FP32 伪输入：

```cpp
std::vector<float> input(infer.inputSize());
std::fill_n(input.begin(), input.size(), 1.0f);
```

调用 `infer()` 后，输入先复制到页锁定主机缓冲区，再通过异步 CUDA 拷贝传到 GPU：

```cpp
std::memcpy(buffer_.h_in, input, bytes);
cudaMemcpyAsync(buffer_.d_in, buffer_.h_in, bytes,
                cudaMemcpyHostToDevice, buffer_.stream);
```

TensorRT 接收的数据必须符合 Engine 的输入形状、数据类型和布局。`fill_n` 仅用于测试推理框架流程，不代表真实模型需要的预处理。

## 7. 绑定地址并执行推理

ExecutionContext 需要知道每个 I/O 张量对应的实际内存地址：

```cpp
context_->setTensorAddress(im_.input, buffer_.d_in);
context_->setTensorAddress(im_.output, buffer_.d_out);
```

绑定后，通过以下调用将推理任务提交到 CUDA stream：

```cpp
context_->enqueueV3(buffer_.stream);
```

输入拷贝、推理执行和后续输出拷贝按同一 stream 的顺序执行，因此通常不需要在每个步骤之间分别等待。

## 8. 将输出取回并等待完成

推理完成后，示例将 GPU 输出异步复制到页锁定主机缓冲区，然后同步 stream：

```cpp
cudaMemcpyAsync(buffer_.h_out, buffer_.d_out, output_bytes,
                cudaMemcpyDeviceToHost, buffer_.stream);
cudaStreamSynchronize(buffer_.stream);
```

同步确保 GPU 上的推理与拷贝都已完成，之后 CPU 才能安全读取输出。最后将结果复制到调用方的 `std::vector<float>`。

循环推理时，可以调用接收 `std::vector<float>&` 的重载来复用输出缓冲区，减少主机侧重复分配；返回新 `std::vector<float>` 的重载则更方便使用。

## 9. 解释输出

伪输入测试可以确认 Engine 是否成功加载、推理是否执行，以及输出大小是否符合预期。它不能验证模型精度或检测结果是否正确：真实应用还需要按模型要求预处理输入，并依据输出张量的形状和语义进行后处理。

## 数据流

```text
CPU 输入
  → 页锁定主机输入缓冲区
  → GPU 输入缓冲区
  → TensorRT ExecutionContext 推理
  → GPU 输出缓冲区
  → 页锁定主机输出缓冲区
  → CPU 输出
```

**Engine** 决定模型如何执行，**ExecutionContext** 管理本次执行状态，**CUDA stream** 组织异步任务顺序，**缓冲区** 承载 CPU 与 GPU 之间的数据交换。
