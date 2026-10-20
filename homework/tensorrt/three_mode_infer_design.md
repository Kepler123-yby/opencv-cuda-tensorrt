# three_mode_infer 程序定义

探究 TensorRT 三种推理方式（同步 / 异步 / 多路并行）的实现与原理。

- 目标文件：`three_mode_infer.cpp` / `three_mode_infer.hpp`
- 测试文件：`test_three_mode_infer.cpp`（`./build_test.sh` 一键编译 + 运行）
- 参考文件：`run_infer.cpp`、`build_engine.cpp`
- 实现状态：**已定稿**。三个接口共用同一个执行器（固定 worker + 有界队列，见 §8）；
  早期"通道池 + 每请求线程"方案已删除，保留在**附录 A** 供对照。
- 实测环境：RTX 3060 Ti / YOLOv8n FP16 / 输入 480×640×3（`model/model.engine`），数据见 §11

---

## 1. 设计决策

**一个类 `Inference` 包装三种方法，不引入多态。**

理由：三种方式的差异在“谁执行、何时等待、多个请求如何分配资源”，不在接口抽象。多态解决的是“调用方不知道具体实现也能统一调用”，这里不需要。

### 1.1 先区分三个层次的“异步”

| 层面 | 问题 | 由什么解决 |
| --- | --- | --- |
| C++ 调用是否阻塞 | 调用线程要不要等结果 | `std::future` |
| CUDA 调用是否阻塞 CPU | 提交后 CPU 能否返回 | `cudaMemcpyAsync` + stream |
| GPU 是否真的并行 | 多个请求能否重叠执行 | 多 context + 多 stream + 多套缓冲区 |

> `future` 解决 CPU 端结果交付，stream 解决 GPU 端任务排序，多套 context 和缓冲区解决并发请求的状态隔离。

### 1.2 并行真正的最小单位

并行单位不是 context，而是：

> **context + 该 context 私有的输入/输出缓冲区 + 一条可以独立入队的 stream**

两条硬约束决定了这一点：

1. **一个 stream 内部严格有序。** 同 stream 上先后 `enqueueV3`，后一个必须等前一个完成 —— context 再多也不重叠。
2. **`setTensorAddress()` 是 per-context 的，但地址由调用方提供。** 两个 context 指向同一块 `d_out`，并发时结果互相覆盖。

各种组合：

| 组合 | context | stream | 缓冲区 | 结果 |
| --- | --- | --- | --- | --- |
| 1 | 1 | 1 | 1 套 | 顺序，正确 |
| 2 | N | 1 | 1 套 | 仍顺序（stream 串行），还有地址冲突隐患 |
| 3 | N | N | 1 套 | 真并发但结果互相覆盖，**错误** |
| 4 | N | N | N 套 | 正确且可重叠 |

因此：**并发数 N ⇒ N 条执行通道（channel）**。单条通道只能走顺序路径。

---

## 2. 整体结构

文件内的组件（**最终实现**）：

```text
Logger              继承 nvinfer1::ILogger，转发 TensorRT 日志（加锁：可能来自内部线程）
IOInfo              记录输入/输出张量名、元素数、字节数
Channel             一条执行通道（context + stream + event + 四块缓冲区）
WorkerPool          执行器：concurrency 个 worker + 有界任务队列，每个 worker 独占一条 Channel
Inference           对外主类：构造 + infer / inferAsync / inferParallel
```

主类骨架：

```cpp
class Inference
{
public:
    explicit Inference(const InferConfig& config);
    ~Inference();                       // 排空队列、join worker 之后才销毁 engine

    Inference(const Inference&) = delete;
    Inference& operator=(const Inference&) = delete;

    Output                           infer(const Input& input);        // 同步
    std::future<Output>              inferAsync(Input input);         // 异步
    std::vector<std::future<Output>> inferParallel(std::vector<Input> inputs);

    size_t inputSize()   const;
    size_t outputSize()  const;
    int    concurrency() const;

private:
    std::vector<char> readEngine(const std::string& engine_path);

    InferConfig                            config_;
    std::vector<char>                      engine_data_;
    Logger                                 logger_;
    std::unique_ptr<nvinfer1::IRuntime>    runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    IOInfo                                 info_;
    std::unique_ptr<WorkerPool>            workers_;   // 最后声明 → 最先析构
};
```

`WorkerPool` 只在本 `.hpp` 里前向声明，定义藏在 `.cpp`：外部不需要知道队列和线程的存在。

三个接口的差异只在"**谁等、何时等**"，执行路径完全共用：

```text
infer()          → workers_->submitBorrowed(input).get()   // 提交后立刻等（借用输入，0 拷贝）
inferAsync()     → workers_->submit(std::move(input))      // 只提交，何时等由调用方决定
inferParallel()  → 循环 submit，返回 future 数组            // 一次提交一批
```

---

## 3. 接口定义

```cpp
using Input  = std::vector<float>;
using Output = std::vector<float>;

struct InferConfig
{
    std::string engine_path;

    // worker(=通道) 数量，即同时在执行的请求数上限；排队数由 queueCapacity 控制
    int concurrency = 1;

    int deviceId = 0;

    // 等待队列长度（不含正在执行的请求）；0 表示默认 2 × concurrency
    size_t queueCapacity = 0;
};
```

签名与参数传递方式是有意设计的：

- `infer()` 返回时结果已可用 → 参数用 `const&`，**不复制**：任务里只存 `const Input*` 借用。
- `inferAsync()` 函数返回后任务还在跑 → 参数**按值接收**，调用者用 `std::move(input)` 把数据移进任务，此时 0 拷贝。
- `inferAsync()` 收到左值就必然拷贝一次 3.6 MB —— 这条路径无法既保证生命周期又不拷贝，只能要求调用方 `std::move`。
- 异步接口**不收裸指针或引用**，否则后台任务可能读到已失效的数据。
- `inferParallel()` 一次提交多个请求，返回与输入一一对应的 future。
- 不设 `InferMode` 成员：模式由"调用哪个方法"决定，避免两套选择机制。
- `concurrency` 同时决定 GPU 侧并行度**和**显存占用（每通道一套 context + 两块 device 缓冲 + 两块 pinned 缓冲）。
- `queueCapacity` 是**背压旋钮**：队列满时 `submit()` 阻塞调用方，所以它就是"允许排队多少请求"的上限。
  写 0 只是取默认值，**不存在无界队列** —— 无界会吃光内存、让延迟失控。

---

## 4. 辅助函数定义

### 4.1 CUDA 错误检查

```cpp
// 匿名 namespace：符号不外泄，多个 TU 各自一份也不会冲突
void cudaCheck(cudaError_t status, const char* call)
{
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(call) + ": " + cudaGetErrorString(status));
}

#define CHECK_CUDA(call) cudaCheck((call), #call)
```

- 宏把**表达式原文**作为第二个参数传进去，报错时能直接看到是哪一行调用失败。
- 只用于“返回值非成功即为错误”的调用。`cudaEventQuery` 的 `cudaErrorNotReady` 是正常状态，不能套这个宏。
- 析构函数里不能抛异常，所以 `Channel::release()` 里的 CUDA 调用**不加检查**，只把指针置空（尽力释放）。

### 4.2 readEngine

```cpp
std::vector<char> Inference::readEngine(const std::string& engine_path)
{
    std::ifstream file(engine_path, std::ios::binary | std::ios::ate);
    if (!file) throw std::runtime_error("读取文件失败！");

    const auto end = file.tellg();
    if (end <= 0) throw std::runtime_error("无法读取engine文件或文件为空！");

    const auto length = static_cast<std::streamoff>(end);
    const auto file_size = static_cast<uintmax_t>(length);
    if (file_size > static_cast<uintmax_t>(std::numeric_limits<size_t>::max()))
        throw std::runtime_error("Engine 文件过大！");
    if (file_size > static_cast<uintmax_t>(std::numeric_limits<std::streamsize>::max()))
        throw std::runtime_error("Engine 文件超出 streamsize 表示范围！");

    std::vector<char> engine_data(static_cast<size_t>(length));

    file.seekg(0, std::ios::beg);
    if (!file) throw std::runtime_error("文件指针定位失败！");
    if (!file.read(engine_data.data(), static_cast<std::streamsize>(length)))
        throw std::runtime_error("Engine 数据读取不完整！");

    std::cout << "Engine 文件读取成功！\n"
              << "Engine 大小: " << engine_data.size() << " bytes\n";
    return engine_data;
}
```

- `binary | ate`：二进制模式，打开即定位到末尾以取长度；读文件不需要写权限，所以用 `ifstream`。
- `seekg(0)` 回到开头再读；**读完要检查 `read()` 的返回值**，短路读取会被静默接受。
- 两次上限检查目的不同：`size_t` 决定能否安全 `resize`，`streamsize` 决定能否安全传给 `read()`。
  用 `uintmax_t` 中转避免有符号/无符号比较问题。
- 结果存进成员 `engine_data_`：`deserializeCudaEngine()` 不保证把字节拷走，构造期间必须保持有效。
- 反序列化之后这块内存就可以不再使用，但保留成员是最省心的做法（几 MB 而已）。
### 4.3 calculateElements

```cpp
size_t IOInfo::calculateElements(const nvinfer1::Dims& dims) const
{
    if (dims.nbDims <= 0)
        throw std::runtime_error("无效张量维度: " + shapeString(dims));

    size_t count = 1;
    for (int i = 0; i < dims.nbDims; ++i)
    {
        const auto dim = dims.d[i];

        if (dim < 0)          // -1 表示动态维度
            throw std::runtime_error(
                "本实现不支持动态 shape（需要 setInputShape/inferShapes），"
                "请使用静态维度 engine: " + shapeString(dims));

        if (dim == 0)
            throw std::runtime_error("本例不支持空张量: " + shapeString(dims));

        const auto size = static_cast<size_t>(dim);
        if (count > std::numeric_limits<size_t>::max() / size)
            throw std::runtime_error("Elements: 元素数溢出！");
        count *= size;
    }
    return count;
}
```

设计规则：

- **先校验有符号原始值，再转无符号**。`dims.d[i] == -1` 若先转 `size_t` 会变成极大正数，
  任何 `<= 0` 判断都失效。
- 动态维度单独报错：`-1` 的含义是"未解析"，不是"非法"，错误信息必须能区分布局问题和形状问题。
- 乘法前做溢出检查，避免回绕。
### 4.4 calculateBytes

```cpp
size_t IOInfo::calculateBytes(const size_t elements) const
{
    if (elements > std::numeric_limits<size_t>::max() / sizeof(float))
        throw std::runtime_error("Bytes: 元素数溢出！");
    return elements * sizeof(float);
}
```

只有实际 I/O 为 FP32 时才正确，这个前提由 `InspectEngine()` 建立。

### 4.5 InspectEngine

```cpp
void IOInfo::InspectEngine(const nvinfer1::ICudaEngine& engine)
{
    if (engine.getNbOptimizationProfiles() > 1)
        throw std::runtime_error("本例只支持单 optimization profile 的静态 engine");

    int inputs = 0;
    int outputs = 0;

    for (int i = 0; i < engine.getNbIOTensors(); ++i)
    {
        const auto name  = engine.getIOTensorName(i);
        const auto mode  = engine.getTensorIOMode(name);
        const auto shape = engine.getTensorShape(name);
        const auto dtype = engine.getTensorDataType(name);
        const auto fmt   = engine.getTensorFormat(name);
        const auto where = engine.getTensorLocation(name);

        std::cout << name << " shape= " << shapeString(shape)
                  << " mode= " << (mode == nvinfer1::TensorIOMode::kINPUT ? "INPUT" : "OUTPUT")
                  << " dtype= " << static_cast<int>(dtype)
                  << " format= " << engine.getTensorFormatDesc(name)
                  << " location= " << static_cast<int>(where) << std::endl;

        if (dtype != nvinfer1::DataType::kFLOAT ||
            fmt   != nvinfer1::TensorFormat::kLINEAR ||
            where != nvinfer1::TensorLocation::kDEVICE ||
            engine.isShapeInferenceIO(name))
        {
            throw std::runtime_error("本例只支持 FP32/LINEAR/DEVICE 且非形状推导 I/O: "
                                     + std::string(name));
        }

        // 静态 shape 才能在这里算出固定元素数；动态维度由 calculateElements 明确报错
        const auto elements = calculateElements(shape);
        const auto bytes    = calculateBytes(elements);

        if (mode == nvinfer1::TensorIOMode::kINPUT)
        {
            input          = name;   // std::string：不要把 engine 的 const char* 长期存起来
            input_elements = elements;
            in_bytes       = bytes;
            ++inputs;
        }
        else if (mode == nvinfer1::TensorIOMode::kOUTPUT)
        {
            output          = name;
            output_elements = elements;
            out_bytes       = bytes;
            ++outputs;
        }
        else
        {
            throw std::runtime_error("输入/输出格式不合法！");
        }
    }

    if (inputs != 1 || outputs != 1)
        throw std::runtime_error("本例只支持一个输入和一个输出张量");

    std::cout << "输入：" << " " << in_bytes << "字节\n";
    std::cout << "输出：" << " " << out_bytes << "字节\n";
}
```

设计要点：

- 改为**成员函数**直接填 `*this`，不再返回 `IOInfo`：调用点变成 `info_.InspectEngine(*engine_)`，
  少一次结构体拷贝，也不用再区分"局部 info"和"成员 info_"。
- 类型检查用 `getTensorDataType` / `getTensorFormat` / `getTensorLocation` 三个枚举；
  `getTensorFormatDesc()` 只用于打印。
- 限制 `kLINEAR` 是因为程序假设"字节数 = 逻辑元素数 × sizeof(float)"，向量化或带填充的布局不满足这个关系。
- `isShapeInferenceIO()` 为真的张量装的是形状而不是数据，字节数不能用元素数推算，直接拒绝。
- 多 profile 的 engine 意味着同一张量在不同 profile 下形状不同，静态缓冲区方案不成立，提前拒绝。
- 打印在检查之前：报错时调用方已经能从 stdout 看到这条 I/O 的实际情况，便于定位。
### 4.6 shapeString

```cpp
static std::string shapeString(const nvinfer1::Dims& shape)
{
    std::string text = "[";
    for (int i = 0; i < shape.nbDims; ++i)
    {
        if (i) text += ",";
        text += std::to_string(shape.d[i]);
    }
    return text + "]";
}
```

`shape.d[i]` 是 `int64_t`，用 `std::to_string` 转换；不要转成 `char`。

---

## 5. 成员与生命周期

```cpp
private:
    InferConfig                            config_;       // 先构造
    std::vector<char>                      engine_data_;  // 反序列化用的原始字节
    Logger                                 logger_;       // runtime/engine 都引用它
    std::unique_ptr<nvinfer1::IRuntime>    runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    IOInfo                                 info_;
    std::unique_ptr<WorkerPool>            workers_;      // 最后构造 → 最先析构
```

规则：

- C++ 成员按**声明顺序构造、逆序析构**。`workers_` 最后声明，所以它最先析构：worker 线程 join、
  通道与 context 销毁，之后 `engine_` 才释放。**顺序颠倒会让 context 在 engine 之后析构，属于未定义行为。**
- 生命周期长短：`logger` > `runtime` > `engine` > `context`。`logger_` 声明在 `runtime_` 之前，
  正好保证它比引用它的 TensorRT 对象活得久。
- `engine_data_` 只在反序列化时需要，但保留成成员最省心：`deserializeCudaEngine` 不保证把数据拷走，
  立刻释放有风险。
- 销毁 context 或设备缓冲区之前，必须确保使用它们的 GPU 工作已经结束：
  - `Channel::release()` 先 `cudaStreamSynchronize(stream)`；
  - `WorkerPool::shutdown()` 先排空队列再 `join`。
- `WorkerPool` 用 `unique_ptr` 且只前向声明：`~Inference()` 要析构不完整类型，因此必须在 `.cpp` 里定义
  （`= default` 即可，但必须放在定义 `WorkerPool` 之后）。

---

## 6. Channel 定义

```cpp
struct Channel
{
    std::unique_ptr<nvinfer1::IExecutionContext> context;
    cudaStream_t stream{};
    cudaEvent_t  done{};
    void* h_in  = nullptr;   // pinned host
    void* h_out = nullptr;   // pinned host
    void* d_in  = nullptr;   // device
    void* d_out = nullptr;   // device
};
```

构造（`Channel(engine, io)`）依次执行：

```cpp
context.reset(engine.createExecutionContext());
if (!context) throw std::runtime_error("创建 execution context 失败");

CHECK_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
CHECK_CUDA(cudaEventCreateWithFlags(&done, cudaEventDisableTiming));

CHECK_CUDA(cudaMallocHost(&h_in,  io.in_bytes));    // 释放用 cudaFreeHost
CHECK_CUDA(cudaMallocHost(&h_out, io.out_bytes));   // 释放用 cudaFreeHost
CHECK_CUDA(cudaMalloc(&d_in,  io.in_bytes));        // 释放用 cudaFree
CHECK_CUDA(cudaMalloc(&d_out, io.out_bytes));       // 释放用 cudaFree

if (!context->setTensorAddress(io.input.c_str(), d_in))
    throw std::runtime_error("绑定输入地址失败");
if (!context->setTensorAddress(io.output.c_str(), d_out))
    throw std::runtime_error("绑定输出地址失败");
```

要点：

- `cudaMallocHost` 分配页锁定（pinned）主机内存，是可靠实现异步主机传输的重要条件，但不要无限制分配。
- 参数单位是**字节数**，不是元素个数。
- `cudaStreamNonBlocking` 避免与 legacy default stream 的隐式同步。
- `cudaEventDisableTiming` 表示不用于计时；以后要测 GPU 时间再单独建启用计时的事件。
- `setTensorAddress()` 只告诉 TensorRT 从哪读、往哪写，**不做任何拷贝**。
- 静态 shape + 固定地址 → 初始化时绑定一次，之后重复使用。
- **构造中途失败必须自己清理**：C++ 不会为"未构造完成"的对象调用析构函数，所以上面的语句都包在
  `try { ... } catch (...) { release(); throw; }` 里，`release()` 是幂等的（分配置空、句柄置零），
  构造路径和析构路径共用同一份清理逻辑。

析构顺序（`Channel::release()`，幂等，构造中途失败也走同一条路径）：
`cudaStreamSynchronize(stream)` → `cudaFreeHost` 释放 pinned → `cudaFree` 释放 device → `cudaEventDestroy` → `cudaStreamDestroy` → `context.reset()`。

> 说明：这个结构取代原来只有 stream/event/缓冲区、没有 context 的 `CudaResource`。

---

## 7. 构建与初始化

```cpp
Inference::Inference(const InferConfig& config)
    : config_(validateConfig(config)), engine_data_(readEngine(config_.engine_path))
{
    int device_count = 0;
    CHECK_CUDA(cudaGetDeviceCount(&device_count));
    if (config_.deviceId >= device_count)
        throw std::runtime_error("deviceId 超出可见 GPU 数量");

    // 1. 选定设备：CUDA 当前设备按线程记录，构造线程必须与之后创建 context 的线程一致
    CHECK_CUDA(cudaSetDevice(config_.deviceId));

    // 2. runtime → engine
    runtime_.reset(nvinfer1::createInferRuntime(logger_));
    if (!runtime_) throw std::runtime_error("runtime 创建失败");
    engine_.reset(runtime_->deserializeCudaEngine(engine_data_.data(), engine_data_.size()));
    if (!engine_) throw std::runtime_error("engine 创建失败");

    // 3. 检查并记录 I/O（静态 shape、FP32、LINEAR、DEVICE、单输入单输出）
    info_.InspectEngine(*engine_);

    // 4. 起执行器：N 个 worker，每个构造并独占一条通道
    const size_t capacity = config_.queueCapacity > 0
        ? config_.queueCapacity
        : static_cast<size_t>(config_.concurrency) * 2;
    workers_ = std::make_unique<WorkerPool>(
        *engine_, info_, config_.concurrency, capacity, config_.deviceId);
}
```

- 执行流程：校验配置 → 读文件 → 选定设备 → runtime → 反序列化 → 检查 I/O → 起 N 个 worker。
- **配置校验放在初始化列表里先做**（`validateConfig`）：否则会先花时间读完 8 MB 文件，
  再报"concurrency 必须 >= 1"。
- 内存占用分两部分，都**不随请求总数增长**：
  - 常驻 = `concurrency × (context + 2×device + 2×pinned)`；
  - 波动 = `queueCapacity × 单请求输入大小`（排队中的任务）。
- CUDA 当前设备**按线程**记录：每个 worker 在 `loop()` 开头各自 `cudaSetDevice` 一次，之后不必重复调用。
- 异常安全：`WorkerPool` 构造中途失败时，已建通道由 `unique_ptr` 清理、已起线程由 `shutdown()` 回收。

---

## 8. 执行器：固定 worker + 有界队列

> 早期版本是"通道池 + 每请求一个 `std::async` 线程"。§13 的推理和 §11 的实测都表明池是多余的：
> 固定 worker 一对一绑定通道后，"名额"和"借还"都不再有意义。
> **最终实现采用固定 worker**，被删除的池方案保留在**附录 A** 供对照。

### 8.1 结构

```text
提交线程 ──submit()──▶ 有界队列 deque<Task> ──▶ worker 0 ──▶ channels_[0]（独占）
                                                worker 1 ──▶ channels_[1]（独占）
                                                ...
```

- **生产者**：任意调用线程调用 `submit()` 把任务放进队列；队列满则**阻塞生产者** —— 这就是背压，
  同时也是在途请求数上限。
- **消费者**：`concurrency` 个常驻 worker，启动时各自绑定一条通道，运行期**永不归还、永不争抢**。

由此得到的性质：

| 性质 | 原因 |
| --- | --- |
| 没有 `acquire/release`、没有池锁竞争 | 通道与 worker 一一绑定 |
| 线程数恒定 = `concurrency` | 不随请求数增长（实测提交 64 个任务线程数不变） |
| pinned 缓冲 / context / stream 全程复用 | 每个 worker 一套，生命周期等于推理器 |
| 每个 worker 只 `cudaSetDevice` 一次 | CUDA 当前设备按线程记录 |

### 8.2 任务与队列

```cpp
struct Task
{
    Input  owned;            // 异步任务：移交所有权（调用方 move 进来，0 拷贝）
    const Input* borrowed{}; // 同步任务：借用调用方内存，调用方阻塞到完成
    std::promise<Output>     promise;

    const Input& input() const { return borrowed ? *borrowed : owned; }
};
```

- `promise` 是产出端，`future` 交给调用方消费。**结果和异常都走这一条路**：
  `set_value()` 交付结果，`set_exception()` 交付异常。
- `owned` / `borrowed` 二选一，让同步路径不必为 3.6 MB 输入多拷一次；
  安全性由"`infer()` 不返回就说明任务没结束"保证。

### 8.3 入队（生产者）

```cpp
std::future<Output> enqueue(std::unique_ptr<Task> task)
{
    auto future = task->promise.get_future();   // 先取 future，任务被取走后仍能拿到
    {
        std::unique_lock<std::mutex> lk(mutex_);
        not_full_.wait(lk, [this] { return closed_ || queue_.size() < capacity_; });
        if (closed_) throw std::runtime_error("推理器正在销毁，拒绝新任务");
        queue_.push_back(std::move(task));
    }
    not_empty_.notify_one();   // 只新增 1 个任务，叫醒 1 个 worker 即可
    return future;
}
```

- 尺寸校验在**入队之前**完成：错误同步抛给调用方，不会被包装成 future 里的异步异常。
- `notify_one()` 放在解锁之后，避免被唤醒的线程立刻又抢锁。
- 队列元素是 `unique_ptr<Task>`：`Task` 含 `std::promise`（不可拷贝），用 `unique_ptr` 语义最清楚。

### 8.4 worker 循环（消费者）

```cpp
void loop(size_t index)
{
    // 每个 worker 只绑定一次设备；之后所有 CUDA 调用都在正确的卡上
    const auto device_status = cudaSetDevice(device_id_);
    Channel* channel = (device_status == cudaSuccess) ? channels_[index].get() : nullptr;

    for (;;)
    {
        std::unique_ptr<Task> task;
        {
            std::unique_lock<std::mutex> lk(mutex_);
            not_empty_.wait(lk, [this] { return closed_ || !queue_.empty(); });
            if (queue_.empty()) return;        // 已关闭且队列排空
            task = std::move(queue_.front());
            queue_.pop_front();
        }
        not_full_.notify_one();

        // 任何异常都通过 promise 交给调用方，绝不让 future 悬挂
        if (!channel)
        {
            task->promise.set_exception(std::make_exception_ptr(
                std::runtime_error("worker 无法设置 CUDA 设备")));
            continue;
        }

        try         { task->promise.set_value(runOnChannel(*channel, info_, task->input())); }
        catch (...) { task->promise.set_exception(std::current_exception()); }
    }
}
```

**关键不变量：worker 绝不跳过队列里的任务。** 哪怕设备绑定失败，也要 `set_exception` 把任务"结掉"，
否则调用方会永远卡在 `future.get()`。这也是把异常塞进 future、而不是抛到线程外的根本原因
—— 线程函数里逃逸的异常会 `std::terminate` 整个进程。

### 8.5 关闭

```cpp
void shutdown()
{
    { std::lock_guard<std::mutex> lk(mutex_); closed_ = true; }
    not_empty_.notify_all();
    not_full_.notify_all();
    for (auto& thread : threads_) if (thread.joinable()) thread.join();
}
```

- 先置 `closed_` 再 `notify_all`：否则阻塞在 `wait` 里的线程可能永远醒不来。
- worker 的退出条件是"`closed_` 且队列为空"，所以**队列里已有的任务仍会被执行完**，调用方的 future 都有结果。
- 阻塞在 `enqueue` 的生产者被唤醒后看到 `closed_` → 抛异常，不会挂死。
- 析构顺序：`workers_` 最后声明 → 最先析构 → worker 先退出、通道与 context 先销毁，`engine_` 才释放。

### 8.6 与池方案对比

| | 池 + 每请求线程（旧） | 固定 worker + 队列（新） |
| --- | --- | --- |
| 线程数 | 每请求一个，随请求数增长 | 恒为 `concurrency` |
| 通道归属 | 动态借还，需锁 + 条件变量 + RAII 租约 | 启动时绑定，运行期无锁 |
| 状态复用 | 每请求重建线程本地状态 | pinned 缓冲 / context / stream 全程复用 |
| 在途上限 | 需要额外的名额计数器 | 队列长度天然就是上限 |
| 关闭 | 唤醒等待者 + 等名额归零 | 置标志 + 排空队列 + join |
| 代价 | 线程创建 + 借还开销 | 有界队列带来的排队延迟 |

实测差额见 §11.3（c≤4 约 +7%~16%，c=8 归零，因为那时瓶颈已经在 GPU）。

---

## 9. 单请求执行

`runOnChannel()` 是匿名 namespace 里的自由函数，被所有 worker 共用（不再需要 `this`）：

```cpp
Output runOnChannel(Channel& ch, const IOInfo& info, const Input& input)
{
    if (input.size() != info.input_elements)
        throw std::runtime_error("输入尺寸不对！");

    // 1. 普通内存拷贝：vector 内存 → pinned 内存
    std::memcpy(ch.h_in, input.data(), info.in_bytes);

    try
    {
        // 2. H2D
        CHECK_CUDA(cudaMemcpyAsync(ch.d_in, ch.h_in, info.in_bytes,
                                   cudaMemcpyHostToDevice, ch.stream));

        // 3. 推理（enqueueV3 成功返回 true）
        if (!ch.context->enqueueV3(ch.stream))
            throw std::runtime_error("无法将输入推送到队列！");

        // 4. D2H
        CHECK_CUDA(cudaMemcpyAsync(ch.h_out, ch.d_out, info.out_bytes,
                                   cudaMemcpyDeviceToHost, ch.stream));

        // 5. 等"本请求"完成：event 比整条 stream 的语义更精确
        CHECK_CUDA(cudaEventRecord(ch.done, ch.stream));
        CHECK_CUDA(cudaEventSynchronize(ch.done));
    }
    catch (...)
    {
        // 失败后把流排空：通道继续复用时不留未完成的异步操作
        cudaStreamSynchronize(ch.stream);
        throw;
    }

    // 6. 拷进调用方拥有的结果对象
    Output result(info.output_elements);
    std::memcpy(result.data(), ch.h_out, info.out_bytes);
    return result;
}
```

设计要点：

- 同一 stream 内 `H2D → 推理 → D2H` 天然有序，中间不需要 CPU 同步。
- 等待用 `cudaEventSynchronize(ch.done)`，**不要用 `cudaDeviceSynchronize()`**：后者会等住其他 worker 的通道，
  直接破坏并发。
- 即便只是"等自己这条流"，用 event 也比 `cudaStreamSynchronize` 更好：将来要做"提交后由别的线程等"，
  event 可以被别的线程等待/查询（`cudaEventQuery`），不必把 `cudaStream_t` 传出去。
- 必须返回**新分配**的 `Output`，不能把 `h_out` 裸指针交出去 —— 下一次推理会覆盖它。
- `catch (...)` 里必须排空 stream：否则一条失败请求会把后续请求的工作一起搅乱。
- 三个接口共用这一个函数，差别只在"谁调用它、何时等待"。

---

## 10. 三个接口的实现

### 10.1 infer（同步）

```cpp
Output Inference::infer(const Input& input)
{
    // 提交任务并立刻等待；borrowed 表示借用 input 的内存，不拷贝
    return workers_->submitBorrowed(input).get();
}
```

"同步"的含义只是**函数返回时结果已经可用**，内部照样用 `cudaMemcpyAsync` 提交、最后统一等待。

两个要注意的行为：

- 同步调用现在也要**排队**：若队列里已有异步任务，它会排在后面（最长约 `queueCapacity` 个请求的等待）。
  早期"直接抢空闲通道"的实现在这一点上行为不同。
- 借用输入的生命周期由"函数不返回就说明任务没结束"保证：`get()` 返回之后 `input` 才可能离开作用域。

### 10.2 inferAsync（异步）

第一版用 `std::async` 学语义（**已淘汰**，对比见 §8.6 / §11.3）：

```cpp
return std::async(std::launch::async, [state = shared_state_, input = std::move(input)] {
    CHECK_CUDA(cudaSetDevice(state->deviceId));
    auto lease = state->pool.acquire();       // 借通道
    return state->runOnChannel(lease.get(), input);
});
```

它能跑，但有三个结构性问题：每请求新建线程、每请求借还通道、在途数还要靠额外的名额计数器限制。
`std::async` 自身也有坑：

- `std::launch::async` 必须显式写，否则实现可以选择延迟到 `get()` / `wait()` 才执行。
- 返回的 future 若是共享状态的最后一个持有者，其析构可能**等任务跑完**；写完就丢弃会表现成阻塞调用。
- 不要捕获 `this`：调用方先销毁推理对象、后台任务再访问 `this` 就是悬空访问。

最终实现改用固定 worker，`inferAsync` 只剩提交：

```cpp
std::future<Output> Inference::inferAsync(Input input)
{
    // 调用方 move 传入时零拷贝；传左值则只拷贝一次
    return workers_->submit(std::move(input));
}
```

- 尺寸校验在 `submit()` 里、**入队之前**完成 → 错误同步抛出，不会变成 future 里的异步异常。
- 结果与异常都经 `std::promise` 交付：worker 里 `set_value` / `set_exception`，**绝不让 future 悬挂**。

### 10.3 inferParallel（多路并行）

```cpp
std::vector<std::future<Output>> Inference::inferParallel(std::vector<Input> inputs)
{
    // 先整体校验：避免中途抛异常后留下半批任务
    for (const auto& input : inputs)
        if (input.size() != info_.input_elements)
            throw std::runtime_error("输入尺寸不对！");

    std::vector<std::future<Output>> futures;
    futures.reserve(inputs.size());

    try
    {
        for (auto& input : inputs)
            futures.push_back(workers_->submit(std::move(input)));   // 逐个 move，0 拷贝
    }
    catch (...)
    {
        for (auto& future : futures) future.wait();   // 等已提交的任务结束再抛出
        throw;
    }

    return futures;
}
```

要点：

- 是否真并行取决于 `concurrency`：10 个输入、`concurrency = 2` 时两个 worker 轮流取任务，其余在队列里等。
- **不要传左值**：`inferParallel(batch)` 会整批拷贝一次（128 × 3.6 MB ≈ 460 MB，实测多花 154 ms）。
  用 `std::move(batch)` 或直接传临时对象。
- 不要在循环里提交完立刻 `get()`，那等于串行。
- 按输入顺序 `get()` 不会强制 GPU 串行，只是调用者可能先等较慢的前一个结果。
- 中途失败时要先 `wait()` 掉已提交的 future 再抛：future 析构本身会阻塞，与其在未知位置阻塞，不如显式等完。

---

## 11. 验证与测量

测量环境：RTX 3060 Ti、`model/model.engine`（YOLOv8n FP16，输入 480×640×3，输出 21×6300）。

### 11.1 正确性与鲁棒性（`test_three_mode_infer.cpp`）

| 用例 | 目的 |
| --- | --- |
| async / parallel 与 sync 逐请求对拍 | 三条路径结果一致（16 组不同输入） |
| 32 请求 / 4 通道 | 通道之间不串数据 |
| 8 线程并发调用同步 `infer()` | 借用输入不会互相踩内存 |
| 输入尺寸不匹配 | 快速失败：不起线程、不进队列 |
| future 未 `get()` 就销毁 `Inference` | 析构能排空队列、不崩溃 |
| `concurrency = 0` / `deviceId` 越界 / engine 路径不存在 | 配置与文件错误被明确拒绝 |

```text
全部通过 (failures=0)                                  # 连续 3 次稳定
compute-sanitizer --leak-check full → 0 bytes leaked, 0 errors
```

线程数实测（`concurrency = 4`）：

```text
启动前 1 → 构造后 8（含 TensorRT 内部线程） → 提交 64 个任务时 8 → 全部完成 8
```

### 11.2 单请求耗时分解

```text
H2D = 0.16 ms    compute = 0.85 ms    D2H = 0.03 ms    total = 1.04 ms
PCIe 实测带宽：H2D 23.6 GB/s，D2H 20.6 GB/s
```

**82% 的时间在 GPU 计算**，拷贝只占 18%。这条结论决定了两件事：

- 想提吞吐，优化方向在 GPU 侧（CUDA Graph、batch > 1），不在调度层；
- 调度层怎么改，单通道延迟都不会低于约 1.0 ms。

### 11.3 调度方案 A/B

同一进程、预热后取 5 次中位数、两边都零拷贝移交，n = 128，单位 ms/请求：

| 并发 | 旧：`std::async` + 池 | 新：固定 worker + 队列 | 提升 |
| --- | --- | --- | --- |
| 1 | 1.469 | 1.368 | +7% |
| 2 | 0.945 | 0.816 | +16% |
| 4 | 0.709 | 0.653 | +9% |
| 8 | 0.645 | 0.652 | 0% |

流水线延迟（窗口 8 个在途，300 请求）：

| 并发 | 旧 p50 / p99 / max | 新 p50 / p99 / max |
| --- | --- | --- |
| 4 | 6.21 / 7.94 / 8.72 ms | 5.80 / 7.29 / 7.86 ms |
| 8 | 6.33 / 7.86 / 8.64 ms | 6.25 / 7.85 / 9.70 ms |

结论：

- **调度层收益是 +7%~16%（c≤4），c=8 时归零** —— 此时 GPU 已饱和。
- 旧实现"看起来慢很多"主要来自**入口拷贝不与推理重叠**：传左值让 `inferParallel` 一次性拷 460 MB，
  那段拷贝期间没有任何任务在跑。改成 `std::move` 后差距消失，与调度方式无关。
  这也说明：**基准一定要先 warm-up、取中位数、并让两边走同样的拷贝路径**，否则很容易得出"优化了 35%"的错觉。
- 固定 worker 的真正价值在**结构**：线程数恒定、无借还竞争、状态全复用、关闭语义清晰。

### 11.4 并发加速比

`concurrency` 从 1 加到 8，吞吐只到约 2.2×（711 → 1597 req/s），瓶颈在 GPU 并发效率而非 CPU 调度。
要突破这个数字只能改 GPU 侧（batch > 1 的 engine、CUDA Graph），不是改线程模型。

---

## 12. 建议实施顺序

1. 定义 `InferConfig` / `Input` / `Output` / `IOInfo`
2. 辅助函数：`cudaCheck` + `CHECK_CUDA`、`readEngine`、`calculateElements`、`calculateBytes`、`InspectEngine`、`shapeString`
3. 定义 `Channel`（构造 + 幂等的 `release()`），先用 **1 条**通道
4. 构造函数：校验配置 → 读文件 → runtime → engine → inspect
5. 实现 `runOnChannel()`，直接调用它把单请求结果跑对
6. 扩成 N 条通道，**先做通道池版本**：理解"名额 + 等待"模型（附录 A）
7. 加 `inferAsync()`（`std::async`）与 `inferParallel()`，跑通三种方式
8. 测量：单请求分解、并发加速比、延迟分布（§11）
9. **改成固定 worker + 有界队列，删掉池**：此时会发现 `acquire/release`、名额计数器全都不需要了
10. 补测试：结果对拍、通道串扰、并发同步调用、销毁时排空队列、非法配置

不要第一步就同时引入线程池、队列、锁和 CUDA。
**先把单请求跑对，再考虑并发，最后才拆掉多余的抽象。**

---

## 13. 关于"是否需要池"的最终结论

推理过程：若采用**固定 worker**，启动 `concurrency` 个线程，**每个线程永久独占一条通道**：

```text
worker i  ↦  channels_[i]      （启动时绑定，运行期不再变化）
```

此时不存在争抢：`acquire/release`、通道池的互斥锁、条件变量、在途名额计数器**全部可以删除**。
池只在"用临时线程处理一次请求"（如 `std::async` 按需起线程）时才必要。

**最终实现就是这条路径**（§8）：

| 早期组件 | 最终归属 |
| --- | --- |
| `ChannelPool` / `ChannelLease` | 删除，通道由 `WorkerPool` 直接持有 |
| `InflightTracker`（在途名额） | 删除，有界队列长度天然就是上限 |
| `SharedState`（供异步任务捕获） | 不需要：任务只是队列里的一个 `Task`，由 worker 直接取用，不再捕获 `this` |
| `Channel` | 保留，成为 worker 的私有资源 |

代价也要写清楚：

- 同步 `infer()` 现在走队列，会排在已有异步任务后面；
- 队列满时 `submit()` 阻塞调用方（旧方案靠名额计数器，行为等价，只是现在有显式旋钮）；
- 队列中的任务会持有输入内存：`queueCapacity × 单请求大小`。

---

## 14. 前提与限制

### 14.1 模型 / engine 限制

- 只支持：**1 个输入 + 1 个输出**、静态 shape、FP32、LINEAR、DEVICE、非形状推导 I/O。
- 只支持单 optimization profile（`getNbOptimizationProfiles() > 1` 直接拒绝）。
- 遇到动态维度（`dims.d[i] < 0`）明确报"不支持动态 shape"，而不是含糊的"未解析维度"。
  真要支持动态 shape，需要 `setInputShape()` + `inferShapes()`，且每通道的输出大小可能不同，
  `Channel` 的缓冲区就不能在构造时一次分配。

### 14.2 每个 context 的显存

- `createExecutionContext()`：TensorRT 内部为每个 context 分配 workspace，所以 N 个 context 本来就有 N 份显存。
- `createExecutionContextWithoutDeviceMemory()`：不分配，由 `setDeviceMemory(void*)` 提供，
  大小取自 context 的 `getDeviceMemorySize()`（具体名称以所用 TensorRT 版本的 `nvinfer1` 头文件为准）。
- 这份内存同样**不能并发共享**：两条通道指向同一块 workspace，推理结果会互相破坏。

### 14.3 行为约定（最终实现）

| 约定 | 说明 |
| --- | --- |
| 线程数 | 恒为 `concurrency`（+ TensorRT 内部线程），与请求数无关 |
| 在途上限 | `concurrency + queueCapacity` |
| 队列满 | `infer` / `inferAsync` / `inferParallel` 都阻塞调用方 |
| 尺寸错误 | 入队前同步抛出，不进队列、不产生 future |
| 结果与异常 | 一律经 `promise` 交付；worker 不会让 future 悬挂 |
| 销毁 | 置 `closed_` → 唤醒 → 排空队列 → `join`；排队中的任务会被执行完 |
| 同步 `infer()` | 与异步共用同一执行器，可能排在已有任务之后 |
| 借用输入 | 仅 `infer()` 使用，由"调用方阻塞到完成"保证生命周期 |

### 14.4 不要做的事

- 不要在 worker 线程里再 `submit()`：队列满时会自我死锁（worker 等队列空，而队列要等 worker 干完）。
- 不要用 `cudaDeviceSynchronize()` 代替按通道 / 按请求的等待。
- 不要把输入左值传给 `inferParallel()`。
- 不要在 future 还没 `get()` / `wait()` 完时销毁 `Inference`：现实现会等，但那是兜底，不是用法。
- 不要给 `queueCapacity` 设一个很大的值充当"无界队列"：排队请求会按 `输入大小 × 长度` 占内存。

---

## 附录 A：早期通道池方案（已删除，保留供对照）

> 以下内容对应 `three_mode_infer.cpp` 的旧版本。最终实现（§8）用固定 worker 取代了它：
> `ChannelPool`、`ChannelLease`、在途名额计数器均已从代码中移除。

### A.1 通道池

作用：N 条通道被多个线程合用，保证同一时刻一条通道只被一个请求占用。

```cpp
class ChannelPool
{
public:
    explicit ChannelPool(std::vector<std::unique_ptr<Channel>> channels)
        : channels_(std::move(channels))
    {
        for (auto& ch : channels_)
            free_.push_back(ch.get());
    }

    ChannelPool(const ChannelPool&) = delete;
    ChannelPool& operator=(const ChannelPool&) = delete;

    ~ChannelPool()
    {
        {
            std::lock_guard<std::mutex> lk(mutex_);
            closing_ = true;
        }
        cv_.notify_all();   // 唤醒全部等待者，避免永久挂起
    }

    ChannelLease acquire()
    {
        std::unique_lock<std::mutex> lk(mutex_);
        cv_.wait(lk, [this] { return closing_ || !free_.empty(); });

        if (closing_)
            throw std::runtime_error("通道池已关闭");

        Channel* ch = free_.back();
        free_.pop_back();
        return ChannelLease(*this, *ch);
    }

    void release(Channel& ch)
    {
        {
            std::lock_guard<std::mutex> lk(mutex_);
            free_.push_back(&ch);
        }
        cv_.notify_one();   // 只腾出 1 个名额，叫醒 1 个即可
    }

private:
    std::vector<std::unique_ptr<Channel>> channels_;  // 所有者
    std::vector<Channel*>                 free_;      // 非 owning 观察者
    std::mutex                            mutex_;
    std::condition_variable               cv_;
    bool                                  closing_ = false;
};
```

不变量：**`free_.size() + 在途请求数 == 通道总数`**。

设计要点（这些坑在池方案里必须逐个处理，换成固定 worker 后整体消失）：

- `free_` 存裸指针是正当用法：所有权在 `channels_`，池只记录"哪个还空着"。
- `channels_` 用 `unique_ptr` 是因为 `acquire()` 交出的 `Channel&` 必须在整个请求期间有效；
  `vector<Channel>` 一旦扩容，元素地址全部失效。
- `cv_.wait(lk, pred)` 等价于 `while (!pred()) cv_.wait(lk);`，谓词版能处理**虚假唤醒**。
- `wait` 必须配 `unique_lock`（睡眠时释放锁、醒来重新加锁）；`release` 只需一次性加解锁，用 `lock_guard`。
- `notify_one()` 放在解锁之后，避免被唤醒的线程立刻又抢锁。
- 析构时 `closing_ = true` + `notify_all()`，否则阻塞在 `acquire` 的线程永久挂住。
- **`release` 的前置条件**：`runOnChannel()` 已等待本请求完成，且结果已拷进独立的 `Output`。
  否则下一个请求会覆盖还没取走的输出。
- 持有通道时不要再 `acquire()`，会自我死锁。

### A.2 RAII 租约

```cpp
class ChannelLease
{
public:
    ChannelLease(ChannelPool& pool, Channel& ch) : pool_(&pool), channel_(&ch) {}

    ChannelLease(ChannelLease&& other) noexcept
        : pool_(other.pool_), channel_(other.channel_)
    {
        other.pool_    = nullptr;
        other.channel_ = nullptr;
    }

    ChannelLease(const ChannelLease&) = delete;
    ChannelLease& operator=(const ChannelLease&) = delete;

    ~ChannelLease() { if (pool_) pool_->release(*channel_); }

    Channel& get() const { return *channel_; }

private:
    ChannelPool* pool_;
    Channel*     channel_;
};
```

- 析构自动 `release`，正常返回和抛异常都能归还，调用方不必写 `try/catch`。
- 移动构造把源对象置空，保证只归还一次；否则 `free_` 里出现重复指针，
  两次 `acquire` 会拿到同一条通道，产生数据竞争。
- 头文件里 `ChannelLease` 必须定义在 `ChannelPool` **之后**（或在前面写 `class ChannelPool;` 前向声明
  并把析构函数移到 `.cpp`），否则 `pool_->release()` 面对的是不完整类型。
