#include "three_mode_infer.hpp"

#include <algorithm>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <deque>
#include <fstream>
#include <iostream>
#include <limits>
#include <thread>

namespace TRTInfer
{

namespace
{

void cudaCheck(cudaError_t status, const char* call)
{
    if (status != cudaSuccess)
    {
        throw std::runtime_error(std::string(call) + ": " + cudaGetErrorString(status));
    }
}

#define CHECK_CUDA(call) cudaCheck((call), #call)

std::string shapeString(const nvinfer1::Dims& dims)
{
    std::string text = "[";
    for (int i = 0; i < dims.nbDims; ++i)
    {
        if (i) text += ",";
        text += std::to_string(dims.d[i]);
    }
    return text + "]";
}

const InferConfig& validateConfig(const InferConfig& config)
{
    if (config.engine_path.empty())
        throw std::runtime_error("engine_path 不能为空");
    if (config.concurrency < 1)
        throw std::runtime_error("concurrency 必须 >= 1");
    if (config.deviceId < 0)
        throw std::runtime_error("deviceId 必须 >= 0");
    if (config.queueCapacity > (1ULL << 20))
        throw std::runtime_error("queueCapacity 过大");
    return config;
}

// 一条通道上完成一次推理：H2D → enqueueV3 → D2H → 等待本请求的 event
Output runOnChannel(Channel& ch, const IOInfo& info, const Input& input)
{
    if (input.size() != info.input_elements)
        throw std::runtime_error("输入尺寸不对！");

    std::memcpy(ch.h_in, input.data(), info.in_bytes);

    try
    {
        CHECK_CUDA(cudaMemcpyAsync(
            ch.d_in, ch.h_in, info.in_bytes, cudaMemcpyHostToDevice, ch.stream));

        // enqueueV3 成功返回 true
        if (!ch.context->enqueueV3(ch.stream))
            throw std::runtime_error("无法将输入推送到队列！");

        CHECK_CUDA(cudaMemcpyAsync(
            ch.h_out, ch.d_out, info.out_bytes, cudaMemcpyDeviceToHost, ch.stream));

        // 用 event 等待"本请求"完成，语义比等整条流更精确
        CHECK_CUDA(cudaEventRecord(ch.done, ch.stream));
        CHECK_CUDA(cudaEventSynchronize(ch.done));
    }
    catch (...)
    {
        // 失败后把流排空，通道继续复用时不带未完成的异步操作
        cudaStreamSynchronize(ch.stream);
        throw;
    }

    Output result(info.output_elements);
    std::memcpy(result.data(), ch.h_out, info.out_bytes);
    return result;
}

}   // namespace

// ------------------------------------------------------------------ IOInfo --

void IOInfo::InspectEngine(const nvinfer1::ICudaEngine& engine)
{
    if (engine.getNbOptimizationProfiles() > 1)
        throw std::runtime_error("本例只支持单 optimization profile 的静态 engine");

    int inputs = 0;
    int outputs = 0;

    for (int i = 0; i < engine.getNbIOTensors(); ++i)
    {
        const auto name = engine.getIOTensorName(i);
        const auto mode = engine.getTensorIOMode(name);
        const auto shape = engine.getTensorShape(name);
        const auto dtype = engine.getTensorDataType(name);
        const auto format = engine.getTensorFormat(name);
        const auto where = engine.getTensorLocation(name);

        std::cout << name << " "
                  << "shape= " << shapeString(shape)
                  << " mode= " << (mode == nvinfer1::TensorIOMode::kINPUT ? "INPUT" : "OUTPUT")
                  << " dtype= " << static_cast<int>(dtype)
                  << " format= " << engine.getTensorFormatDesc(name)
                  << " location= " << static_cast<int>(where) << std::endl;

        if (dtype != nvinfer1::DataType::kFLOAT ||
            format != nvinfer1::TensorFormat::kLINEAR ||
            where != nvinfer1::TensorLocation::kDEVICE ||
            engine.isShapeInferenceIO(name))
        {
            throw std::runtime_error("本例只支持 FP32/LINEAR/DEVICE 且非形状推导 I/O: " + std::string(name));
        }

        // 静态 shape 才能在这里算出固定的元素数；动态维度由 calculateElements 明确报错
        const auto elements = calculateElements(shape);
        const auto bytes = calculateBytes(elements);

        if (mode == nvinfer1::TensorIOMode::kINPUT)
        {
            input = name;
            input_elements = elements;
            in_bytes = bytes;
            ++inputs;
        }
        else if (mode == nvinfer1::TensorIOMode::kOUTPUT)
        {
            output = name;
            output_elements = elements;
            out_bytes = bytes;
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

size_t IOInfo::calculateElements(const nvinfer1::Dims& dims) const
{
    if (dims.nbDims <= 0)
        throw std::runtime_error("无效张量维度: " + shapeString(dims));

    size_t count = 1;
    for (int i = 0; i < dims.nbDims; ++i)
    {
        const auto dim = dims.d[i];
        if (dim < 0)
        {
            throw std::runtime_error(
                "本实现不支持动态 shape（需要 setInputShape/inferShapes），请使用静态维度 engine: "
                + shapeString(dims));
        }
        if (dim == 0)
            throw std::runtime_error("本例不支持空张量: " + shapeString(dims));

        const auto size = static_cast<size_t>(dim);
        if (count > std::numeric_limits<size_t>::max() / size)
            throw std::runtime_error("Elements: 元素数溢出！");
        count *= size;
    }
    return count;
}

size_t IOInfo::calculateBytes(const size_t elements) const
{
    if (elements > std::numeric_limits<size_t>::max() / sizeof(float))
        throw std::runtime_error("Bytes: 元素数溢出！");

    return elements * sizeof(float);
}

// ----------------------------------------------------------------- Channel --

Channel::Channel(nvinfer1::ICudaEngine& engine, const IOInfo& info)
{
    // 调用者必须保证当前线程的 CUDA 设备与 engine 一致（Inference 构造时已 cudaSetDevice）
    try
    {
        context.reset(engine.createExecutionContext());
        if (!context)
            throw std::runtime_error("context 创建失败！");

        // cudaStreamNonBlocking：不参与 legacy default stream 的隐式同步
        CHECK_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        CHECK_CUDA(cudaEventCreateWithFlags(&done, cudaEventDisableTiming));

        // 固定页（pinned）内存：cudaMemcpyAsync 才能真正异步
        CHECK_CUDA(cudaMallocHost(&h_in, info.in_bytes));
        CHECK_CUDA(cudaMallocHost(&h_out, info.out_bytes));
        CHECK_CUDA(cudaMalloc(&d_in, info.in_bytes));
        CHECK_CUDA(cudaMalloc(&d_out, info.out_bytes));

        // setTensorAddress 成功返回 true：每条通道有独立的 device 缓冲，因此并发互不覆盖
        if (!context->setTensorAddress(info.input.c_str(), d_in))
            throw std::runtime_error("绑定输入地址失败");

        if (!context->setTensorAddress(info.output.c_str(), d_out))
            throw std::runtime_error("绑定输出地址失败");
    }
    catch (...)
    {
        // 构造未完成时析构函数不会被调用，必须在这里清理已申请的资源
        release();
        throw;
    }
}

Channel::~Channel()
{
    release();
}

void Channel::release() noexcept
{
    // 先等流上的工作结束，再释放其引用的内存
    if (stream) cudaStreamSynchronize(stream);

    // cudaMallocHost 申请的内存必须用 cudaFreeHost 释放（不是 cudaFree）
    if (h_in) { cudaFreeHost(h_in); h_in = nullptr; }
    if (h_out) { cudaFreeHost(h_out); h_out = nullptr; }
    if (d_in) { cudaFree(d_in); d_in = nullptr; }
    if (d_out) { cudaFree(d_out); d_out = nullptr; }

    if (done) { cudaEventDestroy(done); done = nullptr; }
    if (stream) { cudaStreamDestroy(stream); stream = nullptr; }

    context.reset();
}

// ------------------------------------------------------------------ Logger --

void Logger::log(Severity severity, const char* msg) noexcept
{
    if (severity <= Severity::kWARNING)
    {
        // TensorRT 可能从内部线程回调，加锁避免多线程输出交错
        std::lock_guard<std::mutex> lk(mutex_);
        std::cerr << "[TRT]: " << msg << std::endl;
    }
}

// -------------------------------------------------------------- WorkerPool --
//
// 生产者-消费者执行器：
//   * 生产者：调用线程 submit()，把任务放进有界队列（满了就阻塞 → 背压）
//   * 消费者：concurrency 个常驻 worker，每个 worker 永久独占一条 Channel
// 由此：没有通道池争抢、没有 acquire/release、没有每请求新建线程，
//       pinned 缓冲/context/stream 全部按 worker 复用。
class WorkerPool
{
public:
    WorkerPool(
        nvinfer1::ICudaEngine& engine,
        const IOInfo& info,
        int workers,
        size_t capacity,
        int deviceId
    )
    : info_(info), capacity_(std::max<size_t>(capacity, 1)), device_id_(deviceId)
    {
        channels_.reserve(static_cast<size_t>(workers));
        for (int i = 0; i < workers; ++i)
        {
            channels_.push_back(std::make_unique<Channel>(engine, info_));
        }

        try
        {
            threads_.reserve(static_cast<size_t>(workers));
            for (int i = 0; i < workers; ++i)
            {
                threads_.emplace_back([this, i] { loop(i); });
            }
        }
        catch (...)
        {
            shutdown();   // 已启动的 worker 必须回收
            throw;
        }
    }

    ~WorkerPool()
    {
        shutdown();
    }

    WorkerPool(const WorkerPool&) = delete;
    WorkerPool& operator=(const WorkerPool&) = delete;

    // 异步：移交输入所有权（调用方 std::move 进来时为 0 拷贝）
    std::future<Output> submit(Input input)
    {
        checkInput(input);
        auto task = std::make_unique<Task>();
        task->owned = std::move(input);
        return enqueue(std::move(task));
    }

    // 同步：借用调用方内存，调用方必须阻塞到 future 完成（仅 infer() 使用）
    std::future<Output> submitBorrowed(const Input& input)
    {
        checkInput(input);
        auto task = std::make_unique<Task>();
        task->borrowed = &input;
        return enqueue(std::move(task));
    }

    size_t workerCount() const { return channels_.size(); }
    size_t capacity() const { return capacity_; }

private:
    struct Task
    {
        Input owned;              // 异步任务持有
        const Input* borrowed{};  // 同步任务借用（生命周期由调用方阻塞保证）
        std::promise<Output> promise;

        const Input& input() const { return borrowed ? *borrowed : owned; }
    };

    void checkInput(const Input& input) const
    {
        if (input.size() != info_.input_elements)
            throw std::runtime_error("输入尺寸不对！");
    }

    std::future<Output> enqueue(std::unique_ptr<Task> task)
    {
        auto future = task->promise.get_future();
        {
            std::unique_lock<std::mutex> lk(mutex_);
            // 队列满则阻塞生产者：这就是背压，也是在途请求数上限
            not_full_.wait(lk, [this] { return closed_ || queue_.size() < capacity_; });
            if (closed_)
                throw std::runtime_error("推理器正在销毁，拒绝新任务");
            queue_.push_back(std::move(task));
        }
        // 只新增 1 个任务，叫醒 1 个 worker 即可
        not_empty_.notify_one();
        return future;
    }

    void loop(size_t index)
    {
        // 每个 worker 一次：绑定设备后常驻，之后不再有 cudaSetDevice 的反复开销
        const auto device_status = cudaSetDevice(device_id_);
        Channel* channel = (device_status == cudaSuccess) ? channels_[index].get() : nullptr;

        for (;;)
        {
            std::unique_ptr<Task> task;
            {
                std::unique_lock<std::mutex> lk(mutex_);
                not_empty_.wait(lk, [this] { return closed_ || !queue_.empty(); });
                if (queue_.empty())
                    return;   // 已关闭且队列排空
                task = std::move(queue_.front());
                queue_.pop_front();
            }
            not_full_.notify_one();

            // 任何异常都通过 promise 传给调用方，绝不让 future 永远悬挂
            if (!channel)
            {
                task->promise.set_exception(std::make_exception_ptr(
                    std::runtime_error("worker 无法设置 CUDA 设备")));
                continue;
            }

            try
            {
                task->promise.set_value(runOnChannel(*channel, info_, task->input()));
            }
            catch (...)
            {
                task->promise.set_exception(std::current_exception());
            }
        }
    }

    // 关闭并排空：已在队列中的任务仍会被执行完，调用方的 future 都有结果
    void shutdown()
    {
        {
            std::lock_guard<std::mutex> lk(mutex_);
            closed_ = true;
        }
        not_empty_.notify_all();
        not_full_.notify_all();

        for (auto& thread : threads_)
        {
            if (thread.joinable()) thread.join();
        }
    }

    IOInfo info_;
    size_t capacity_;
    int device_id_;

    std::vector<std::unique_ptr<Channel>> channels_;   // 每条通道归一个 worker 独占
    std::vector<std::thread> threads_;
    std::deque<std::unique_ptr<Task>> queue_;
    std::mutex mutex_;
    std::condition_variable not_empty_;
    std::condition_variable not_full_;
    bool closed_ = false;
};

// --------------------------------------------------------------- Inference --

Inference::Inference(const InferConfig& config)
: config_(validateConfig(config)), engine_data_(readEngine(config_.engine_path))
{
    int device_count = 0;
    CHECK_CUDA(cudaGetDeviceCount(&device_count));
    if (config_.deviceId >= device_count)
        throw std::runtime_error("deviceId 超出可见 GPU 数量");

    // CUDA 当前设备是按线程记录的；构造线程必须与被创建的 context 一致
    CHECK_CUDA(cudaSetDevice(config_.deviceId));

    runtime_.reset(nvinfer1::createInferRuntime(logger_));
    if (!runtime_)
        throw std::runtime_error("runtime 创建失败！");

    engine_.reset(runtime_->deserializeCudaEngine(engine_data_.data(), engine_data_.size()));
    if (!engine_)
        throw std::runtime_error("engine 创建失败！");

    info_.InspectEngine(*engine_);

    // 默认队列长度 2 × concurrency：够流水起来，又能及时形成背压
    const size_t capacity = config_.queueCapacity > 0
        ? config_.queueCapacity
        : static_cast<size_t>(config_.concurrency) * 2;

    workers_ = std::make_unique<WorkerPool>(
        *engine_, info_, config_.concurrency, capacity, config_.deviceId);
}

Inference::~Inference() = default;

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

Output Inference::infer(const Input& input)
{
    // 同步模式 = 提交任务并等待：与异步共用同一条执行路径，行为可预测
    return workers_->submitBorrowed(input).get();
}

std::future<Output> Inference::inferAsync(Input input)
{
    // 调用方 std::move 传入时零拷贝；传左值则只拷贝一次
    return workers_->submit(std::move(input));
}

std::vector<std::future<Output>> Inference::inferParallel(std::vector<Input> inputs)
{
    // 先整体校验：避免中途抛异常后留下半批任务
    for (const auto& input : inputs)
    {
        if (input.size() != info_.input_elements)
            throw std::runtime_error("输入尺寸不对！");
    }

    std::vector<std::future<Output>> futures;
    futures.reserve(inputs.size());

    try
    {
        for (auto& input : inputs)
        {
            futures.push_back(workers_->submit(std::move(input)));
        }
    }
    catch (...)
    {
        // 等已提交的任务结束，避免 future 析构时阻塞在未知位置
        for (auto& future : futures) future.wait();
        throw;
    }

    return futures;
}

} // namespace TRTInfer
