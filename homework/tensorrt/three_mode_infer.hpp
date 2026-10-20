#ifndef THREE_MODE_INFER_HPP_
#define THREE_MODE_INFER_HPP_

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <future>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace TRTInfer
{

using Input = std::vector<float>;
using Output = std::vector<float>;

// 执行器：固定 worker 线程 + 有界队列，每个 worker 独占一条通道（定义在 .cpp）
class WorkerPool;

struct IOInfo
{
public:
    // 输入输出命名
    std::string input;
    std::string output;

    // 输入输出浮点数目
    size_t input_elements = 0;
    size_t output_elements = 0;

    // 输入输出字节数
    size_t in_bytes = 0;
    size_t out_bytes = 0;

    // 构造方法
    void InspectEngine(const nvinfer1::ICudaEngine& engine);

private:
    // 工具函数
    size_t calculateElements(const nvinfer1::Dims& shape) const;
    size_t calculateBytes(const size_t elements) const;
};

struct Channel
{
public:
    // 相关必须属性
    std::unique_ptr<nvinfer1::IExecutionContext> context;
    cudaStream_t stream{};
    cudaEvent_t done{};

    // cpu内存
    void* h_in = nullptr;
    void* h_out = nullptr;

    // gpu内存
    void* d_in = nullptr;
    void* d_out = nullptr;

    Channel(nvinfer1::ICudaEngine& engine, const IOInfo& info);
    ~Channel();

    Channel(const Channel&) = delete;
    Channel& operator=(const Channel&) = delete;

private:
    // 幂等释放：构造函数中途失败与析构函数共用，避免资源泄漏
    void release() noexcept;
};

struct InferConfig
{
    std::string engine_path;
    // worker(=通道)数量，即同时在执行的请求数上限；排队数由 queueCapacity 控制
    int concurrency = 1;
    int deviceId = 0;
    // 等待队列长度（不含正在执行的请求）；0 表示默认 2 × concurrency
    size_t queueCapacity = 0;
};

class Logger : public nvinfer1::ILogger
{
public:
    void log(Severity severity, const char* msg) noexcept override;

private:
    // TensorRT 的日志可能来自内部线程，串行化输出避免交错
    std::mutex mutex_;
};

class Inference
{
public:
    explicit Inference(const InferConfig& config);

    // 等待队列排空、所有 worker 退出后才销毁 engine
    ~Inference();

    Inference(const Inference&) = delete;
    Inference& operator=(const Inference&) = delete;

    // 同步：提交一个任务并等待结果
    Output infer(const Input& input);

    // 异步：提交后立刻返回 future
    std::future<Output> inferAsync(Input input);

    // 批量异步：按顺序提交，返回与输入一一对应的 future
    std::vector<std::future<Output>> inferParallel(std::vector<Input> inputs);

    size_t inputSize() const { return info_.input_elements; }
    size_t outputSize() const { return info_.output_elements; }
    int concurrency() const { return config_.concurrency; }

private:
    std::vector<char> readEngine(const std::string& engine_path);

private:
    InferConfig config_;
    std::vector<char> engine_data_;
    Logger logger_;
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;

    IOInfo info_;
    // 必须最后声明：最先析构，保证通道/context 先于 engine 销毁
    std::unique_ptr<WorkerPool> workers_;
};

}   // namespace TRTInfer
#endif // THREE_MODE_INFER_HPP_
