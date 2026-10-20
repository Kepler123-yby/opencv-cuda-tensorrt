// TensorRT 10.10 + CUDA Runtime，使用 C++17 编译。
// 对应课程：lessons/module-3/3.3-sync-async-parallel.md。
// 本例集中演示调度与资源生命周期：只接收一个静态输入和一个静态输出，
// 且两者必须是 FP32 / LINEAR / DEVICE、非形状推导 I/O。
// 这些是教学示例的边界；动态形状、多 I/O、其他类型和数据依赖输出需要另行设计。
#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

using Clock = std::chrono::steady_clock;

// TensorRT 通过此回调报告诊断信息；Logger 必须比使用它的 Runtime、Engine、Context 活得久。
// noexcept 是接口约定：不要让异常从回调传播到 TensorRT；这里只输出警告及更严重的信息。
class Logger final : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* message) noexcept override {
        if (severity <= Severity::kWARNING) std::cerr << "[TRT] " << message << '\n';
    }
};

// CUDA API 的返回值要逐次检查；抛异常后由 main 统一报告，并让 RAII 对象回收资源。
// 异步提交成功不代表 GPU 已完成。执行错误还可能在后续 query / synchronize 时才被观察到，
// 因此错误文字标识的是“在哪个调用发现问题”，不一定就是最初产生问题的位置。
void checkCuda(cudaError_t status, const char* call) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(call) + ": " + cudaGetErrorString(status));
}
#define CUDA_CHECK(call) checkCuda((call), #call)

// 这里只读取已经构建的 engine 二进制，不解析 ONNX，也不在运行时自动构建。
// binary 保留原始字节；ate 让文件打开后先停在末尾，以便获取完整文件长度。
std::vector<char> readEngine(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) throw std::runtime_error("Cannot open engine: " + path);
    const auto end = file.tellg();
    if (end <= 0) throw std::runtime_error("Empty engine or invalid file length");
    // 文件位置、vector 长度、read 的长度参数属于不同类型。
    // 先排除无效/负长度，再检查各目标类型的范围，最后转换，避免截断后分配过小的缓冲区。
    const auto length = static_cast<std::streamoff>(end);
    if (static_cast<unsigned long long>(length) > std::numeric_limits<size_t>::max()
        || length > std::numeric_limits<std::streamsize>::max())
        throw std::runtime_error("Engine file is too large");
    std::vector<char> bytes(static_cast<size_t>(length));
    file.seekg(0);
    // 已分配足够空间仍需检查 read：文件可能无法完整读取，不能把半份 engine 交给 TensorRT。
    if (!file.read(bytes.data(), static_cast<std::streamsize>(length)))
        throw std::runtime_error("Incomplete engine read");
    return bytes;
}

// frames、lanes 都是正整数。除了检查转换错误/越界，还要求完整消费字符串，
// 因而不会把 "12abc" 当作 12，也不会允许 0 进入后面的取模或统计计算。
size_t positiveInteger(const char* text) {
    const std::string value(text);
    size_t number = 0;
    const auto result = std::from_chars(value.data(), value.data() + value.size(), number);
    if (result.ec != std::errc{} || result.ptr != value.data() + value.size() || number == 0)
        throw std::runtime_error("Expected a positive integer: " + value);
    return number;
}

// 仅用于诊断打印；真正的容量检查由 elements / floatBytes 完成。
std::string shapeString(const nvinfer1::Dims& dims) {
    std::string text = "[";
    for (int i = 0; i < dims.nbDims; ++i) {
        if (i) text += ',';
        text += std::to_string(dims.d[i]);
    }
    return text + ']';
}

size_t elements(const nvinfer1::Dims& dims) {
    if (dims.nbDims < 0) throw std::runtime_error("Invalid tensor rank");
    size_t count = 1; // rank 为 0 的标量有一个元素，不等于长度为 0 的张量。
    for (int i = 0; i < dims.nbDims; ++i) {
        // -1 可能表示动态/未解析维度；0 表示本例不支持的空维度。
        // 必须在转 size_t 前拒绝负数，否则负维度会变成非常大的无符号数。
        if (dims.d[i] <= 0)
            throw std::runtime_error("Dynamic, unresolved, or empty shape: " + shapeString(dims));
        const auto dim = static_cast<size_t>(dims.d[i]);
        // 先用除法判断再乘；不能等 count * dim 已经溢出后再检查结果。
        if (count > std::numeric_limits<size_t>::max() / dim)
            throw std::runtime_error("Tensor element count overflow");
        count *= dim;
    }
    return count;
}

// 元素数合法仍不代表字节数合法；FP32 每元素还要乘 sizeof(float)。
// 此函数只适用于前面明确检查过的 FP32 / LINEAR I/O。
size_t floatBytes(size_t count) {
    if (count > std::numeric_limits<size_t>::max() / sizeof(float))
        throw std::runtime_error("Tensor byte count overflow");
    return count * sizeof(float);
}

// 保存经检查的名字与容量，供所有 Slot 使用同一份元信息。
// 名字存成 std::string，自行持有文本；元素数用于 vector 长度，字节数用于 CUDA 拷贝/分配。
struct IOInfo {
    std::string input;
    std::string output;
    size_t in_elements = 0;
    size_t out_elements = 0;
    size_t in_bytes = 0;
    size_t out_bytes = 0;
};

// 从实际 engine 查询 I/O，不能根据模型文件名、常见输入名或 FP16 构建选项猜测 I/O 类型。
IOInfo inspectEngine(const nvinfer1::ICudaEngine& engine) {
    IOInfo info;
    int inputs = 0, outputs = 0;
    for (int i = 0; i < engine.getNbIOTensors(); ++i) {
        const char* name = engine.getIOTensorName(i);
        if (!name) throw std::runtime_error("Missing I/O tensor name");
        const auto mode = engine.getTensorIOMode(name);
        const auto shape = engine.getTensorShape(name);
        std::cout << name << ' '
                  << (mode == nvinfer1::TensorIOMode::kINPUT ? "INPUT" : "OUTPUT")
                  << " shape=" << shapeString(shape)
                  << " dtype=" << static_cast<int>(engine.getTensorDataType(name))
                  << " format=" << engine.getTensorFormatDesc(name)
                  << " location=" << static_cast<int>(engine.getTensorLocation(name)) << '\n';
        // 非 LINEAR 格式可能含向量化/填充，不能直接按维度乘积解释为 float 数组。
        // HOST I/O 与形状推导 I/O 的绑定/准备规则也不同，本例明确拒绝而不是套用同一条路径。
        if (engine.getTensorDataType(name) != nvinfer1::DataType::kFLOAT
            || engine.getTensorFormat(name) != nvinfer1::TensorFormat::kLINEAR
            || engine.getTensorLocation(name) != nvinfer1::TensorLocation::kDEVICE
            || engine.isShapeInferenceIO(name))
            throw std::runtime_error("Requires FP32/LINEAR/DEVICE, non-shape I/O: "
                                     + std::string(name));
        const size_t count = elements(shape); // 分配资源前就拒绝动态、未解析或空维度。
        if (mode == nvinfer1::TensorIOMode::kINPUT) {
            info.input = name;
            info.in_elements = count;
            ++inputs;
        } else if (mode == nvinfer1::TensorIOMode::kOUTPUT) {
            info.output = name;
            info.out_elements = count;
            ++outputs;
        } else {
            throw std::runtime_error("Invalid I/O mode");
        }
    }
    // 逐项计数后再确认恰好一进一出，避免循环覆盖名字而悄悄漏掉其他张量。
    if (inputs != 1 || outputs != 1)
        throw std::runtime_error("This example requires exactly one input and one output");
    info.in_bytes = floatBytes(info.in_elements);
    info.out_bytes = floatBytes(info.out_elements);
    std::cout << "I/O per slot: input=" << info.in_bytes << " bytes, output=" << info.out_bytes
              << " bytes (context internal memory is additional).\n";
    return info;
}

// 一路推理独占这组资源：h_in / h_out 是 CPU 可访问的页锁定内存，
// d_in / d_out 是 GPU 输入/输出显存。地址绑定和 H2D / D2H 都不会替我们申请这四块内存。
// 句柄与地址初始为空，分配到一半抛异常时也能按“已成功创建的部分”清理。
struct CudaResources {
    cudaStream_t stream{};
    cudaEvent_t done{};
    void* d_in = nullptr;
    void* d_out = nullptr;
    void* h_in = nullptr;
    void* h_out = nullptr;

    CudaResources() = default;
    CudaResources(const CudaResources&) = delete;
    CudaResources& operator=(const CudaResources&) = delete;
    ~CudaResources() noexcept {
        // 先尽力等本 stream 的工作结束，再销毁事件、缓冲区与 stream，避免 GPU 仍在使用内存。
        // 析构不能抛异常，尤其不能在处理原异常时再抛一个；这里不把清理作为错误恢复机制。
        // 正常路径在 collect 中显式同步并检查错误；出现提交/执行错误后，本例直接退出。
        if (stream) cudaStreamSynchronize(stream);
        if (done) cudaEventDestroy(done);
        if (d_in) cudaFree(d_in);
        if (d_out) cudaFree(d_out);
        if (h_in) cudaFreeHost(h_in);
        if (h_out) cudaFreeHost(h_out);
        if (stream) cudaStreamDestroy(stream);
    }
};

// Slot 是“一组独立 Context + stream + 缓冲区 + 完成事件”，最多持有一帧未消费的结果。
// 多个 Slot 可共享 engine 权重，但不会并发复用同一个 Context 或同一组 I/O 缓冲区。
// 本例由一个 CPU 线程调度所有 Slot；busy_ 只是该线程的状态记录，不是线程同步锁。
class Slot {
public:
    Slot(nvinfer1::ICudaEngine& engine, const IOInfo& io)
        : io_(io), context_(engine.createExecutionContext()) {
        if (!context_) throw std::runtime_error("createExecutionContext failed");
        // 显式使用非默认 stream。NonBlocking 控制与 legacy 默认 stream 的隐式同步关系，
        // 不保证所有 API 都立即返回，也不会让后面的 EventSynchronize 变成非阻塞调用。
        CUDA_CHECK(cudaStreamCreateWithFlags(&cuda_.stream, cudaStreamNonBlocking));
        // done 只表示“这帧是否完成”，不用于测 GPU 时间，因此关闭事件时间戳记录。
        CUDA_CHECK(cudaEventCreateWithFlags(&cuda_.done, cudaEventDisableTiming));
        // 前面只允许静态形状，这里仍检查 Context 的形状推导是否完整，并再次确认所需元素数。
        // inferShapes(0, nullptr) 不请求返回缺失张量名；非 0 就不应继续按猜测的容量执行。
        if (context_->inferShapes(0, nullptr) != 0)
            throw std::runtime_error("Shape inference failed");
        if (elements(context_->getTensorShape(io_.input.c_str())) != io_.in_elements
            || elements(context_->getTensorShape(io_.output.c_str())) != io_.out_elements)
            throw std::runtime_error("Context shape differs from the static engine");
        CUDA_CHECK(cudaMalloc(&cuda_.d_in, io_.in_bytes));
        CUDA_CHECK(cudaMalloc(&cuda_.d_out, io_.out_bytes));
        CUDA_CHECK(cudaMallocHost(&cuda_.h_in, io_.in_bytes));
        CUDA_CHECK(cudaMallocHost(&cuda_.h_out, io_.out_bytes));
        // DEVICE I/O 绑定的是显存地址。setTensorAddress 只登记“张量从哪里读/往哪里写”，
        // 不会把 h_in 内容复制到 d_in；真正的数据传输在每次 submit 中显式排队。
        if (!context_->setTensorAddress(io_.input.c_str(), cuda_.d_in)
            || !context_->setTensorAddress(io_.output.c_str(), cuda_.d_out))
            throw std::runtime_error("setTensorAddress failed");
    }

    Slot(const Slot&) = delete;
    Slot& operator=(const Slot&) = delete;

    bool busy() const noexcept { return busy_; }
    uint64_t frameId() const noexcept { return frame_id_; }

    // 只有 idle Slot 才能接新帧；初始即为空闲，使用过的槽位必须 collect 完成后才恢复空闲。
    // GPU 已完成但尚未消费结果时仍然 busy，不能提前把这个槽位交给下一帧。
    // 输入 vector 只在本函数的普通 memcpy 期间被读取；返回后 CPU 可以继续准备其他帧。
    void submit(uint64_t id, const std::vector<float>& input) {
        if (busy_) throw std::logic_error("submit requires an idle slot");
        if (input.size() != io_.in_elements) throw std::invalid_argument("Wrong input length");
        // 异步 H2D 的源始终是本 Slot 独占的 pinned h_in，不直接借用调用方普通 vector 的内存。
        // h_in 在本帧结束前保持不变；输入长度也必须精确匹配，避免越界读取或留下旧输入尾部。
        std::memcpy(cuda_.h_in, input.data(), io_.in_bytes);
        frame_id_ = id;
        busy_ = true; // 提交中途失败也不标回 idle；main 会退出，由 RAII 等待并清理。
        // 同一 stream 保证 H2D → 推理 → D2H → done 的依赖顺序，无需在各步之间阻塞 CPU。
        // enqueueV3 返回 true 只表示提交成功，不能据此读取 h_out 或复用这组内存。
        CUDA_CHECK(cudaMemcpyAsync(cuda_.d_in, cuda_.h_in, io_.in_bytes,
                                   cudaMemcpyHostToDevice, cuda_.stream));
        if (!context_->enqueueV3(cuda_.stream)) throw std::runtime_error("enqueueV3 failed");
        CUDA_CHECK(cudaMemcpyAsync(cuda_.h_out, cuda_.d_out, io_.out_bytes,
                                   cudaMemcpyDeviceToHost, cuda_.stream));
        CUDA_CHECK(cudaEventRecord(cuda_.done, cuda_.stream)); // 放在 D2H 后，覆盖整条结果回传链。
    }

    // 查询是非阻塞的观察操作，不会消费结果，也不会把 Slot 变回 idle。
    // 调度方可查询一次后处理其他工作；不要反复空转查询，耗尽 CPU 却没有推进准备/消费。
    bool ready() const {
        if (!busy_) return false; // idle 时没有待取结果，也避免查询尚未记录过的事件。
        const auto status = cudaEventQuery(cuda_.done);
        if (status == cudaSuccess) return true;
        // NotReady 是正常状态；其他错误必须上报，不能一律当作“还没完成”而无限等待。
        if (status == cudaErrorNotReady) return false;
        checkCuda(status, "cudaEventQuery(done)");
        return false;
    }

    // 按原 frame_id 取回结果，避免环形复用 Slot 时把上一帧结果挂到新帧名下。
    // expected_id 是本例单线程内部调度的核对值，不是面向并发调用方的外部请求句柄。
    void collect(uint64_t expected_id, std::vector<float>& output) {
        if (!busy_) throw std::logic_error("collect requires a busy slot");
        if (frame_id_ != expected_id) throw std::logic_error("Output frame ID mismatch");
        // 等到 done 才能读取 pinned h_out；同时在这里检查此前异步工作可能延迟报告的错误。
        CUDA_CHECK(cudaEventSynchronize(cuda_.done));
        output.resize(io_.out_elements);
        // 将结果复制给调用方拥有的 vector，调用方不再依赖稍后会被下一帧覆盖的 h_out。
        std::memcpy(output.data(), cuda_.h_out, io_.out_bytes);
        busy_ = false; // 输出复制完成后，四块缓冲区才一起恢复为可复用状态。
    }

private:
    // 这里只借用 IOInfo，不复制它；调用方必须让 io 在所有 Slot 析构前一直有效。
    // 同理，创建 Context 的 engine 以及其 Runtime / Logger 也必须保持有效。
    const IOInfo& io_;
    // 成员按声明的相反顺序析构：先由 cuda_ 等待 GPU 并释放 I/O，再销毁 context_。
    // 该顺序对构造函数体中途抛异常同样成立，因为这两个成员那时已经完成构造。
    std::unique_ptr<nvinfer1::IExecutionContext> context_;
    CudaResources cuda_;
    // busy_ 表示资源/结果归属；完成事件表示 GPU 进度，二者含义不同。
    bool busy_ = false;
    uint64_t frame_id_ = 0;
};

// 用帧编号生成可重复的伪输入，把“准备下一帧”表示成一段真实 CPU 工作。
// id % 7 对应 7 种输入，所以 id=7 时可复用第 0 类同步参考，无需保存每帧参考。
// 调用方先按模型容量 resize；这里仅填数据，不在逐帧路径中分配 vector。
// 这些数据不代表真实图片，也不能用来判断模型精度；某些模型可能对它们输出相近。
void prepareInput(uint64_t id, std::vector<float>& input) {
    const auto phase = static_cast<size_t>(id % 7);
    for (size_t i = 0; i < input.size(); ++i)
        input[i] = static_cast<float>((i % 251 + phase * 17) % 251) / 250.0F;
}

// 每个 Context 都单独预热，并立即回收每次结果，避免预热请求遗留到正式测量。
// 10 次只是教学起点；实际部署应观察懒初始化、设备时钟等是否已经稳定。
void warmup(Slot& slot, const IOInfo& io) {
    std::vector<float> input(io.in_elements), output(io.out_elements);
    for (uint64_t id = 0; id < 10; ++id) {
        prepareInput(id, input);
        slot.submit(id, input);
        slot.collect(id, output);
    }
}

// 数组下标是“输入模式编号”，每个 vector 拥有该模式完整的同步参考输出。
// 只保存 7 份输出，避免按测量帧数保存全部参考而占用大量主机内存。
using References = std::array<std::vector<float>, 7>;

// 用独立的同步槽位建立基线，函数结束时该槽位已排空并销毁。
// 三种待比较模式后续都复用同一个 Engine 和这组参考，而不是重新构建模型。
References makeReferences(nvinfer1::ICudaEngine& engine, const IOInfo& io) {
    Slot slot(engine, io);
    warmup(slot, io);
    References reference;
    std::vector<float> input(io.in_elements);
    for (uint64_t id = 0; id < reference.size(); ++id) {
        prepareInput(id, input);
        slot.submit(id, input);
        slot.collect(id, reference[id]);
        // 参考本身有 NaN / Inf 时应先排查模型或输入，不能把它当作合法比较基准。
        for (float value : reference[id])
            if (!std::isfinite(value)) throw std::runtime_error("Non-finite reference output");
    }
    return reference;
}

// 检查整份输出，而不只看首元素或 checksum，便于发现错帧、拷贝错误等问题。
// 容差为 atol + rtol * abs(reference)：接近零时靠绝对误差，其余靠相对误差。
// 这组阈值是示例约定；真实模型应先确定允许的误差，再做模式比较。
void compareOutput(uint64_t id, const std::vector<float>& output, const References& reference) {
    constexpr double atol = 1e-4, rtol = 1e-3;
    const auto& expected = reference[id % reference.size()]; // 必须与 prepareInput 的分组一致。
    if (output.size() != expected.size()) throw std::runtime_error("Output length mismatch");
    for (size_t i = 0; i < output.size(); ++i) {
        const double actual = output[i], target = expected[i];
        if (!std::isfinite(actual) || !std::isfinite(target)
            || std::abs(actual - target) > atol + rtol * std::abs(target))
            throw std::runtime_error("Validation failed: frame=" + std::to_string(id)
                + " element=" + std::to_string(i) + " actual=" + std::to_string(actual)
                + " reference=" + std::to_string(target));
    }
}

// 三种模式共用 Slot 的安全提交/回收接口，只改变 CPU 调度顺序和槽位数量。
enum class Mode { Sync, Async, Parallel };

const char* modeName(Mode mode) {
    switch (mode) {
    case Mode::Sync: return "sync";
    case Mode::Async: return "async";
    case Mode::Parallel: return "parallel";
    }
    throw std::logic_error("Invalid mode");
}

struct Measurement {
    double wall_seconds = 0;          // 整批首帧开始到末帧取走输出的窗口，用于计算 FPS。
    std::vector<double> latency_ms;   // 每个请求自己的延迟，不能用 wall / frames 替代。
};

// Consumer 是“拿到一帧输出后做什么”的回调类型：
// 正确性运行传入逐元素比较，性能运行只做轻量 checksum，避免复制两套调度代码。
// [&] lambda 通过引用访问外层对象；consume 必须在本次调用内完成，若要保存结果，
// 需自行复制 output，因为同一个输出 vector 在下一次 collect 时还会被复用。
// 内部前置条件：main 已保证 frames > 0，runMode 已创建至少一个空闲槽位。
template <class Consumer>
Measurement runSchedule(Mode mode, std::vector<std::unique_ptr<Slot>>& slots,
                        const IOInfo& io, size_t frames, Consumer consume) {
    // CPU 缓冲与时间戳数组在计时前分配；测量覆盖填数据和传输，不混入这些分配。
    // 两份 input 用于异步模式的交替准备，它们都是普通 CPU 内存。
    // 真正作为 cudaMemcpyAsync 源地址的 pinned 输入由各个 Slot 独立持有。
    std::array<std::vector<float>, 2> input{
        std::vector<float>(io.in_elements), std::vector<float>(io.in_elements)};
    std::vector<float> output(io.out_elements); // 从槽位复制出来，consume 返回后可被下一帧覆盖。
    std::vector<Clock::time_point> starts(frames);
    Measurement result;
    result.latency_ms.resize(frames);
    Clock::time_point first_start{}, last_finish{};
    // 用 steady_clock 的同一时间基准记录 CPU 侧端到端区间，不能与 CUDA event 相减。
    // id 为本轮 0..frames-1；下一轮只有在本轮全部排空后才会复用这些编号。
    auto startFrame = [&](size_t id) {
        starts[id] = Clock::now();
        if (id == 0) first_start = starts[id];
    };
    auto collectFrame = [&](Slot& slot, size_t id) {
        slot.collect(id, output); // 核对请求身份，等待 D2H，再复制到独立的 CPU vector。
        // 本例以“输出复制完成”为请求结束点，下面 consume 的耗时不计入本帧延迟。
        // 若接入 CPU NMS 等业务逻辑并报告完整业务延迟，应把结束时间移到它们之后。
        last_finish = Clock::now();
        result.latency_ms[id] =
            std::chrono::duration<double, std::milli>(last_finish - starts[id]).count();
        consume(id, output);
    };

    if (mode == Mode::Sync) {
        // 同步基线：准备 -> 提交 -> 等待并取走，前一帧结束后才开始下一帧。
        Slot& slot = *slots[0];
        for (size_t id = 0; id < frames; ++id) {
            startFrame(id);
            prepareInput(id, input[0]);
            slot.submit(id, input[0]);
            collectFrame(slot, id); // 内部用异步 API，应用接口仍是“提交后立即等”。
        }
    } else if (mode == Mode::Async) {
        // 单路异步：只有一个 GPU 请求在途，但等待前先在 CPU 上准备下一帧。
        // 第 0 帧没有前驱，先单独准备并提交；最后一帧不再生成多余的下一帧。
        Slot& slot = *slots[0];
        startFrame(0);
        prepareInput(0, input[0]);
        slot.submit(0, input[0]);
        for (size_t id = 0; id < frames; ++id) {
            if (id + 1 < frames) {
                startFrame(id + 1);
                // id 与 id+1 使用不同的 CPU vector；此时 GPU 仍可能在处理第 id 帧。
                // 不能把下一帧直接写入该槽位的 pinned h_in，它可能仍被 H2D 读取。
                prepareInput(id + 1, input[(id + 1) % 2]);
            }
            // 示范一次非等待查询；false 只代表尚未完成，collect 仍会负责正确等待。
            // 这里不循环查询，以免用一个 CPU 核不停忙轮询。
            (void)slot.ready();
            collectFrame(slot, id);
            // 旧帧输出已经取走，槽位恢复空闲，才能提交刚才准备好的下一帧。
            if (id + 1 < frames) slot.submit(id + 1, input[(id + 1) % 2]);
        }
    } else {
        // 多路并行：一条 CPU 调度线程轮流向多个 Context / stream 提交。
        // 不同 stream 有重叠机会，但实际并行程度取决于 GPU、模型与资源竞争。
        for (size_t id = 0; id < frames; ++id) {
            // 在等空槽位之前记开始时间，把槽位满时的背压等待计入本帧延迟。
            startFrame(id);
            Slot& slot = *slots[id % slots.size()];
            // 环形索引绕回时，这里装着 id-lanes 的旧帧。
            // 先回收旧输出再写新输入；直接覆盖会破坏在途数据或丢失结果。
            if (slot.busy()) collectFrame(slot, static_cast<size_t>(slot.frameId()));
            prepareInput(id, input[0]);
            slot.submit(id, input[0]);
        }
        // 尾部排空（drain）：停止提交后，最后 min(frames, lanes) 帧仍未被消费。
        // 例如 frames=8、lanes=3，待取帧为 5/6/7，槽位顺序为 2/0/1。
        // 按帧编号取模，而不是直接遍历槽位 0/1/2，才能保序并覆盖非整除情形。
        const size_t first_pending = frames > slots.size() ? frames - slots.size() : 0;
        for (size_t id = first_pending; id < frames; ++id)
            collectFrame(*slots[id % slots.size()], id);
    }
    // FPS 窗口覆盖首帧到尾帧，包含填充/排空和中间调度开销，不能在最后一次提交就停表。
    // 结束点仍是末帧 output 复制完成：中间 consume 会影响窗口，末帧 consume 在窗口外。
    // 本例消费仅用于检查或 checksum；这个窗口不代表包含 NMS 的完整检测器耗时。
    result.wall_seconds = std::chrono::duration<double>(last_finish - first_start).count();
    return result;
}

// sorted 必须已升序且非空，fraction 在 [0,1]；本例只查询 p50 / p95。
// 在位置 (N-1)*p 的相邻样本间线性插值，与第 5.2 节统计工具保持同一口径。
double percentile(const std::vector<double>& sorted, double fraction) {
    const double position = fraction * static_cast<double>(sorted.size() - 1);
    const auto low = static_cast<size_t>(position);
    const auto high = std::min(low + 1, sorted.size() - 1);
    return sorted[low] + (sorted[high] - sorted[low]) * (position - low);
}

void runMode(nvinfer1::ICudaEngine& engine, const IOInfo& io, const References& reference,
             Mode mode, size_t frames, size_t requested_lanes) {
    // sync / async 始终只有一个槽位；命令行 lanes 只控制 parallel 的并发上限。
    const size_t lanes = mode == Mode::Parallel ? requested_lanes : 1;
    // 模式依次创建、运行、销毁，避免三种模式的 Context / 缓冲显存同时叠加。
    // unique_ptr 负责唯一所有权，也避免 vector 扩容时复制含 GPU 资源的 Slot。
    std::vector<std::unique_ptr<Slot>> slots;
    slots.reserve(lanes);
    for (size_t i = 0; i < lanes; ++i) {
        slots.push_back(std::make_unique<Slot>(engine, io));
        warmup(*slots.back(), io);
    }
    // 至少覆盖全部 7 类输入，并让槽位轮转超过两圈，检查复用与尾部排空。
    // 即使正式测量只请求 1 帧，也会先做这段独立验证；验证时间不属于性能窗口。
    const size_t validation_frames = std::max<size_t>(7, 2 * lanes + 3);
    runSchedule(mode, slots, io, validation_frames,
        [&](uint64_t id, const std::vector<float>& output) { compareOutput(id, output, reference); });
    std::cout << modeName(mode) << ": validation passed (" << validation_frames
              << " frames, all output elements, atol=1e-4 rtol=1e-3).\n";

    // 再跑一轮正式测量：逐元素比较已在上面完成，计时路径只做轻量输出摘要。
    // checksum 方便观察程序确实读到输出，不能替代数值正确性检查或错帧验证。
    double checksum = 0;
    auto measurement = runSchedule(mode, slots, io, frames,
        [&](uint64_t, const std::vector<float>& output) {
            checksum += static_cast<double>(output.front()) + output.back();
        });
    // mean 是各请求 latency 的平均值；FPS 则用整批完成数除以总窗口。
    // 多路请求的延迟区间有重叠，因此 FPS 通常不等于 1000 / mean。
    const double mean = std::accumulate(measurement.latency_ms.begin(),
                                        measurement.latency_ms.end(), 0.0) / frames;
    std::sort(measurement.latency_ms.begin(), measurement.latency_ms.end()); // 分位数要求先排序。
    std::cout << std::fixed << std::setprecision(3)
              << modeName(mode) << ": lanes=" << lanes << " frames=" << frames
              << " wall=" << measurement.wall_seconds * 1000.0 << " ms"
              << " FPS=" << frames / measurement.wall_seconds
              << " request-latency mean=" << mean
              << " ms p50=" << percentile(measurement.latency_ms, 0.50)
              << " ms p95=" << percentile(measurement.latency_ms, 0.95)
              << " ms checksum=" << checksum << '\n';
}

int main(int argc, char** argv) {
    if (argc < 2 || argc > 4) {
        std::cerr << "Usage: " << argv[0] << " model.engine [frames=300] [lanes=3]\n";
        return 1;
    }
    try {
        // argv[1] 是已构建的 .engine，不是 ONNX；本程序不负责构建或缓存 Engine。
        // 帧数和路数严格解析为正整数，防止空统计、环形索引除零等无意义输入。
        const size_t frames = argc >= 3 ? positiveInteger(argv[2]) : 300;
        const size_t lanes = argc >= 4 ? positiveInteger(argv[3]) : 3;
        // 后面用 2*lanes+3 计算验证帧数，提前避免无符号整数溢出。
        if (lanes > (std::numeric_limits<size_t>::max() - 3) / 2)
            throw std::runtime_error("Lane count is too large");
        // 局部对象逆序销毁：各 runMode 的槽位先销毁，再 Engine、Runtime，最后 Logger。
        // io 也会活到全部 Slot 结束，满足 Slot 内部对 IOInfo 的引用生命周期。
        Logger logger;
        const auto bytes = readEngine(argv[1]);
        auto runtime = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(logger));
        if (!runtime) throw std::runtime_error("createInferRuntime failed");
        auto engine = std::unique_ptr<nvinfer1::ICudaEngine>(
            runtime->deserializeCudaEngine(bytes.data(), bytes.size()));
        if (!engine) throw std::runtime_error("deserializeCudaEngine failed");
        // 先检查模型契约，再创建参考和各模式资源；不支持的 I/O 会在这里明确拒绝。
        const auto io = inspectEngine(*engine);
        const auto reference = makeReferences(*engine, io);
        std::cout << "Seven sync reference patterns generated; finite-output check passed.\n"
                  << "Timing: steady_clock; request starts before CPU input preparation "
                     "(parallel: before waiting for a slot), ends after copying output to "
                     "an owned vector.\n"
                  << "FPS includes CPU preparation, input/output copies, fill and drain; "
                     "excludes resource creation, reference, warmup and validation.\n"
                  << "Synthetic inputs test scheduling consistency, not model accuracy.\n";
        // 固定顺序便于学习。正式性能比较应交错/随机顺序多轮运行，控制热状态与负载。
        for (const auto mode : {Mode::Sync, Mode::Async, Mode::Parallel})
            runMode(*engine, io, reference, mode, frames, lanes);
        return 0;
    } catch (const std::exception& error) {
        // 异常沿栈展开时由 RAII 先等待、再释放资源；本例停止运行，不盲目复用失败槽位。
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
