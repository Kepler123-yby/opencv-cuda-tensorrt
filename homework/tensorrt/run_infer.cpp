#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <iostream>
#include <vector>
#include <algorithm>
#include <fstream>
#include <limits>
#include <memory>
#include <cstring>
#include <stdexcept>
#include <string>

void checkCuda(cudaError_t status, const char* call)
{
    if (status != cudaSuccess)
    {
        throw std::runtime_error(std::string(call) + ": " + cudaGetErrorString(status));
    }
}
#define CUDA_CHECK(call) checkCuda((call), #call)

class Logger : public nvinfer1::ILogger
{
public:
    void log(Severity severity, const char* msg) noexcept override
    {
        if (severity <= Severity::kWARNING)
            std::cout << "[TRT] " << msg << '\n';
    }
};

std::string shapeString(const nvinfer1::Dims& dims) {
    std::string text = "[";
    for (int i = 0; i < dims.nbDims; ++i) {
        if (i) text += ",";
        text += std::to_string(dims.d[i]);
    }
    return text + "]";
}

struct Information
{
    const char* input = nullptr;
    const char* output = nullptr;
    int inputs = 0;
    int outputs = 0;
    std::vector<const char*> names;
    std::vector<nvinfer1::TensorIOMode> modes;
    std::vector<nvinfer1::Dims> shapes;
    std::vector<nvinfer1::DataType> dtypes;
    std::vector<nvinfer1::TensorFormat> formats;
    std::vector<nvinfer1::TensorLocation> wheres;

    void getInformation(const std::unique_ptr<nvinfer1::ICudaEngine>& engine)
    {
        for (int i = 0; i < engine->getNbIOTensors(); ++i)
        {
            const char* name = engine->getIOTensorName(i);
            auto mode = engine->getTensorIOMode(name);
            auto shape = engine->getTensorShape(name);
            auto dtype = engine->getTensorDataType(name);
            auto format = engine->getTensorFormat(name);
            auto where = engine->getTensorLocation(name);

            std::cout << name << " "
                      << (mode == nvinfer1::TensorIOMode::kINPUT ? "INPUT" : "OUTPUT")
                      << " engine=" << shapeString(shape)
                      << " dtype=" << static_cast<int>(dtype)
                      << " format=" << engine->getTensorFormatDesc(name)
                      << " location=" << static_cast<int>(where) << '\n';

            names.push_back(name);
            modes.push_back(mode);
            shapes.push_back(shape);
            dtypes.push_back(dtype);
            formats.push_back(format);
            wheres.push_back(where);

            if (dtype != nvinfer1::DataType::kFLOAT ||
                format != nvinfer1::TensorFormat::kLINEAR ||
                where != nvinfer1::TensorLocation::kDEVICE ||
                engine->isShapeInferenceIO(name))
                throw std::runtime_error("本例只支持 FP32/LINEAR/DEVICE 且非形状推导 I/O: " + std::string(name));

            if (mode == nvinfer1::TensorIOMode::kINPUT) {
                input = name;
                ++inputs;
            } else if (mode == nvinfer1::TensorIOMode::kOUTPUT) {
                output = name;
                ++outputs;
            }
        }
        if (inputs != 1 || outputs != 1)
            throw std::runtime_error("本例只支持一个输入和一个输出张量");
    }
};

struct CudaResources
{
    cudaStream_t stream{};
    void* d_in = nullptr;
    void* d_out = nullptr;
    void* h_in = nullptr;
    void* h_out = nullptr;

    CudaResources() = default;
    CudaResources(const CudaResources&) = delete;
    CudaResources& operator=(const CudaResources&) = delete;

    ~CudaResources()
    {
        if (stream) cudaStreamSynchronize(stream);
        if (d_in) cudaFree(d_in);
        if (d_out) cudaFree(d_out);
        if (h_in) cudaFreeHost(h_in);
        if (h_out) cudaFreeHost(h_out);
        if (stream) cudaStreamDestroy(stream);
    }
};

class TRTInfer
{
public:
    explicit TRTInfer(const std::string& engine_path) : engine_date_(read_engine(engine_path))
    {
        runtime_.reset(nvinfer1::createInferRuntime(logger_));
        if (!runtime_) throw std::runtime_error("TensorRT Runtime 创建失败");
        engine_.reset(runtime_->deserializeCudaEngine(engine_date_.data(), engine_date_.size()));
        if (!engine_) throw std::runtime_error("Engine 反序列化失败");
        context_.reset(engine_->createExecutionContext());
        if (!context_) throw std::runtime_error("ExecutionContext 创建失败");

        im_.getInformation(engine_);

        CUDA_CHECK(cudaStreamCreateWithFlags(&buffer_.stream, cudaStreamNonBlocking));
        const auto actual = engine_->getTensorShape(im_.input);
        for (int i = 0; i < actual.nbDims; ++i)
            if (actual.d[i] <= 0)
                throw std::runtime_error("需要使用静态维度的engine");

        if (context_->inferShapes(0, nullptr) != 0)
            throw std::runtime_error("形状推导失败或输入信息不完整！");

        const auto indims = context_->getTensorShape(im_.input);
        const auto outdims = context_->getTensorShape(im_.output);
        input_elements_ = get_elements(indims);
        output_elements_ = get_elements(outdims);
        const auto in_bytes = floatBytes(input_elements_);
        const auto out_bytes = floatBytes(output_elements_);

        std::cout << "输入：" << shapeString(indims) << " " << in_bytes << "字节\n";
        std::cout << "输出：" << shapeString(outdims) << " " << out_bytes << "字节\n";

        CUDA_CHECK(cudaMalloc(&buffer_.d_in, in_bytes));
        CUDA_CHECK(cudaMalloc(&buffer_.d_out, out_bytes));
        CUDA_CHECK(cudaMallocHost(&buffer_.h_in, in_bytes));
        CUDA_CHECK(cudaMallocHost(&buffer_.h_out, out_bytes));
    }

    TRTInfer(const TRTInfer&) = delete;
    TRTInfer& operator=(const TRTInfer&) = delete;

    // 外部调用接口：输入/输出均为连续 FP32 数据，输出元素数由 outputSize() 获取。
    std::vector<float> infer(const float* input, size_t input_size)
    {
        std::vector<float> output;
        infer(input, input_size, output);
        return output;
    }

    // 可复用调用方输出缓冲区，适合循环推理以减少主机内存分配。
    // 同一个实例不可被多个线程同时调用。
    void infer(const float* input, size_t input_size, std::vector<float>& output)
    {
        if (!input) throw std::runtime_error("推理输入不能为空");
        if (input_size != input_elements_)
            throw std::runtime_error("推理输入元素数量与 Engine 输入形状不匹配");

        output.resize(output_elements_);
        const auto bytes = floatBytes(input_elements_);
        std::memcpy(buffer_.h_in, input, bytes);
        CUDA_CHECK(cudaMemcpyAsync(buffer_.d_in, buffer_.h_in, bytes, cudaMemcpyHostToDevice, buffer_.stream));

        if (!context_->setTensorAddress(im_.input, buffer_.d_in) ||
            !context_->setTensorAddress(im_.output, buffer_.d_out))
            throw std::runtime_error("Tensor 地址绑定失败");
        if (!context_->enqueueV3(buffer_.stream))
            throw std::runtime_error("TensorRT 推理执行失败");

        const auto output_bytes = floatBytes(output_elements_);
        CUDA_CHECK(cudaMemcpyAsync(buffer_.h_out, buffer_.d_out, output_bytes, cudaMemcpyDeviceToHost, buffer_.stream));
        CUDA_CHECK(cudaStreamSynchronize(buffer_.stream));

        std::memcpy(output.data(), buffer_.h_out, output_bytes);
    }

    size_t inputSize() const { return input_elements_; }
    size_t outputSize() const { return output_elements_; }

private:
    std::vector<char> engine_date_;
    Logger logger_;
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;
    Information im_;
    CudaResources buffer_;
    size_t input_elements_ = 0;
    size_t output_elements_ = 0;

    std::vector<char> read_engine(const std::string& engine_path) const
    {
        std::vector<char> engine;
        std::ifstream file(engine_path, std::ios::binary | std::ios::ate);
        if (!file) throw std::runtime_error("无法读取 Engine 文件！");

        auto end = file.tellg();
        if (end <= 0) throw std::runtime_error("无法获取文件长度！");
        const auto length = static_cast<std::streamoff>(end);
        const auto file_size = static_cast<std::uintmax_t>(length);
        if (file_size > std::numeric_limits<size_t>::max() ||
            file_size > static_cast<std::uintmax_t>(std::numeric_limits<std::streamsize>::max()))
            throw std::runtime_error("Engine 文件过大！");

        engine.resize(static_cast<size_t>(length));
        file.seekg(0, std::ios::beg);
        if (!file) throw std::runtime_error("文件指针定位失败！");
        if (!file.read(engine.data(), static_cast<std::streamsize>(length)))
            throw std::runtime_error("Engine 数据读取不完整！");

        std::cout << "Engine 文件读取成功！\n"
                  << "Engine 大小: " << engine.size() << " bytes\n";
        return engine;
    }

    size_t get_elements(const nvinfer1::Dims& dims) const {
        if (dims.nbDims < 0) throw std::runtime_error("无效形状");
        size_t count = 1;
        for (int i = 0; i < dims.nbDims; ++i) {
            if (dims.d[i] <= 0)
                throw std::runtime_error("未解析维度或本例不支持的空张量: " + shapeString(dims));
            const auto dim = static_cast<size_t>(dims.d[i]);
            if (count > std::numeric_limits<size_t>::max() / dim)
                throw std::runtime_error("元素数溢出");
            count *= dim;
        }
        return count;
    }

    size_t floatBytes(size_t count) const
    {
        if (count > std::numeric_limits<size_t>::max() / sizeof(float))
            throw std::runtime_error("字节数溢出！");
        return count * sizeof(float);
    }
};

int main(int argc, char* argv[])
{
    const std::string engine_path = argc > 1 ? argv[1] : "model/model.engine";

    try
    {
        TRTInfer infer(engine_path);
        std::vector<float> input(infer.inputSize());
        std::fill_n(input.begin(), input.size(), 1.0f);

        std::vector<float> output;
        infer.infer(input.data(), input.size(), output);

        std::cout << "伪输入推理完成，输入元素数: " << input.size()
                  << "，输出元素数: " << output.size() << '\n';
        if (!output.empty())
            std::cout << "输出首元素: " << output.front() << '\n';
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "推理测试失败: " << error.what() << '\n';
        return 1;
    }
}
