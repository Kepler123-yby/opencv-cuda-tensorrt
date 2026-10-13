#include <NvInfer.h>
#include <NvOnnxParser.h>

#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

// TensorRT 10.x 对象由工厂函数创建，使用 delete 释放。
// unique_ptr 默认删除器即可管理生命周期，不再使用旧版 destroy()。
template <typename T>
using TrtPtr = std::unique_ptr<T>;

class Logger final : public nvinfer1::ILogger
{
public:
    void log(Severity severity, const char* message) noexcept override
    {
        // 只输出警告和错误；枚举值越小，严重程度越高。
        if (severity <= Severity::kWARNING)
            std::cerr << "[TensorRT] " << message << '\n';
    }
};

class EngineBuilder
{
public:
    // 完整流程：解析 ONNX → 构建并序列化 TensorRT 引擎 → 保存 .engine 文件。
    // 此示例面向固定输入形状；动态输入还需要配置 optimization profile。
    void build(const std::string& onnxPath, const std::string& enginePath)
    {
        // 1. 创建构建器，负责优化网络、选择计算实现并生成引擎。
        TrtPtr<nvinfer1::IBuilder> builder{nvinfer1::createInferBuilder(logger_)};
        if (!builder)
            throw std::runtime_error("Failed to create TensorRT builder");

        // 2. 创建网络容器，用于接收 ONNX 解析出的层、权重和输入输出信息。
        // TensorRT 10.x 始终使用显式 batch，传 0 即可，无需 kEXPLICIT_BATCH。
        TrtPtr<nvinfer1::INetworkDefinition> network{builder->createNetworkV2(0)};
        if (!network)
            throw std::runtime_error("Failed to create network");

        TrtPtr<nvonnxparser::IParser> parser{
            nvonnxparser::createParser(*network, logger_)};
        if (!parser)
            throw std::runtime_error("Failed to create ONNX parser");

        // 解析只是在内存中描述网络，此时尚未生成 engine。
        // parseFromFile 成功返回 true，失败返回 false。
        if (!parser->parseFromFile(onnxPath.c_str(),
                                  static_cast<int>(nvinfer1::ILogger::Severity::kWARNING)))
        {
            throw std::runtime_error("Failed to parse ONNX file: " + onnxPath);
        }

        // 3. 配置构建参数，控制可用的计算精度和工作区上限。
        TrtPtr<nvinfer1::IBuilderConfig> config{builder->createBuilderConfig()};
        if (!config)
            throw std::runtime_error("Failed to create builder config");

        // 工作区上限为 1 GiB，影响可选择的计算实现，不代表引擎文件大小。
        config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1ULL << 30);
        // 允许选择 FP16 实现，不强制所有层都使用 FP16。
        // 此处使用默认的弱类型网络；kFP16 在较新的 10.x 中已标记为弃用。
        config->setFlag(nvinfer1::BuilderFlag::kFP16);

        // 4. 真正构建引擎：优化网络并选择计算实现，再返回序列化的二进制数据。
        // 这是耗时步骤，需要可用的 NVIDIA GPU；返回的数据尚未写入磁盘。
        TrtPtr<nvinfer1::IHostMemory> serialized{
            builder->buildSerializedNetwork(*network, *config)};
        if (!serialized)
            throw std::runtime_error("TensorRT failed to build the engine");

        // 5. 将序列化数据按二进制写入文件，这才得到磁盘上的 .engine。
        // 推理时使用 IRuntime::deserializeCudaEngine() 加载这些数据。
        std::ofstream output(enginePath, std::ios::binary);
        if (!output)
            throw std::runtime_error("Cannot open engine output file: " + enginePath);
        output.write(static_cast<const char*>(serialized->data()),
                     static_cast<std::streamsize>(serialized->size()));
        // 显式关闭，确保缓冲区写入和关闭时的错误也能被检查到。
        output.close();
        if (!output)
            throw std::runtime_error("Failed while writing engine file: " + enginePath);

        std::cout << "Engine written to " << enginePath << '\n';
    }

private:
    // logger 必须比引用它的 TensorRT 对象存活更久；构建对象均在 build 内释放。
    Logger logger_;
};

int main(int argc, char* argv[])
{
    // 用法：./build_engine model.onnx model.engine
    if (argc != 3)
    {
        std::cerr << "Usage: " << argv[0] << " <model.onnx> <model.engine>\n";
        return 2;
    }

    try
    {
        EngineBuilder{}.build(argv[1], argv[2]);
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "Engine build failed: " << error.what() << '\n';
        return 1;
    }
}
