#include <NvInfer.h>
#include <NvOnnxParser.h>

#include <iostream>

class Logger : public nvinfer1::ILogger
{
public:
    void log(Severity severity, const char* msg) noexcept override
    {
        if (severity <= Severity::kWARNING)
            std::cout << "[TRT]" << msg << "/n";
    }

};