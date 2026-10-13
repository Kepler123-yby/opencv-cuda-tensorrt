#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudafilters.hpp>
#include <cuda_runtime_api.h>

#include <chrono>
#include <iomanip>
#include <iostream>

namespace {

using Clock = std::chrono::steady_clock;
constexpr int kWarmupIterations = 10;
constexpr int kBenchmarkIterations = 100;

template <typename Function>
double averageMilliseconds(Function&& function, int iterations) {
    const auto start = Clock::now();
    for (int i = 0; i < iterations; ++i) {
        function();
    }
    const auto end = Clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count() / iterations;
}

cv::Mat generator_image(int width = 640, int height = 480) {
    cv::Mat image(height, width, CV_8UC3, cv::Scalar(0, 0, 0));
    image(cv::Rect(100, 100, 200, 200)) = cv::Scalar(255, 0, 0);
    return image;
}

}  // namespace

int main() {
    const int deviceCount = cv::cuda::getCudaEnabledDeviceCount();
    if (deviceCount <= 0) {
        std::cerr << "没有可用的 CUDA 设备，无法比较 GPU 耗时。\n";
        return 1;
    }

    cv::cuda::setDevice(0);
    cv::cuda::printCudaDeviceInfo(0);

    const cv::Mat image = generator_image();
    cv::Mat cpuGray, cpuBlur;
    cv::cuda::GpuMat gpuSrc, gpuGray, gpuBlur;
    cv::Mat grayOutput, blurOutput;

    auto gaussianFilter = cv::cuda::createGaussianFilter(
        CV_8UC3, CV_8UC3, cv::Size(5, 5), 1.5);

    // 预热，避免把首次调用的初始化开销算进测量结果。
    for (int i = 0; i < kWarmupIterations; ++i) {
        cv::cvtColor(image, cpuGray, cv::COLOR_BGR2GRAY);
        cv::GaussianBlur(image, cpuBlur, cv::Size(5, 5), 1.5);
        gpuSrc.upload(image);
        cv::cuda::cvtColor(gpuSrc, gpuGray, cv::COLOR_BGR2GRAY);
        gpuGray.download(grayOutput);
        gpuSrc.upload(image);
        gaussianFilter->apply(gpuSrc, gpuBlur);
        gpuBlur.download(blurOutput);
    }
    cudaDeviceSynchronize();

    const double cpuColorMs = averageMilliseconds([&] {
        cv::cvtColor(image, cpuGray, cv::COLOR_BGR2GRAY);
    }, kBenchmarkIterations);
    const double cpuGaussianMs = averageMilliseconds([&] {
        cv::GaussianBlur(image, cpuBlur, cv::Size(5, 5), 1.5);
    }, kBenchmarkIterations);

    // Each GPU iteration measures the complete CPU->GPU->CPU path.
    const double gpuColorMs = averageMilliseconds([&] {
        gpuSrc.upload(image);
        cv::cuda::cvtColor(gpuSrc, gpuGray, cv::COLOR_BGR2GRAY);
        gpuGray.download(grayOutput);
    }, kBenchmarkIterations);
    const double gpuGaussianMs = averageMilliseconds([&] {
        gpuSrc.upload(image);
        gaussianFilter->apply(gpuSrc, gpuBlur);
        gpuBlur.download(blurOutput);
    }, kBenchmarkIterations);

    std::cout << std::fixed << std::setprecision(4)
              << "图像尺寸: " << image.cols << "x" << image.rows
              << "，基准迭代: " << kBenchmarkIterations << " 次（显示每次平均耗时）\n"
              << "说明: GPU 耗时包含输入上传、CUDA 运算和结果下载。\n\n"
              << "处理操作             CPU (ms)       GPU (ms)       CPU/GPU\n"
              << "BGR 转灰度           " << cpuColorMs << "       " << gpuColorMs
              << "       " << cpuColorMs / gpuColorMs << "x\n"
              << "高斯滤波 5x5         " << cpuGaussianMs << "       " << gpuGaussianMs
              << "       " << cpuGaussianMs / gpuGaussianMs << "x\n";

    std::cout << "\n结果检查: 灰度尺寸 " << grayOutput.cols << "x" << grayOutput.rows
              << "，模糊图均值 " << cv::mean(blurOutput)[0] << "\n";
    return 0;
}
