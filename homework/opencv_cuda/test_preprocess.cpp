#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaarithm.hpp>

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
const cv::Scalar kLetterboxColor(114, 114, 114);
constexpr int kTargetWidth = 640;
constexpr int kTargetHeight = 640;

template <typename Function>
double averageMilliseconds(Function&& function, int iterations) {
    const auto start = Clock::now();
    for (int i = 0; i < iterations; ++i) {
        function();
    }
    const auto end = Clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count() / iterations;
}

cv::Size letterboxSize(const cv::Size& source, int targetWidth, int targetHeight) {
    if (source.empty() || targetWidth <= 0 || targetHeight <= 0) {
        throw std::invalid_argument("源图像和目标尺寸必须有效");
    }
    const double scale = std::min(
        static_cast<double>(targetWidth) / source.width,
        static_cast<double>(targetHeight) / source.height);
    return cv::Size(
        std::max(1, static_cast<int>(source.width * scale)),
        std::max(1, static_cast<int>(source.height * scale)));
}

cv::Mat letterbox(const cv::Mat& source, int targetWidth, int targetHeight,
                  bool paddingCenter = true) {
    const cv::Size resizedSize = letterboxSize(source.size(), targetWidth, targetHeight);
    cv::Mat resized;
    cv::resize(source, resized, resizedSize, 0.0, 0.0, cv::INTER_LINEAR);

    const int dw = targetWidth - resizedSize.width;
    const int dh = targetHeight - resizedSize.height;
    const int left = paddingCenter ? dw / 2 : 0;
    const int top = paddingCenter ? dh / 2 : 0;
    const int right = dw - left;
    const int bottom = dh - top;

    cv::Mat output;
    cv::copyMakeBorder(resized, output, top, bottom, left, right,
                       cv::BORDER_CONSTANT, kLetterboxColor);
    return output;
}

cv::cuda::GpuMat letterboxGpu(const cv::cuda::GpuMat& source,
                              int targetWidth, int targetHeight,
                              bool paddingCenter = true) {
    const cv::Size resizedSize = letterboxSize(source.size(), targetWidth, targetHeight);
    cv::cuda::GpuMat resized;
    cv::cuda::resize(source, resized, resizedSize, 0.0, 0.0, cv::INTER_LINEAR);

    const int dw = targetWidth - resizedSize.width;
    const int dh = targetHeight - resizedSize.height;
    const int left = paddingCenter ? dw / 2 : 0;
    const int top = paddingCenter ? dh / 2 : 0;
    const int right = dw - left;
    const int bottom = dh - top;

    cv::cuda::GpuMat output;
    cv::cuda::copyMakeBorder(resized, output, top, bottom, left, right,
                             cv::BORDER_CONSTANT, kLetterboxColor);
    return output;
}

cv::Mat normalizeChw(const cv::Mat& rgbImage) {
    cv::Mat floatImage;
    rgbImage.convertTo(floatImage, CV_32F, 1.0 / 255.0);

    std::vector<cv::Mat> planes;
    cv::split(floatImage, planes);
    cv::Mat chw(3 * rgbImage.rows, rgbImage.cols, CV_32FC1);
    for (int channel = 0; channel < 3; ++channel) {
        planes[channel].copyTo(chw.rowRange(channel * rgbImage.rows,
                                           (channel + 1) * rgbImage.rows));
    }
    return chw;
}

cv::cuda::GpuMat normalizeChwGpu(const cv::cuda::GpuMat& rgbImage) {
    cv::cuda::GpuMat floatImage;
    rgbImage.convertTo(floatImage, CV_32F, 1.0 / 255.0);

    std::vector<cv::cuda::GpuMat> planes;
    cv::cuda::split(floatImage, planes);
    cv::cuda::GpuMat chw(3 * rgbImage.rows, rgbImage.cols, CV_32FC1);
    for (int channel = 0; channel < 3; ++channel) {
        planes[channel].copyTo(chw.rowRange(channel * rgbImage.rows,
                                          (channel + 1) * rgbImage.rows));
    }
    return chw;
}

cv::Mat preprocessCpu(const cv::Mat& source, int targetWidth, int targetHeight) {
    cv::Mat boxed = letterbox(source, targetWidth, targetHeight);
    cv::Mat rgb;
    cv::cvtColor(boxed, rgb, cv::COLOR_BGR2RGB);
    return normalizeChw(rgb);
}

cv::Mat preprocessGpu(const cv::Mat& source, int targetWidth, int targetHeight) {
    cv::cuda::GpuMat gpuSource, gpuBoxed, gpuRgb;
    gpuSource.upload(source);
    gpuBoxed = letterboxGpu(gpuSource, targetWidth, targetHeight);
    cv::cuda::cvtColor(gpuBoxed, gpuRgb, cv::COLOR_BGR2RGB);
    cv::cuda::GpuMat gpuChw = normalizeChwGpu(gpuRgb);

    // Include device-to-host transfer in the end-to-end GPU preprocessing time.
    cv::Mat chw;
    gpuChw.download(chw);
    return chw;
}

cv::Mat makeInputImage() {
    cv::Mat image(640, 480, CV_8UC3);
    for (int y = 0; y < image.rows; ++y) {
        for (int x = 0; x < image.cols; ++x) {
            image.at<cv::Vec3b>(y, x) = cv::Vec3b(
                static_cast<uchar>((x * 3) % 256),
                static_cast<uchar>((y * 2) % 256),
                static_cast<uchar>((x + y) % 256));
        }
    }
    return image;
}

}  // namespace

int main() {
    const int deviceCount = cv::cuda::getCudaEnabledDeviceCount();
    if (deviceCount <= 0) {
        std::cerr << "没有可用的 CUDA 设备，无法运行 GPU 预处理。\n";
        return 1;
    }
    cv::cuda::setDevice(0);

    const cv::Mat input = makeInputImage();
    constexpr int warmupIterations = 5;
    constexpr int benchmarkIterations = 50;

    // Warm up both implementations before collecting timing samples.
    for (int i = 0; i < warmupIterations; ++i) {
        preprocessCpu(input, kTargetWidth, kTargetHeight);
        preprocessGpu(input, kTargetWidth, kTargetHeight);
    }

    const cv::Mat cpuOutput = preprocessCpu(input, kTargetWidth, kTargetHeight);
    const cv::Mat gpuOutput = preprocessGpu(input, kTargetWidth, kTargetHeight);
    if (cpuOutput.size() != gpuOutput.size() || cpuOutput.type() != gpuOutput.type()) {
        std::cerr << "CPU/GPU 输出尺寸或类型不一致。\n";
        return 2;
    }
    const double maxDifference = cv::norm(cpuOutput, gpuOutput, cv::NORM_INF);

    const double cpuMs = averageMilliseconds([&] {
        (void)preprocessCpu(input, kTargetWidth, kTargetHeight);
    }, benchmarkIterations);
    const double gpuMs = averageMilliseconds([&] {
        (void)preprocessGpu(input, kTargetWidth, kTargetHeight);
    }, benchmarkIterations);

    std::cout << std::fixed << std::setprecision(4)
              << "输入尺寸: " << input.cols << "x" << input.rows << '\n'
              << "输出 CHW 尺寸: 3x" << kTargetHeight << "x" << kTargetWidth << '\n'
              << "CPU/GPU 输出最大绝对误差: " << maxDifference << '\n'
              << "基准迭代: " << benchmarkIterations << " 次，每次平均耗时\n"
              << "GPU 耗时包含输入上传、GPU 预处理和输出下载。\n"
              << "CPU: " << cpuMs << " ms\n"
              << "GPU: " << gpuMs << " ms\n"
              << "CPU/GPU 耗时比: " << cpuMs / gpuMs << "x\n";
    return 0;
}
