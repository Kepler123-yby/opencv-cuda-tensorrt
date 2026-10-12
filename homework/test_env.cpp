#include <opencv2/core/cuda.hpp>
#include <iostream>

int main() {
    int count = cv::cuda::getCudaEnabledDeviceCount();
    std::cout << "Number of CUDA-enabled devices: " << count << std::endl;

    if (count > 0) {
        for (int i = 0; i < count; ++i) {
            cv::cuda::DeviceInfo deviceInfo(i);
            std::cout << "Device " << i << ": " << deviceInfo.name() << std::endl;
        }
    } else {
        std::cout << "No CUDA-enabled devices found." << std::endl;
    }
    
    return 0;
}