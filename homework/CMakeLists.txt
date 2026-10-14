cmake_minimum_required(VERSION 3.24)

# Prefer the CUDA and TensorRT installations used by this workspace, while
# allowing either location to be overridden from the environment or command line.
if(NOT DEFINED CUDAToolkit_ROOT)
    if(DEFINED ENV{CUDAToolkit_ROOT})
        set(CUDAToolkit_ROOT "$ENV{CUDAToolkit_ROOT}" CACHE PATH "CUDA Toolkit root")
    elseif(DEFINED ENV{CUDA_HOME})
        set(CUDAToolkit_ROOT "$ENV{CUDA_HOME}" CACHE PATH "CUDA Toolkit root")
    elseif(EXISTS "/home/kepler/cuda-11.8/bin/nvcc")
        set(CUDAToolkit_ROOT "/home/kepler/cuda-11.8" CACHE PATH "CUDA Toolkit root")
    endif()
endif()

if(NOT DEFINED CMAKE_CUDA_COMPILER AND CUDAToolkit_ROOT AND
   EXISTS "${CUDAToolkit_ROOT}/bin/nvcc")
    set(CMAKE_CUDA_COMPILER "${CUDAToolkit_ROOT}/bin/nvcc" CACHE FILEPATH "CUDA compiler")
endif()

project(opencv_cuda_tensorrt_homework LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(CUDAToolkit REQUIRED)

if(NOT DEFINED OpenCV_DIR)
    if(DEFINED ENV{OpenCV_DIR})
        set(OpenCV_DIR "$ENV{OpenCV_DIR}" CACHE PATH "OpenCV package directory")
    elseif(DEFINED ENV{OPENCV_ROOT})
        set(OpenCV_DIR "$ENV{OPENCV_ROOT}/lib/cmake/opencv4" CACHE PATH "OpenCV package directory")
    elseif(EXISTS "/home/kepler/opt/opencv-cuda/lib/cmake/opencv4/OpenCVConfig.cmake")
        set(OpenCV_DIR "/home/kepler/opt/opencv-cuda/lib/cmake/opencv4" CACHE PATH "OpenCV package directory")
    endif()
endif()
find_package(OpenCV 4 REQUIRED CONFIG COMPONENTS
                core
                imgproc
                imgcodecs
                cudaarithm
                cudaimgproc
                cudawarping
                cudafilters
            )

# CMAKE_PREFIX_PATH from ~/.bashrc is honored by find_path/find_library. The
# local fallback also works when CMake is launched by an editor that skips bashrc.
if(DEFINED ENV{TENSORRT_ROOT})
    set(_tensorrt_root "$ENV{TENSORRT_ROOT}")
elseif(EXISTS "/home/kepler/tensorrt/TensorRT-10.10.0.31/include/NvInfer.h")
    set(_tensorrt_root "/home/kepler/tensorrt/TensorRT-10.10.0.31")
else()
    set(_tensorrt_root "")
endif()

find_path(TensorRT_INCLUDE_DIR NvInfer.h
    HINTS "${_tensorrt_root}"
    PATH_SUFFIXES include
    REQUIRED)
find_library(TensorRT_NVINFER_LIBRARY nvinfer
    HINTS "${_tensorrt_root}"
    PATH_SUFFIXES lib lib64
    REQUIRED)
find_library(TensorRT_ONNXPARSER_LIBRARY nvonnxparser
    HINTS "${_tensorrt_root}"
    PATH_SUFFIXES lib lib64
    REQUIRED)

add_library(TensorRT::nvinfer UNKNOWN IMPORTED)
set_target_properties(TensorRT::nvinfer PROPERTIES
    IMPORTED_LOCATION "${TensorRT_NVINFER_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${TensorRT_INCLUDE_DIR}")
add_library(TensorRT::nvonnxparser UNKNOWN IMPORTED)
set_target_properties(TensorRT::nvonnxparser PROPERTIES
    IMPORTED_LOCATION "${TensorRT_ONNXPARSER_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${TensorRT_INCLUDE_DIR}")

add_executable(test_env opencv_cuda/test_env.cpp)
add_executable(test_operator opencv_cuda/test_operator.cpp)
add_executable(test_preprocess opencv_cuda/test_preprocess.cpp)
add_executable(build_engine tensorrt/build_engine.cpp)
add_executable(run_infer tensorrt/run_infer.cpp)
target_link_libraries(test_env PRIVATE
    ${OpenCV_LIBS}
    CUDA::cudart
    TensorRT::nvinfer)
target_link_libraries(test_operator PRIVATE
    ${OpenCV_LIBS}
    CUDA::cudart
    TensorRT::nvinfer)
target_link_libraries(test_preprocess PRIVATE
    ${OpenCV_LIBS}
    CUDA::cudart)
target_link_libraries(build_engine PRIVATE
    ${OpenCV_LIBS}
    CUDA::cudart
    TensorRT::nvinfer
    TensorRT::nvonnxparser)
target_link_libraries(run_infer PRIVATE
    CUDA::cudart
    TensorRT::nvinfer)

get_filename_component(_tensorrt_libdir "${TensorRT_NVINFER_LIBRARY}" DIRECTORY)
get_filename_component(_tensorrt_parser_libdir "${TensorRT_ONNXPARSER_LIBRARY}" DIRECTORY)
# Preserve library locations for launching executables from the build directory.
set(_runtime_rpath "${OpenCV_DIR}/../../..;${_tensorrt_libdir};${_tensorrt_parser_libdir};${CUDAToolkit_ROOT}/lib64")
set_target_properties(test_env test_operator test_preprocess build_engine run_infer PROPERTIES
    BUILD_RPATH "${_runtime_rpath}")
