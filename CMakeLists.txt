# CMakeList.txt : CMake project for ISATestsGen, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.19)

set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
set(CMAKE_CUDA_ARCHITECTURES "35")

project(CudaBasics LANGUAGES CXX CUDA)

enable_language(CUDA)

find_package(CUDA  REQUIRED)
include_directories("${CUDA_INCLUDE_DIRS}")

# Add source to this project's executable.
add_library(cuda_basics "cpuGpuArray.h" "cpuGpuObject.h" "cpuGpuTransporter.h" "cudaBasics.cpp" "cudaBasics.h" "cudaBench.h" "cudaStream.cpp" "cudaStream.h"  "cudaTimer.cpp" "cudaTimer.h" "gpuArray.h" "matrix.h" "span.h")

target_compile_features(cuda_basics PUBLIC cxx_std_17)
set_target_properties(cuda_basics PROPERTIES CUDA_SEPARABLE_COMPILATION ON)