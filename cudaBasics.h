#pragma once

#include "cuda_runtime.h"
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>
#include <functional>

namespace cudabasic
{

    template<typename T>
    class cpuGpuArray
    {
    private:
        int32_t arrLength;
        T* gpuArray;
        std::vector<T>* cpuArray;

    public:
        cpuGpuArray(const int32_t length)
        {
            this->arrLength = length;
            const cudaError_t status = cudaMalloc(&this->gpuArray, length * sizeof(T));
            if (status != cudaError::cudaSuccess)
            {
                throw std::runtime_error("Failed to allocate cuda memory.");
            }

            this->cpuArray = new std::vector<T>(length);
        }

        void copyToGPU()
        {
            const cudaError_t status = cudaMemcpy(gpuArray, &(*cpuArray)[0], arrLength * sizeof(T), cudaMemcpyKind::cudaMemcpyHostToDevice);
            if (status != cudaError::cudaSuccess)
            {
                throw std::runtime_error("Failed to copy from host to device.");
            }
        }

        std::vector<T>& copyFromGPU()
        {
            const cudaError_t status = cudaMemcpy(&(*cpuArray)[0], gpuArray, arrLength * sizeof(T), cudaMemcpyKind::cudaMemcpyDeviceToHost);
            if (status != cudaError::cudaSuccess)
            {
                throw std::runtime_error("Failed to copy from device from host.");
            }

            return getCPUArray();
        }

        T* getGPUArray()
        {
            return gpuArray;
        }

        std::vector<T>& getCPUArray()
        {
            return *cpuArray;
        }

        ~cpuGpuArray()
        {
            //deallocate gpu array
            const cudaError_t status = cudaFree(gpuArray);
            if (status != cudaError::cudaSuccess)
            {
                throw std::runtime_error("Failed to deallocate cuda memory.");
            }

            //deallocate cpu array
            delete cpuArray;
        }
    };

    void setCudaDevice(const int32_t device);
    void resetCudaDevice();
    void checkForCudaError();
    void cudaSynchronize();


    class cudaTimer
    {
    private:
        cudaEvent_t startTime;
        cudaEvent_t endTime;

    public:
        cudaTimer();
        ~cudaTimer();

        void startTimer();
        void stopTimer();
        float getElapsedMiliseconds();
    };

    /// <summary>
    /// Executes a cuda kernel
    /// </summary>
    /// <typeparam name="...Args"></typeparam>
    /// <param name="kernel">Function pointer to the kernel</param>
    /// <param name="blockDim">Block dimension for the kernel</param>
    /// <param name="gridDim">Grid dimension for the kernel</param>
    /// <param name="...args">Kernel arguments</param>
    template<typename... Args>
    void executeKernel(void(*kernel)(Args...), dim3 blockDim, dim3 gridDim, Args... args)
    {
        (*kernel) << <gridDim, blockDim >> > (args...);
    }

    /// <summary>
    /// Benchmarks a cuda kernel by taking the average execution time over a set amount of kernel executions
    /// </summary>
    /// <typeparam name="...Args">Kernel argument types</typeparam>
    template<typename... Args>
    class cudaBench
    {
    private:
        cudabasic::cudaTimer timer;
        void(*kernel)(Args...);
        int benchCount;

    public:
        /// <summary>
        /// Make a benchmark for a specific kernel by running it a set amount of times
        /// </summary>
        /// <param name="cudaKernel">Cuda kernel to benchmark</param>
        /// <param name="kernelExecutionCount">How many times to execute the kernel when benchmarking</param>
        cudaBench(void(*cudaKernel)(Args...), uint32_t kernelExecutionCount)
        {
            kernel = cudaKernel;
            benchCount = kernelExecutionCount;
        }

        /// <summary>
        /// Benchmarks with the specified grid and kernel dimensions and kernel arguments
        /// </summary>
        /// <param name="blockDim">Size of a single block on threads</param>
        /// <param name="gridDim">Size of the grid in blocks</param>
        /// <param name="...args">Kernel arguments</param>
        /// <returns>The average execution time</returns>
        float benchmark(dim3 blockDim, dim3 gridDim, Args... args)
        {
            float time = 0.0f;
            for (size_t i = 0; i < benchCount; i++)
            {
                timer.startTimer();
                executeKernel(kernel, blockDim, gridDim, args...);
                timer.stopTimer();
                time += timer.getElapsedMiliseconds();
            }
            // << <gridDim, blockDim >> > 
            return time / benchCount;
        }
    };
}