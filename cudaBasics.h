#pragma once

#include "cuda_runtime.h"
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

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

}