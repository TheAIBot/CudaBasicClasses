#include "cudaBasics.h"
namespace cudabasic 
{


    void setCudaDevice(const int32_t device)
    {
        if (cudaSetDevice(device) != cudaError::cudaSuccess)
        {
            throw std::runtime_error("cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        }
    }

    void resetCudaDevice()
    {
        const cudaError_t cudaStatus = cudaDeviceReset();
        if (cudaStatus != cudaError::cudaSuccess)
        {
            throw std::runtime_error("cudaDeviceReset failed!");
        }
    }

    void checkForCudaError()
    {
        const cudaError_t cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaError::cudaSuccess)
        {
            const std::string errorString(cudaGetErrorString(cudaStatus));
            throw std::runtime_error("Cuda error: " + errorString);
        }
    }

    void cudaSynchronize()
    {
        const cudaError_t cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaError::cudaSuccess)
        {
            throw std::runtime_error("Cuda error code: " + cudaStatus);
        }
    }

    cudaTimer::cudaTimer()
    {
        cudaError_t cudaStatus = cudaEventCreate(&startTime);
        if (cudaStatus != cudaError::cudaSuccess)
        {
            throw std::runtime_error("Cuda error code: " + cudaStatus);
        }

        cudaStatus = cudaEventCreate(&endTime);
        if (cudaStatus != cudaError::cudaSuccess)
        {
            throw std::runtime_error("Cuda error code: " + cudaStatus);
        }
    }
    cudaTimer::~cudaTimer()
    {
        cudaError_t cudaStatus = cudaEventDestroy(startTime);
        if (cudaStatus != cudaError::cudaSuccess)
        {
            throw std::runtime_error("Cuda error code: " + cudaStatus);
        }

        cudaStatus = cudaEventDestroy(endTime);
        if (cudaStatus != cudaError::cudaSuccess)
        {
            throw std::runtime_error("Cuda error code: " + cudaStatus);
        }
    }

    void cudaTimer::startTimer()
    {
        const cudaError_t cudaStatus = cudaEventRecord(startTime);
    }
    void cudaTimer::stopTimer()
    {
        cudaError_t cudaStatus = cudaEventRecord(endTime);
        if (cudaStatus != cudaError::cudaSuccess)
        {
            throw std::runtime_error("Cuda error code: " + cudaStatus);
        }

        cudaStatus = cudaEventSynchronize(endTime);
        if (cudaStatus != cudaError::cudaSuccess)
        {
            throw std::runtime_error("Cuda error code: " + cudaStatus);
        }
    }
    float cudaTimer::getElapsedMiliseconds()
    {
        float time;
        const cudaError_t cudaStatus = cudaEventElapsedTime(&time, startTime, endTime);
        if (cudaStatus != cudaError::cudaSuccess)
        {
            throw std::runtime_error("Cuda error code: " + cudaStatus);
        }

        return time;
    }
}