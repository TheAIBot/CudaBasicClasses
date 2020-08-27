#include <stdexcept>
#include "cudaTimer.h"

namespace cudabasic
{
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
    cudaTimer::~cudaTimer() noexcept(false)
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
        startTimer(0);
    }
    void cudaTimer::startTimer(cudaStream_t stream)
    {
        const cudaError_t cudaStatus = cudaEventRecord(startTime, stream);
        if (cudaStatus != cudaError::cudaSuccess)
        {
            throw std::runtime_error("Cuda error code: " + cudaStatus);
        }
    }

    void cudaTimer::stopTimer()
    {
        stopTimer(0);
    }
    void cudaTimer::stopTimer(cudaStream_t stream)
    {
        cudaError_t cudaStatus = cudaEventRecord(endTime, stream);
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