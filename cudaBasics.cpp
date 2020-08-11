#include "cudaBasics.h"
namespace cudabasic {
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
}