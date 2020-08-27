#pragma once

#include <cuda_runtime.h>
#include <memory>
#include "cpuGpuObject.h"

namespace cudabasic
{
	template<typename ObjectType, typename DataType>
	class cpuGpuTransporter
	{
	private:
		std::shared_ptr<cpuGpuObject<ObjectType, DataType>> cpuObject;
		float* gpuPtr;
		int gpuPtrLengthInBytes;

	public:
		cpuGpuTransporter(std::shared_ptr<cpuGpuObject<ObjectType, DataType>> cpuObject)
		{
			if (cpuObject->getMemoryLocation() == memPlacmenet::GPU)
			{
				throw std::runtime_error("Object was on the GPU but it has to be on the CPU.");
			}

			this->cpuObject = cpuObject;

			this->gpuPtrLengthInBytes = cpuObject->getMemOnGPU() * sizeof(DataType);
			const cudaError_t status = cudaMalloc(&this->gpuPtr, gpuPtrLengthInBytes);
			if (status != cudaError::cudaSuccess)
			{
				throw std::runtime_error("Failed to allocate cuda memory.");
			}
		}

		ObjectType getGPUObject()
		{
			return cpuObject->getAsGPUObject(gpuPtr);
		}

		void copyToGPU()
		{
			const cudaError_t status = cudaMemcpy(gpuPtr, cpuObject->getCPUPtr(), gpuPtrLengthInBytes, cudaMemcpyKind::cudaMemcpyHostToDevice);
			if (status != cudaError::cudaSuccess)
			{
				throw std::runtime_error("Failed to copy from host to device.");
			}
		}

		void copyFromGPU()
		{
			const cudaError_t status = cudaMemcpy(cpuObject->getCPUPtr(), gpuPtr, gpuPtrLengthInBytes, cudaMemcpyKind::cudaMemcpyDeviceToHost);
			if (status != cudaError::cudaSuccess)
			{
				throw std::runtime_error("Failed to copy from device from host.");
			}
		}

		~cpuGpuTransporter()
		{
			const cudaError_t status = cudaFree(gpuPtr);
			if (status != cudaError::cudaSuccess)
			{
				throw std::runtime_error("Failed to deallocate cuda memory.");
			}
		}
	};
}