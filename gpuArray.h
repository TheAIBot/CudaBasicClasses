#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <cassert>

namespace cudabasic
{
	template<typename T>
	class gpuArray
	{
	private:
		uint32_t arrLength;
		T* gpuArr;

	public:
		gpuArray(const uint32_t length)
		{
			this->arrLength = length;
			const cudaError_t status = cudaMalloc(&this->gpuArr, length * sizeof(T));
			if (status != cudaError::cudaSuccess)
			{
				throw std::runtime_error("Failed to allocate cuda memory.");
			}
		}

		T* getGPUArray()
		{
			return gpuArr;
		}

		const T* getGPUArrayConst()
		{
			return gpuArr;
		}

		~gpuArray() noexcept(false)
		{
			//deallocate gpu array
			const cudaError_t status = cudaFree(gpuArr);
			if (status != cudaError::cudaSuccess)
			{
				throw std::runtime_error("Failed to deallocate cuda memory.");
			}
		}
	};
}