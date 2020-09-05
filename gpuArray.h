#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <cassert>
#include <stdexcept>
#include "span.h"

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
		~gpuArray() noexcept(false)
		{
			//deallocate gpu array
			const cudaError_t status = cudaFree(gpuArr);
			if (status != cudaError::cudaSuccess)
			{
				throw std::runtime_error("Failed to deallocate cuda memory.");
			}
		}

		void copyToGPU(std::vector<T>& src)
		{
			copyToGPU(src, 0, (uint32_t)src.size(), 0);
		}
		void copyToGPU(std::vector<T>& src, const uint32_t srcOffset, const uint32_t srcLength, const uint32_t dstOffset)
		{
			assert(srcOffset + srcLength <= src.size());
			assert(dstOffset + srcLength <= arrLength);

			const cudaError_t status = cudaMemcpy(gpuArr + dstOffset, &src[0] + srcOffset, srcLength * sizeof(T), cudaMemcpyKind::cudaMemcpyHostToDevice);
			if (status != cudaError::cudaSuccess)
			{
				throw std::runtime_error("Failed to copy from host to device.");
			}
		}

		std::vector<T> copyToCPU() const
		{
			return copyToCPU(0, arrLength);
		}
		std::vector<T> copyToCPU(const uint32_t copySrcOffset, const uint32_t copyLength) const
		{
			std::vector<T> values(copyLength);

			const cudaError_t status = cudaMemcpy(&values[0], gpuArr + copySrcOffset, copyLength * sizeof(T), cudaMemcpyKind::cudaMemcpyDeviceToHost);
			if (status != cudaError::cudaSuccess)
			{
				throw std::exception("Failed to copy from device from host.");
			}

			return values;
		}

		span<T> getGPUArray() const
		{
			return span<T>(gpuArr, arrLength);
		}

		const span<T> getGPUArrayConst() const
		{
			return span<T>(gpuArr, arrLength);
		}

		uint32_t size() const
		{
			return arrLength;
		}
	};
}