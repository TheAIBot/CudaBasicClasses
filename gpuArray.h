#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <cassert>
#include <stdexcept>
#include <iostream>
#include <vector>
#include "span.h"
#include "cudaStream.h"

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
			if constexpr (std::is_same<T, bool>::value)
			{
				throw std::runtime_error("Currently not handling copying bool to the GPU.");
			}
			else
			{
				assert(srcOffset + srcLength <= src.size());
				assert(dstOffset + srcLength <= arrLength);

				void* dstPtr = reinterpret_cast<void*>(gpuArr + dstOffset);
				const void* srcPtr = reinterpret_cast<void*>(src.data() + srcOffset);
				const cudaError_t status = cudaMemcpy(dstPtr, srcPtr, srcLength * sizeof(T), cudaMemcpyKind::cudaMemcpyHostToDevice);
				if (status != cudaError::cudaSuccess)
				{
					throw std::runtime_error("Failed to copy from host to device.");
				}	
			}
		}

		std::vector<T> copyToCPU() const
		{
			return copyToCPU(0, arrLength);
		}
		std::vector<T> copyToCPU(const uint32_t copySrcOffset, const uint32_t copyLength) const
		{
			if constexpr (std::is_same<T, bool>::value)
			{
				throw std::runtime_error("Currently not handling copying bool from the GPU.");
			}
			else
			{
				std::vector<T> values(copyLength);

				void* dstPtr = reinterpret_cast<void*>(values.data());
				const void* srcPtr = reinterpret_cast<void*>(gpuArr + copySrcOffset);
				const cudaError_t status = cudaMemcpy(dstPtr, srcPtr, copyLength * sizeof(T), cudaMemcpyKind::cudaMemcpyDeviceToHost);
				if (status != cudaError::cudaSuccess)
				{
					throw std::runtime_error("Failed to copy from device from host.");
				}

				return values;
			}
		}

		void copyToGPUArray(gpuArray<T>& gpuArr) const
		{
			assert(gpuArr.size() == this->size());

			const cudaError_t status = cudaMemcpy(gpuArr.gpuArr, this->gpuArr, this->size() * sizeof(T), cudaMemcpyKind::cudaMemcpyDeviceToDevice);
			if (status != cudaError::cudaSuccess)
			{
				std::cout << "Failed to copy from device to device." << std::endl;
			}
		}

		void copyToGPUArray(gpuArray<T>& gpuArr, const cudaStream_t& stream) const
		{
			assert(gpuArr.size() == this->size());

			const cudaError_t status = cudaMemcpyAsync(gpuArr.gpuArr, this->gpuArr, this->size() * sizeof(T), cudaMemcpyKind::cudaMemcpyDeviceToDevice, stream);
			if (status != cudaError::cudaSuccess)
			{
				std::cout << "Failed to copy from device to device." << std::endl;
			}
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