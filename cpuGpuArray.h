#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <stdexcept>
#include <cassert>
#include "span.h"

namespace cudabasic
{
	enum class cudaPinOpt
	{
		Pinned,
		NotPinned
	};

	template<typename T>
	class cpuGpuArray
	{
	private:
		T* gpuArray;
		span<T> cpuArray;
		cudaPinOpt pinChoise;

	public:
		cpuGpuArray(const uint32_t length) : cpuGpuArray(length, cudaPinOpt::NotPinned)
		{
		}

		cpuGpuArray(const uint32_t length, const cudaPinOpt pinOpt) : pinChoise(pinOpt)
		{
			{
				const cudaError_t status = cudaMalloc(&this->gpuArray, length * sizeof(T));
				if (status != cudaError::cudaSuccess)
				{
					throw std::runtime_error("Failed to allocate cuda memory.");
				}
			}

			if (pinOpt == cudaPinOpt::Pinned)
			{
				T* ptr;
				const cudaError_t status = cudaMallocHost(&ptr, length * sizeof(T));
				if (status != cudaError::cudaSuccess)
				{
					throw std::runtime_error("Failed to allocate pinned cuda memory.");
				}

				cpuArray = span<T>(ptr, length);
			}
			else if (pinOpt == cudaPinOpt::NotPinned)
			{
				cpuArray = span<T>(new T[length], length);
			}
			else
			{
				throw std::runtime_error("Invalid pin option.");
			}
		}

		void copyToGPU()
		{
			copyToGPU(0, cpuArray.size());
		}
		void copyToGPU(const uint32_t offset, const uint32_t elementCount)
		{
			assert(offset + elementCount <= cpuArray.size());

			const cudaError_t status = cudaMemcpy(gpuArray + offset, cpuArray.begin() + offset, elementCount * sizeof(T), cudaMemcpyKind::cudaMemcpyHostToDevice);
			if (status != cudaError::cudaSuccess)
			{
				throw std::runtime_error("Failed to copy from host to device.");
			}
		}

		void copyToGPUAsync(const cudaStream& stream)
		{
			copyToGPUAsync(0, cpuArray.size(), stream);
		}
		void copyToGPUAsync(const uint32_t offset, const uint32_t elementCount, const cudaStream& stream)
		{
			assert(pinChoise == cudaPinOpt::Pinned);
			assert(offset + elementCount <= cpuArray.size());

			const cudaError_t status = cudaMemcpyAsync(gpuArray + offset, cpuArray.begin() + offset, elementCount * sizeof(T), cudaMemcpyKind::cudaMemcpyHostToDevice, stream);
			if (status != cudaError::cudaSuccess)
			{
				throw std::runtime_error("Failed to copy from host to device.");
			}
		}

		span<T> copyFromGPU()
		{
			return copyFromGPU(0, cpuArray.size());
		}
		span<T> copyFromGPU(const uint32_t offset, const uint32_t elementCount)
		{
			assert(offset + elementCount <= cpuArray.size());

			const cudaError_t status = cudaMemcpy(cpuArray.begin() + offset, gpuArray + offset, elementCount * sizeof(T), cudaMemcpyKind::cudaMemcpyDeviceToHost);
			if (status != cudaError::cudaSuccess)
			{
				throw std::runtime_error("Failed to copy from device from host.");
			}

			return getCPUArray();
		}

		void copyFromGPUAsync(const cudaStream& stream)
		{
			copyFromGPU(0, cpuArray.size());
		}
		void copyFromGPUAsync(const uint32_t offset, const uint32_t elementCount, const cudaStream& stream)
		{
			assert(pinChoise == cudaPinOpt::Pinned);
			assert(offset + elementCount <= cpuArray.size());

			const cudaError_t status = cudaMemcpyAsync(cpuArray.begin() + offset, gpuArray + offset, elementCount * sizeof(T), cudaMemcpyKind::cudaMemcpyDeviceToHost, stream);
			if (status != cudaError::cudaSuccess)
			{
				throw std::runtime_error("Failed to copy from device from host.");
			}
		}

		T* getGPUArray() const
		{
			return gpuArray;
		}
		const T* getGPUArrayConst() const
		{
			return gpuArray;
		}

		span<T> getCPUArray() const
		{
			return cpuArray;
		}

		int32_t size() const
		{
			return cpuArray.size();
		}

		~cpuGpuArray() noexcept(false)
		{
			//deallocate gpu array
			const cudaError_t status = cudaFree(gpuArray);
			if (status != cudaError::cudaSuccess)
			{
				throw std::runtime_error("Failed to deallocate cuda memory.");
			}

			//deallocate cpu array
			if (pinChoise == cudaPinOpt::Pinned)
			{
				const cudaError_t status = cudaFreeHost(cpuArray.begin());
				if (status != cudaError::cudaSuccess)
				{
					throw std::runtime_error("Failed to deallocate pinned cuda memory.");
				}
			}
			else if (pinChoise == cudaPinOpt::NotPinned)
			{
				delete[] cpuArray.begin();
			}
			else
			{
				throw std::runtime_error("Invalid pin option.");
			}
		}
	};
}