#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <cassert>

namespace cudabasic
{
	template<typename T>
	class span
	{
	private:
		T* arr;
		uint32_t length;

	public:
		__device__ __host__ span(T* arrPtr, uint32_t length)
		{
			assert(arrPtr != nullptr);
			this->arr = arrPtr;
			this->length = length;
		}
		__device__ __host__ span()
		{
			this->arr = nullptr;
			this->length = 0;
		}

		__device__ __host__ T& operator[](const uint32_t i)
		{
			assert(i < length);
			return arr[i];
		}

		__device__ __host__ uint32_t size() const
		{
			return length;
		}

		__device__ __host__ span<T> slice(uint32_t offset, uint32_t length) const
		{
			assert(offset + length <= this->length);
			return span<T>(arr + offset, length);
		}

		__device__ __host__ T* begin() const
		{
			return arr;
		}

		__device__ __host__ T* end() const
		{
			return arr + length;
		}
	};
}