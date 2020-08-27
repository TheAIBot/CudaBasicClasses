#pragma once

#include <cuda_runtime.h>

namespace cudabasic
{
	enum class memPlacmenet
	{
		CPU,
		GPU
	};

	template<typename ObjectType, typename DataType>
	class cpuGpuObject
	{
	protected:
		memPlacmenet place;

		__device__ __host__ cpuGpuObject(memPlacmenet placce)
		{
			this->place = placce;
		}

	public:
		virtual int getMemOnGPU() = 0;
		virtual ObjectType getAsGPUObject(DataType* gpuPtr) = 0;
		virtual DataType* getCPUPtr() = 0;
		memPlacmenet getMemoryLocation()
		{
			return place;
		}
	};
}