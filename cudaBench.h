#pragma once

#include <cuda_runtime.h>

namespace cudabasic
{

	/// <summary>
	/// Benchmarks a cuda kernel by taking the average execution time over a set amount of kernel executions
	/// </summary>
	/// <typeparam name="...Args">Kernel argument types</typeparam>
	template<typename... Args>
	class cudaBench
	{
	private:
		cudabasic::cudaTimer timer;
		void(*kernel)(Args...);
		int benchCount;

	public:
		/// <summary>
		/// Make a benchmark for a specific kernel by running it a set amount of times
		/// </summary>
		/// <param name="cudaKernel">Cuda kernel to benchmark</param>
		/// <param name="kernelExecutionCount">How many times to execute the kernel when benchmarking</param>
		cudaBench(void(*cudaKernel)(Args...), uint32_t kernelExecutionCount)
		{
			kernel = cudaKernel;
			benchCount = kernelExecutionCount;
		}

		/// <summary>
		/// Benchmarks with the specified kernel configuration
		/// </summary>
		/// <param name="blockDim">Block dimension for the kernel</param>
		/// <param name="gridDim">Grid dimension for the kernel</param>
		/// <param name="...args">Kernel arguments</param>
		/// <returns>Average runtime of the kernel</returns>
		float benchmark(dim3 blockDim, dim3 gridDim, Args... args)
		{
			return benchmark(blockDim, gridDim, 0, args...);
		}

		/// <summary>
		/// Benchmarks with the specified kernel configuration
		/// </summary>
		/// <param name="blockDim">Block dimension for the kernel</param>
		/// <param name="gridDim">Grid dimension for the kernel</param>
		/// <param name="sharedMemSize">Shared memory in bytes</param>
		/// <param name="...args">Kernel arguments</param>
		/// <returns>Average runtime of the kernel</returns>
		float benchmark(dim3 blockDim, dim3 gridDim, int sharedMemSize, Args... args)
		{
			return benchmark(blockDim, gridDim, sharedMemSize, 0, args...);
		}

		/// <summary>
		/// Benchmarks with the specified kernel configuration
		/// </summary>
		/// <param name="blockDim">Block dimension for the kernel</param>
		/// <param name="gridDim">Grid dimension for the kernel</param>
		/// <param name="sharedMemSize">Shared memory in bytes</param>
		/// <param name="stream">Cuda stream to launch the kernel on</param>
		/// <param name="...args">Kernel arguments</param>
		/// <returns>Average runtime of the kernel</returns>
		float benchmark(dim3 blockDim, dim3 gridDim, int sharedMemSize, cudaStream_t stream, Args... args)
		{
			float time = 0.0f;
			for (size_t i = 0; i < benchCount; i++)
			{
				timer.startTimer(stream);
				executeKernel(kernel, blockDim, gridDim, sharedMemSize, stream, args...);
				timer.stopTimer(stream);
				time += timer.getElapsedMiliseconds();
			}
			// << <gridDim, blockDim >> > 
			return time / benchCount;
		}
	};
}