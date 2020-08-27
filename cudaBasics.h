#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>
#include <functional>
#include <limits>
#include <random>
#include <memory>
#include <cassert>

namespace cudabasic
{
    void setCudaDevice(const int32_t device);
    void resetCudaDevice();
    void checkForCudaError();
    void cudaSynchronize();

	/// <summary>
	/// Executes a cuda kernel
	/// </summary>
	/// <typeparam name="...Args">Kernel argument types</typeparam>
	/// <param name="kernel">Function pointer to the kernel</param>
	/// <param name="blockDim">Block dimension for the kernel</param>
	/// <param name="gridDim">Grid dimension for the kernel</param>
	/// <param name="...args">Kernel arguments</param>
    template<typename... Args>
    void executeKernel(void(*kernel)(Args...), dim3 blockDim, dim3 gridDim, Args... args)
    {
		executeKernel<Args...>(kernel, blockDim, gridDim, 0, args...);
    }


	/// <summary>
	/// Executes a cuda kernel
	/// </summary>
	/// <typeparam name="...Args">Kernel argument types</typeparam>
	/// <param name="kernel">Function pointer to the kernel</param>
	/// <param name="blockDim">Block dimension for the kernel</param>
	/// <param name="gridDim">Grid dimension for the kernel</param>
	/// <param name="sharedMemSize">Shared memory in bytes</param>
	/// <param name="...args">Kernel arguments</param>
	template<typename... Args>
	void executeKernel(void(*kernel)(Args...), dim3 blockDim, dim3 gridDim, size_t sharedMemSize, Args... args)
	{
		executeKernel<Args...>(kernel, blockDim, gridDim, sharedMemSize, 0, args...);
	}

	/// <summary>
	/// Executes a cuda kernel
	/// </summary>
	/// <typeparam name="...Args">Kernel argument types</typeparam>
	/// <param name="kernel">Function pointer to the kernel</param>
	/// <param name="blockDim">Block dimension for the kernel</param>
	/// <param name="gridDim">Grid dimension for the kernel</param>
	/// <param name="sharedMemSize">Shared memory in bytes</param>
	/// <param name="stream">Cuda stream to launch the kernel on</param>
	/// <param name="...args">Kernel arguments</param>
	template<typename... Args>
	void executeKernel(void(*kernel)(Args...), dim3 blockDim, dim3 gridDim, size_t sharedMemSize, cudaStream_t stream, Args... args)
	{
		std::array<void*, sizeof...(args)> arguments = {
			&args...
		};

		const cudaError_t status = cudaLaunchKernel((void*)kernel, gridDim, blockDim, &arguments[0], sharedMemSize, stream);
		if (status != cudaError::cudaSuccess)
		{
			throw std::runtime_error("Failed to launch kernel.");
		}
	}

	/// <summary>
	/// Executes a cooperative cuda kernel
	/// </summary>
	/// <typeparam name="...Args">Kernel argument types</typeparam>
	/// <param name="kernel">Function pointer to the cooperative kernel</param>
	/// <param name="blockDim">Block dimension for the kernel</param>
	/// <param name="gridDim">Grid dimension for the kernel</param>
	/// <param name="sharedMemSize">Shared memory in bytes</param>
	/// <param name="stream">Cuda stream to launch the kernel on</param>
	/// <param name="...args">Kernel arguments</param>
	template<typename... Args>
	void executeCoopKernel(void(*kernel)(Args...), dim3 blockDim, dim3 gridDim, size_t sharedMemSize, cudaStream_t stream, Args... args)
	{
		std::array<void*, sizeof...(args)> arguments = {
			&args...
		};

		const cudaError_t status = cudaLaunchCooperativeKernel((void*)kernel, gridDim, blockDim, &arguments[0], sharedMemSize, stream);
		if (status != cudaError::cudaSuccess)
		{
			throw std::runtime_error("Failed to launch kernel.");
		}
	}
}