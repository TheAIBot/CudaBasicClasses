#pragma once

#include "cuda_runtime.h"
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

	class cudaStream
	{
	private:
		cudaStream_t stream;
	public:
		cudaStream()
		{
			const cudaError_t status = cudaStreamCreate(&stream);
			if (status != cudaError::cudaSuccess)
			{
				throw std::runtime_error("Failed to allocate cuda stream.");
			}
		}
		~cudaStream() noexcept(false)
		{
			const cudaError_t status = cudaStreamDestroy(stream);
			if (status != cudaError::cudaSuccess)
			{
				throw std::runtime_error("Failed to deallocate cuda stream.");
			}
		}

		void synchronize() const
		{
			cudaStreamSynchronize(stream);
		}

		operator cudaStream_t() const 
		{ 
			return stream; 
		}
	};

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
			const cudaError_t status = cudaMemcpy(gpuArray + offset, cpuArray.begin()  + offset, elementCount * sizeof(T), cudaMemcpyKind::cudaMemcpyHostToDevice);
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

	template<typename T>
	class gpuArray
	{
	private:
		int32_t arrLength;
		T* gpuArr;

	public:
		gpuArray(const int32_t length)
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

    void setCudaDevice(const int32_t device);
    void resetCudaDevice();
    void checkForCudaError();
    void cudaSynchronize();


    class cudaTimer
    {
    private:
        cudaEvent_t startTime;
        cudaEvent_t endTime;

    public:
        cudaTimer();
        ~cudaTimer() noexcept(false);

        void startTimer();
        void stopTimer();
        float getElapsedMiliseconds();
    };

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
				timer.startTimer();
				executeKernel(kernel, blockDim, gridDim, sharedMemSize, stream, args...);
				timer.stopTimer();
				time += timer.getElapsedMiliseconds();
			}
			// << <gridDim, blockDim >> > 
			return time / benchCount;
		}
    };





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


	/// <summary>
	/// Matrix class that's able to exist on both the CPU and GPU
	/// </summary>
	class Matrix : public cpuGpuObject<Matrix, float>
	{
	public:
		float* ptr;
		int columns;
		int rows;

		Matrix(int columns, int rows) : cpuGpuObject(memPlacmenet::CPU)
		{
			this->ptr = new float[columns * rows];
			this->columns = columns;
			this->rows = rows;
		}

		__device__ __host__ Matrix(float* ptr, int columns, int rows) : cpuGpuObject(memPlacmenet::GPU)
		{
			this->ptr = ptr;
			this->columns = columns;
			this->rows = rows;
		}

		__device__ __host__ ~Matrix()
		{
			if (place == memPlacmenet::CPU)
			{
				delete[] ptr;
			}
		}

		int getMemOnGPU() override
		{
			return columns * rows;
		}

		Matrix getAsGPUObject(float* gpuPtr) override
		{
			return Matrix(gpuPtr, columns, rows);
		}

		float* getCPUPtr() override
		{
			return ptr;
		}

		__device__ __host__ float* operator[](const int row)
		{
			return ptr + row * columns;
		}

		void makeRandom(float minValue, float maxValue)
		{
			std::default_random_engine rngGen;
			std::uniform_real_distribution<float> dist(minValue, maxValue);

			for (size_t i = 0; i < columns * rows; i++)
			{
				ptr[i] = dist(rngGen);
			}
		}

		void transpose()
		{
			for (int i = 0; i < rows; i++)
			{
				for (int j = 0; j < i; j++)
				{
					float temp = (*this)[i][j];
					(*this)[i][j] = (*this)[j][i];
					(*this)[j][i] = temp;
				}
			}
			int temp = rows;
			rows = columns;
			columns = temp;
		}

		std::shared_ptr<Matrix> operator*(Matrix& b)
		{
			if (rows != b.columns)
			{
				throw std::runtime_error("Matrix multiplication dimensions are incorrect.");
			}

			auto c = std::make_shared<Matrix>(columns, b.rows);
			b.transpose();
			for (int y = 0; y < (*c).columns; y++) {
				for (int x = 0; x < (*c).columns; x++) {
					for (int k = 0; k < columns; k++) {
						(*c)[y][x] += (*this)[y][k] * b[x][k];
					}
				}
			}
			b.transpose();

			return c;
		}
	};

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