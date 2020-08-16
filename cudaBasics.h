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

namespace cudabasic
{

    template<typename T>
    class cpuGpuArray
    {
    private:
        int32_t arrLength;
        T* gpuArray;
        std::vector<T>* cpuArray;

    public:
        cpuGpuArray(const int32_t length)
        {
            this->arrLength = length;
            const cudaError_t status = cudaMalloc(&this->gpuArray, length * sizeof(T));
            if (status != cudaError::cudaSuccess)
            {
                throw std::runtime_error("Failed to allocate cuda memory.");
            }

            this->cpuArray = new std::vector<T>(length);
        }

        void copyToGPU()
        {
            const cudaError_t status = cudaMemcpy(gpuArray, &(*cpuArray)[0], arrLength * sizeof(T), cudaMemcpyKind::cudaMemcpyHostToDevice);
            if (status != cudaError::cudaSuccess)
            {
                throw std::runtime_error("Failed to copy from host to device.");
            }
        }

        std::vector<T>& copyFromGPU()
        {
            const cudaError_t status = cudaMemcpy(&(*cpuArray)[0], gpuArray, arrLength * sizeof(T), cudaMemcpyKind::cudaMemcpyDeviceToHost);
            if (status != cudaError::cudaSuccess)
            {
                throw std::runtime_error("Failed to copy from device from host.");
            }

            return getCPUArray();
        }

        T* getGPUArray()
        {
            return gpuArray;
        }

        std::vector<T>& getCPUArray()
        {
            return *cpuArray;
        }

        ~cpuGpuArray()
        {
            //deallocate gpu array
            const cudaError_t status = cudaFree(gpuArray);
            if (status != cudaError::cudaSuccess)
            {
                throw std::runtime_error("Failed to deallocate cuda memory.");
            }

            //deallocate cpu array
            delete cpuArray;
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
        ~cudaTimer();

        void startTimer();
        void stopTimer();
        float getElapsedMiliseconds();
    };

    /// <summary>
    /// Executes a cuda kernel
    /// </summary>
    /// <typeparam name="...Args"></typeparam>
    /// <param name="kernel">Function pointer to the kernel</param>
    /// <param name="blockDim">Block dimension for the kernel</param>
    /// <param name="gridDim">Grid dimension for the kernel</param>
    /// <param name="...args">Kernel arguments</param>
    template<typename... Args>
    void executeKernel(void(*kernel)(Args...), dim3 blockDim, dim3 gridDim, Args... args)
    {
        (*kernel)<<<gridDim, blockDim>>>(args...);
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
        /// Benchmarks with the specified grid and kernel dimensions and kernel arguments
        /// </summary>
        /// <param name="blockDim">Size of a single block on threads</param>
        /// <param name="gridDim">Size of the grid in blocks</param>
        /// <param name="...args">Kernel arguments</param>
        /// <returns>The average execution time</returns>
        float benchmark(dim3 blockDim, dim3 gridDim, Args... args)
        {
            float time = 0.0f;
            for (size_t i = 0; i < benchCount; i++)
            {
                timer.startTimer();
                executeKernel(kernel, blockDim, gridDim, args...);
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

		cpuGpuObject(memPlacmenet placce)
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
	private:
		Matrix(float* ptr, int columns, int rows) : cpuGpuObject(memPlacmenet::GPU)
		{
			this->ptr = ptr;
			this->columns = columns;
			this->rows = rows;
		}

	public:
		float* ptr;
		int columns;
		int rows;
	public:
		Matrix(int columns, int rows) : cpuGpuObject(memPlacmenet::CPU)
		{
			this->ptr = new float[columns * rows];
			this->columns = columns;
			this->rows = rows;
		}

		~Matrix()
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