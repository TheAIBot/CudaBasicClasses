#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <stdexcept>
#include <memory>
#include <random>
#include "cpuGpuObject.h"

namespace cudabasic
{
	/// <summary>
	/// Matrix class that's able to exist on both the CPU and GPU
	/// </summary>
	class matrix : public cpuGpuObject<matrix, float>
	{
	public:
		float* ptr;
		uint32_t columns;
		uint32_t rows;

		matrix(uint32_t columns, uint32_t rows) : cpuGpuObject(memPlacmenet::CPU)
		{
			this->ptr = new float[columns * rows];
			this->columns = columns;
			this->rows = rows;
		}

		__device__ __host__ matrix(float* ptr, uint32_t columns, uint32_t rows) : cpuGpuObject(memPlacmenet::GPU)
		{
			this->ptr = ptr;
			this->columns = columns;
			this->rows = rows;
		}

		__device__ __host__ ~matrix()
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

		matrix getAsGPUObject(float* gpuPtr) override
		{
			return matrix(gpuPtr, columns, rows);
		}

		float* getCPUPtr() override
		{
			return ptr;
		}

		__device__ __host__ float* operator[](const uint32_t row)
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

		std::shared_ptr<matrix> operator*(matrix& b)
		{
			if (rows != b.columns)
			{
				throw std::runtime_error("Matrix multiplication dimensions are incorrect.");
			}

			auto c = std::make_shared<matrix>(columns, b.rows);
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
}