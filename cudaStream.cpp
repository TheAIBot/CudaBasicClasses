#include <stdexcept>
#include "cudaStream.h"

namespace cudabasic
{
	cudaStream::cudaStream()
	{
		const cudaError_t status = cudaStreamCreate(&stream);
		if (status != cudaError::cudaSuccess)
		{
			throw std::runtime_error("Failed to allocate cuda stream.");
		}
	}
	cudaStream::~cudaStream() noexcept(false)
	{
		const cudaError_t status = cudaStreamDestroy(stream);
		if (status != cudaError::cudaSuccess)
		{
			throw std::runtime_error("Failed to deallocate cuda stream.");
		}
	}

	void cudaStream::synchronize() const
	{
		cudaStreamSynchronize(stream);
	}

	cudaStream::operator cudaStream_t() const
	{
		return stream;
	}
}