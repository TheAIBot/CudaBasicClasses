#pragma once

#include <cuda_runtime.h>

namespace cudabasic
{
	class cudaStream
	{
	private:
		cudaStream_t stream;
	public:
		cudaStream();
		~cudaStream() noexcept(false);

		void synchronize() const;
		operator cudaStream_t() const;
	};
}