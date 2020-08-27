#pragma once

#include <cuda_runtime.h>

namespace cudabasic
{
    class cudaTimer
    {
    private:
        cudaEvent_t startTime;
        cudaEvent_t endTime;

    public:
        cudaTimer();
        ~cudaTimer() noexcept(false);

        void startTimer();
        void startTimer(cudaStream_t stream);
        void stopTimer();
        void stopTimer(cudaStream_t stream);
        float getElapsedMiliseconds();
    };
}