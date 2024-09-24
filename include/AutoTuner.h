#ifndef AUTO_TUNER_H
#define AUTO_TUNER_H

#include <cuda_runtime.h>

class AutoTuner {
public:
    // Tune kernel configuration for a given problem size
    static void tuneKernel(dim3& gridDim, dim3& blockDim, int n);

    // Benchmark a kernel with a specific configuration
    template <typename KernelFunc, typename... Args>
    static float benchmarkKernel(KernelFunc kernel, dim3 gridDim, dim3 blockDim, Args... args);

private:
    // Helper function to get the best configuration
    static void getBestConfiguration(dim3& gridDim, dim3& blockDim, int n);
};

#endif // AUTO_TUNER_H