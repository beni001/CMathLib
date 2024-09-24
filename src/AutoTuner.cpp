#include "AutoTuner.h"
#include <cuda_runtime.h>
#include <iostream>
#include <chrono>

void AutoTuner::tuneKernel(dim3& gridDim, dim3& blockDim, int n) {
    getBestConfiguration(gridDim, blockDim, n);
}

template <typename KernelFunc, typename... Args>
float AutoTuner::benchmarkKernel(KernelFunc kernel, dim3 gridDim, dim3 blockDim, Args... args) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    kernel<<<gridDim, blockDim>>>(args...);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return milliseconds;
}

void AutoTuner::getBestConfiguration(dim3& gridDim, dim3& blockDim, int n) {
    // Simplified example; real implementation would benchmark different configurations
    int bestBlockSize = 256;
    float bestTime = FLT_MAX;

    for (int blockSize = 32; blockSize <= 1024; blockSize *= 2) {
        dim3 tempBlockDim(blockSize);
        dim3 tempGridDim((n + blockSize - 1) / blockSize);

        // Example kernel for benchmarking
        auto kernel = [] __global__ (int* data, int n) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < n) {
                data[idx] = data[idx] * 2;
            }
        };

        int* d_data;
        cudaMalloc(&d_data, n * sizeof(int));
        float time = benchmarkKernel(kernel, tempGridDim, tempBlockDim, d_data, n);
        cudaFree(d_data);

        if (time < bestTime) {
            bestTime = time;
            bestBlockSize = blockSize;
        }
    }

    blockDim = dim3(bestBlockSize);
    gridDim = dim3((n + bestBlockSize - 1) / bestBlockSize);
}

// Explicit instantiation of the template function
template float AutoTuner::benchmarkKernel<void(*)(int*, int), int*, int>(void(*)(int*, int), dim3, dim3, int*, int);