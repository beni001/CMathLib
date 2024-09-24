// CUDA kernels for vector operations
#include <cuda_runtime.h>

__global__ void vectorAddKernel(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

__global__ void vectorSubtractKernel(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] - b[i];
    }
}

__global__ void vectorDotKernel(const float* a, const float* b, float* result, int n) {
    __shared__ float cache[256];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int cacheIndex = threadIdx.x;

    float temp = 0;
    while (i < n) {
        temp += a[i] * b[i];
        i += blockDim.x * gridDim.x;
    }

    cache[cacheIndex] = temp;

    __syncthreads();

    int j = blockDim.x / 2;
    while (j != 0) {
        if (cacheIndex < j) {
            cache[cacheIndex] += cache[cacheIndex + j];
        }
        __syncthreads();
        j /= 2;
    }

    if (cacheIndex == 0) {
        atomicAdd(result, cache[0]);
    }
}

__global__ void vectorCrossKernel(const float* a, const float* b, float* result) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i == 0) {
        result[0] = a[1] * b[2] - a[2] * b[1];
        result[1] = a[2] * b[0] - a[0] * b[2];
        result[2] = a[0] * b[1] - a[1] * b[0];
    }
}