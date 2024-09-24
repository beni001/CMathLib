// CUDA kernels for basic math operations
// Contains optimized kernels for addition, subtraction, multiplication, and division
#include <cuda_runtime.h>

__inline__ __device__ float warpReduceSum(float val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void addKernel(float* a, float* b, float* c, int n) {
    __shared__ float s_a[256];
    __shared__ float s_b[256];

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        s_a[threadIdx.x] = a[i];
        s_b[threadIdx.x] = b[i];
        __syncthreads();

        float val = s_a[threadIdx.x] + s_b[threadIdx.x];
        val = warpReduceSum(val);

        if (threadIdx.x % warpSize == 0) {
            c[blockIdx.x * blockDim.x + threadIdx.x / warpSize] = val;
        }
    }
}