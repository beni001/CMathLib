// CUDA kernels for transformation operations
#include <cuda_runtime.h>

__global__ void translateKernel(float* matrix, float x, float y, float z) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i == 0) {
        matrix[12] += x;
        matrix[13] += y;
        matrix[14] += z;
    }
}

__global__ void rotateKernel(float* matrix, const float* quaternion) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i == 0) {
        float q[16];
        quaternionToMatrixKernel<<<1, 1>>>(quaternion, q);
        __syncthreads();

        float temp[16];
        for (int row = 0; row < 4; ++row) {
            for (int col = 0; col < 4; ++col) {
                temp[row * 4 + col] = 0;
                for (int k = 0; k < 4; ++k) {
                    temp[row * 4 + col] += matrix[row * 4 + k] * q[k * 4 + col];
                }
            }
        }

        for (int j = 0; j < 16; ++j) {
            matrix[j] = temp[j];
        }
    }
}

__global__ void scaleKernel(float* matrix, float sx, float sy, float sz) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i == 0) {
        matrix[0] *= sx;
        matrix[5] *= sy;
        matrix[10] *= sz;
    }
}