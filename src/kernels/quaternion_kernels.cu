// CUDA kernels for quaternion operations
#include <cuda_runtime.h>

__global__ void quaternionMultiplyKernel(const float* q1, const float* q2, float* result) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i == 0) {
        result[0] = q1[0] * q2[0] - q1[1] * q2[1] - q1[2] * q2[2] - q1[3] * q2[3];
        result[1] = q1[0] * q2[1] + q1[1] * q2[0] + q1[2] * q2[3] - q1[3] * q2[2];
        result[2] = q1[0] * q2[2] - q1[1] * q2[3] + q1[2] * q2[0] + q1[3] * q2[1];
        result[3] = q1[0] * q2[3] + q1[1] * q2[2] - q1[2] * q2[1] + q1[3] * q2[0];
    }
}

__global__ void quaternionNormalizeKernel(float* q) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i == 0) {
        float norm = sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
        q[0] /= norm;
        q[1] /= norm;
        q[2] /= norm;
        q[3] /= norm;
    }
}

__global__ void quaternionToMatrixKernel(const float* q, float* matrix) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i == 0) {
        matrix[0] = 1 - 2 * q[2] * q[2] - 2 * q[3] * q[3];
        matrix[1] = 2 * q[1] * q[2] - 2 * q[3] * q[0];
        matrix[2] = 2 * q[1] * q[3] + 2 * q[2] * q[0];
        matrix[3] = 0;

        matrix[4] = 2 * q[1] * q[2] + 2 * q[3] * q[0];
        matrix[5] = 1 - 2 * q[1] * q[1] - 2 * q[3] * q[3];
        matrix[6] = 2 * q[2] * q[3] - 2 * q[1] * q[0];
        matrix[7] = 0;

        matrix[8] = 2 * q[1] * q[3] - 2 * q[2] * q[0];
        matrix[9] = 2 * q[2] * q[3] + 2 * q[1] * q[0];
        matrix[10] = 1 - 2 * q[1] * q[1] - 2 * q[2] * q[2];
        matrix[11] = 0;

        matrix[12] = 0;
        matrix[13] = 0;
        matrix[14] = 0;
        matrix[15] = 1;
    }
}