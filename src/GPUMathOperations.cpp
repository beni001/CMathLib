// GPU implementation of math operations
// Implements the IMathOperations interface using CUDA for GPU operations
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "IMathOperations.h"
#include "kernels/math_kernels.cu"
#include "kernels/matrix_kernels.cu"
#include "kernels/sparse_kernels.cu"
#include "MemoryPool.h"

class GPUMathOperations : public IMathOperations<float> {
public:
    GPUMathOperations() {
        cudaHostAlloc(&pinnedMemory, poolSize * sizeof(float), cudaHostAllocDefault);
    }

    ~GPUMathOperations() {
        cudaFreeHost(pinnedMemory);
    }

    void add(float* a, float* b, float* c, int n) override {
        float *d_a, *d_b, *d_c;
        cudaStream_t stream;
        cudaStreamCreate(&stream);

        cudaMalloc((void**)&d_a, n * sizeof(float));
        cudaMalloc((void**)&d_b, n * sizeof(float));
        cudaMalloc((void**)&d_c, n * sizeof(float));

        cudaMemcpyAsync(d_a, a, n * sizeof(float), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_b, b, n * sizeof(float), cudaMemcpyHostToDevice, stream);

        int blockSize = 256;
        int numBlocks = (n + blockSize - 1) / blockSize;
        addKernel<<<numBlocks, blockSize, 0, stream>>>(d_a, d_b, d_c, n);

        cudaMemcpyAsync(c, d_c, n * sizeof(float), cudaMemcpyDeviceToHost, stream);

        cudaStreamSynchronize(stream);
        cudaStreamDestroy(stream);

        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
    }

    void matrixMultiply(float* A, float* B, float* C, int N) override {
        float *d_A, *d_B, *d_C;
        cudaStream_t stream;
        cudaStreamCreate(&stream);

        cudaMalloc((void**)&d_A, N * N * sizeof(float));
        cudaMalloc((void**)&d_B, N * N * sizeof(float));
        cudaMalloc((void**)&d_C, N * N * sizeof(float));

        cudaMemcpyAsync(d_A, A, N * N * sizeof(float), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_B, B, N * N * sizeof(float), cudaMemcpyHostToDevice, stream);

        dim3 blockSize(16, 16);
        dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (N + blockSize.y - 1) / blockSize.y);
        matrixMulKernel<<<gridSize, blockSize, 0, stream>>>(d_A, d_B, d_C, N);

        cudaMemcpyAsync(C, d_C, N * N * sizeof(float), cudaMemcpyDeviceToHost, stream);

        cudaStreamSynchronize(stream);
        cudaStreamDestroy(stream);

        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
    }

    void sparseMatrixVectorMultiply(int* rowPtr, int* colIdx, float* values, float* x, float* y, int numRows) override {
        int *d_rowPtr, *d_colIdx;
        float *d_values, *d_x, *d_y;
        cudaStream_t stream;
        cudaStreamCreate(&stream);

        cudaMalloc((void**)&d_rowPtr, (numRows + 1) * sizeof(int));
        cudaMalloc((void**)&d_colIdx, nnz * sizeof(int));
        cudaMalloc((void**)&d_values, nnz * sizeof(float));
        cudaMalloc((void**)&d_x, numCols * sizeof(float));
        cudaMalloc((void**)&d_y, numRows * sizeof(float));

        cudaMemcpyAsync(d_rowPtr, rowPtr, (numRows + 1) * sizeof(int), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_colIdx, colIdx, nnz * sizeof(int), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_values, values, nnz * sizeof(float), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_x, x, numCols * sizeof(float), cudaMemcpyHostToDevice, stream);

        int blockSize = 256;
        int numBlocks = (numRows + blockSize - 1) / blockSize;
        sparseMatrixVectorMulKernel<<<numBlocks, blockSize, 0, stream>>>(d_rowPtr, d_colIdx, d_values, d_x, d_y, numRows);

        cudaMemcpyAsync(y, d_y, numRows * sizeof(float), cudaMemcpyDeviceToHost, stream);

        cudaStreamSynchronize(stream);
        cudaStreamDestroy(stream);

        cudaFree(d_rowPtr);
        cudaFree(d_colIdx);
        cudaFree(d_values);
        cudaFree(d_x);
        cudaFree(d_y);
    }

    void generateUniformRandomNumbers(float* result, int n, unsigned long seed) override {
        curandState* d_state;
        cudaMalloc(&d_state, n * sizeof(curandState));

        int blockSize = 256;
        int numBlocks = (n + blockSize - 1) / blockSize;
        setupKernel<<<numBlocks, blockSize>>>(d_state, seed);
        generateUniformKernel<<<numBlocks, blockSize>>>(d_state, result, n);

        cudaFree(d_state);
    }

private:
    float* pinnedMemory;
    const size_t poolSize = 1024 * 1024; // Example size
};