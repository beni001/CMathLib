// CUDA kernels for sparse matrix operations
// Contains optimized kernels for sparse matrix-vector multiplication using CSR format
__global__ void sparseMatrixVectorMulKernel(int* rowPtr, int* colIdx, float* values, float* x, float* y, int numRows) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < numRows) {
        float sum = 0.0f;
        for (int j = rowPtr[row]; j < rowPtr[row + 1]; ++j) {
            sum += values[j] * x[colIdx[j]];
        }
        y[row] = sum;
    }
}