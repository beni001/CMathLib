// CPU implementation of math operations
// Implements the IMathOperations interface using CPU operations
#include "IMathOperations.h"

class CPUMathOperations : public IMathOperations<float> {
public:
    void add(float* a, float* b, float* c, int n) override {
        for (int i = 0; i < n; ++i) {
            c[i] = a[i] + b[i];
        }
    }

    void matrixMultiply(float* A, float* B, float* C, int N) override {
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                float sum = 0.0f;
                for (int k = 0; k < N; ++k) {
                    sum += A[i * N + k] * B[k * N + j];
                }
                C[i * N + j] = sum;
            }
        }
    }

    void sparseMatrixVectorMultiply(int* rowPtr, int* colIdx, float* values, float* x, float* y, int numRows) override {
        for (int row = 0; row < numRows; ++row) {
            float sum = 0.0f;
            for (int j = rowPtr[row]; j < rowPtr[row + 1]; ++j) {
                sum += values[j] * x[colIdx[j]];
            }
            y[row] = sum;
        }
    }

    void generateUniformRandomNumbers(float* result, int n, unsigned long seed) override {
        srand(seed);
        for (int i = 0; i < n; ++i) {
            result[i] = static_cast<float>(rand()) / RAND_MAX;
        }
    }
};