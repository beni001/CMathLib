// Unit tests for matrix operations
#include <iostream>
#include "Matrix.h"

void testMatrixMultiplication() {
    const int N = 2;
    float A[N * N] = {1.0f, 2.0f, 3.0f, 4.0f};
    float B[N * N] = {5.0f, 6.0f, 7.0f, 8.0f};
    float C[N * N];

    Matrix<float>::multiply(A, B, C, N);

    float expected[N * N] = {19.0f, 22.0f, 43.0f, 50.0f};
    for (int i = 0; i < N * N; ++i) {
        if (C[i] != expected[i]) {
            std::cerr << "Matrix multiplication test failed at index " << i << std::endl;
            return;
        }
    }

    std::cout << "Matrix multiplication test passed" << std::endl;
}

int main() {
    testMatrixMultiplication();
    return 0;
}