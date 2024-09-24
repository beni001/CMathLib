// src/Matrix.cpp
#include "Matrix.h"

template <typename T>
void Matrix<T>::multiply(const T* A, const T* B, T* C, int N) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            T sum = 0;
            for (int k = 0; k < N; ++k) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

template <typename T>
void Matrix<T>::transpose(const T* A, T* B, int N) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            B[j * N + i] = A[i * N + j];
        }
    }
}

template <typename T>
void Matrix<T>::invert(const T* A, T* B, int N) {
    // Simplified example; real implementation would use a numerical method
}