// include/Matrix.h
#ifndef MATRIX_H
#define MATRIX_H

template <typename T>
class Matrix {
public:
    static void multiply(const T* A, const T* B, T* C, int N);
    static void transpose(const T* A, T* B, int N);
    static void invert(const T* A, T* B, int N);
};

#endif // MATRIX_H