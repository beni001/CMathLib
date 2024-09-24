// src/Transform.cpp
#include "Transform.h"

template <typename T>
void Transform<T>::translate(T* matrix, T x, T y, T z) {
    matrix[12] += x;
    matrix[13] += y;
    matrix[14] += z;
}

template <typename T>
void Transform<T>::rotate(T* matrix, const T* quaternion) {
    // Apply rotation using quaternion
}

template <typename T>
void Transform<T>::scale(T* matrix, T sx, T sy, T sz) {
    matrix[0] *= sx;
    matrix[5] *= sy;
    matrix[10] *= sz;
}