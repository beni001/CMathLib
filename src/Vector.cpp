// src/Vector.cpp
#include "Vector.h"

template <typename T>
void Vector<T>::add(const T* a, const T* b, T* c, int n) {
    for (int i = 0; i < n; ++i) {
        c[i] = a[i] + b[i];
    }
}

template <typename T>
void Vector<T>::subtract(const T* a, const T* b, T* c, int n) {
    for (int i = 0; i < n; ++i) {
        c[i] = a[i] - b[i];
    }
}

template <typename T>
void Vector<T>::dot(const T* a, const T* b, T& result, int n) {
    result = 0;
    for (int i = 0; i < n; ++i) {
        result += a[i] * b[i];
    }
}

template <typename T>
void Vector<T>::cross(const T* a, const T* b, T* result) {
    result[0] = a[1] * b[2] - a[2] * b[1];
    result[1] = a[2] * b[0] - a[0] * b[2];
    result[2] = a[0] * b[1] - a[1] * b[0];
}