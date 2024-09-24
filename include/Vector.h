// include/Vector.h
#ifndef VECTOR_H
#define VECTOR_H

template <typename T>
class Vector {
public:
    static void add(const T* a, const T* b, T* c, int n);
    static void subtract(const T* a, const T* b, T* c, int n);
    static void dot(const T* a, const T* b, T& result, int n);
    static void cross(const T* a, const T* b, T* result);
};

#endif // VECTOR_H