// include/Quaternion.h
#ifndef QUATERNION_H
#define QUATERNION_H

template <typename T>
class Quaternion {
public:
    static void multiply(const T* q1, const T* q2, T* result);
    static void normalize(T* q);
    static void toMatrix(const T* q, T* matrix);
};

#endif // QUATERNION_H