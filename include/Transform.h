// include/Transform.h
#ifndef TRANSFORM_H
#define TRANSFORM_H

template <typename T>
class Transform {
public:
    static void translate(T* matrix, T x, T y, T z);
    static void rotate(T* matrix, const T* quaternion);
    static void scale(T* matrix, T sx, T sy, T sz);
};

#endif // TRANSFORM_H