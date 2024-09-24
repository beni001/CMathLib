// include/Random.h
#ifndef RANDOM_H
#define RANDOM_H

#include <curand_kernel.h>

class Random {
public:
    static void generateUniform(float* result, int n, unsigned long seed);
    static void generateNormal(float* result, int n, unsigned long seed);
};

#endif // RANDOM_H