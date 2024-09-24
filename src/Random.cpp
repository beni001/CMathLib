// src/Random.cpp
#include "Random.h"

__global__ void setupKernel(curandState* state, unsigned long seed) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed, id, 0, &state[id]);
}

__global__ void generateUniformKernel(curandState* state, float* result, int n) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n) {
        result[id] = curand_uniform(&state[id]);
    }
}

__global__ void generateNormalKernel(curandState* state, float* result, int n) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n) {
        result[id] = curand_normal(&state[id]);
    }
}

void Random::generateUniform(float* result, int n, unsigned long seed) {
    curandState* d_state;
    cudaMalloc(&d_state, n * sizeof(curandState));

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    setupKernel<<<numBlocks, blockSize>>>(d_state, seed);
    generateUniformKernel<<<numBlocks, blockSize>>>(d_state, result, n);

    cudaFree(d_state);
}

void Random::generateNormal(float* result, int n, unsigned long seed) {
    curandState* d_state;
    cudaMalloc(&d_state, n * sizeof(curandState));

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    setupKernel<<<numBlocks, blockSize>>>(d_state, seed);
    generateNormalKernel<<<numBlocks, blockSize>>>(d_state, result, n);

    cudaFree(d_state);
}