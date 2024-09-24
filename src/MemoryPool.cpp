// Implementation of memory pool
// Manages a pool of pre-allocated memory blocks to reduce allocation overhead
#include "MemoryPool.h"

MemoryPool::MemoryPool(size_t poolSize) : poolSize(poolSize) {
    cudaMalloc(&pool, poolSize);
}

MemoryPool::~MemoryPool() {
    cudaFree(pool);
}

void* MemoryPool::allocate(size_t size) {
    if (!freeBlocks.empty()) {
        void* ptr = freeBlocks.back();
        freeBlocks.pop_back();
        return ptr;
    }
    return nullptr; // Simplified for brevity; should handle allocation from pool
}

void MemoryPool::deallocate(void* ptr) {
    freeBlocks.push_back(ptr);
}