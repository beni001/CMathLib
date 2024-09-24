// Memory pool for efficient memory management
// Manages a pool of pre-allocated memory blocks to reduce allocation overhead
#include <vector>

class MemoryPool {
public:
    MemoryPool(size_t poolSize);
    ~MemoryPool();
    void* allocate(size_t size);
    void deallocate(void* ptr);

private:
    std::vector<void*> freeBlocks;
    void* pool;
    size_t poolSize;
};