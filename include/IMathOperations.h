// Interface for math operations
// Defines the virtual methods for basic and advanced math operations
template <typename T>
class IMathOperations {
public:
    virtual void add(T* a, T* b, T* c, int n) = 0;
    virtual void matrixMultiply(T* A, T* B, T* C, int N) = 0;
    virtual void sparseMatrixVectorMultiply(int* rowPtr, int* colIdx, T* values, T* x, T* y, int numRows) = 0;
    virtual void generateUniformRandomNumbers(T* result, int n, unsigned long seed) = 0;
    virtual ~IMathOperations() = default;
};