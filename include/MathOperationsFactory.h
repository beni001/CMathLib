// Factory for creating math operations objects
// Provides a method to create either CPU or GPU implementations
#include "IMathOperations.h"

enum class ImplementationType {
    CPU,
    GPU
};

class MathOperationsFactory {
public:
    static IMathOperations<float>* createMathOperations(ImplementationType type);
};