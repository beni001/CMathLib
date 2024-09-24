// Implementation of the factory pattern
// Creates either CPU or GPU implementations of math operations based on the specified type
#include "MathOperationsFactory.h"
#include "CPUMathOperations.cpp"
#include "GPUMathOperations.cpp"

IMathOperations<float>* MathOperationsFactory::createMathOperations(ImplementationType type) {
    switch (type) {
        case ImplementationType::CPU:
            return new CPUMathOperations();
        case ImplementationType::GPU:
            return new GPUMathOperations();
        default:
            return nullptr;
    }
}