// Example usage of the library
// Demonstrates how to use the library to perform math operations
#include <iostream>
#include "MathOperationsFactory.h"

int main() {
    const int N = 1000;
    float a[N], b[N], c[N];

    for (int i = 0; i < N; ++i) {
        a[i] = static_cast<float>(i);
        b[i] = static_cast<float>(i * 2);
    }

    IMathOperations<float>* mathOps = MathOperationsFactory::createMathOperations(ImplementationType::GPU);
    mathOps->add(a, b, c, N);

    for (int i = 0; i < 10; ++i) {
        std::cout << c[i] << " ";
    }
    std::cout << std::endl;

    delete mathOps;
    return 0;
}