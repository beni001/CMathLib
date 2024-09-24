// Unit tests for math operations
// Tests the correctness of basic and advanced math operations
#include <iostream>
#include "MathOperationsFactory.h"

void testAddition() {
    const int N = 1000;
    float a[N], b[N], c[N];

    for (int i = 0; i < N; ++i) {
        a[i] = static_cast<float>(i);
        b[i] = static_cast<float>(i * 2);
    }

    IMathOperations<float>* mathOps = MathOperationsFactory::createMathOperations(ImplementationType::GPU);
    mathOps->add(a, b, c, N);

    for (int i = 0; i < N; ++i) {
        if (c[i] != a[i] + b[i]) {
            std::cerr << "Addition test failed at index " << i << std::endl;
            return;
        }
    }

    std::cout << "Addition test passed" << std::endl;
    delete mathOps;
}

int main() {
    testAddition();
    return 0;
}