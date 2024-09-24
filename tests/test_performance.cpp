// Performance tests and benchmarks
// Measures the performance of math operations and compares with established libraries
#include <iostream>
#include <chrono>
#include "MathOperationsFactory.h"

void benchmarkAddition() {
    const int N = 1000000;
    float a[N], b[N], c[N];

    for (int i = 0; i < N; ++i) {
        a[i] = static_cast<float>(i);
        b[i] = static_cast<float>(i * 2);
    }

    IMathOperations<float>* mathOps = MathOperationsFactory::createMathOperations(ImplementationType::GPU);

    auto start = std::chrono::high_resolution_clock::now();
    mathOps->add(a, b, c, N);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> duration = end - start;
    std::cout << "Addition took " << duration.count() << " seconds" << std::endl;

    delete mathOps;
}

int main() {
    benchmarkAddition();
    return 0;
}