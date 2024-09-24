// Unit tests for random number generation
#include <iostream>
#include "Random.h"

void testUniformRandomNumbers() {
    const int N = 1000;
    float randomNumbers[N];

    Random::generateUniform(randomNumbers, N, 1234);

    for (int i = 0; i < N; ++i) {
        if (randomNumbers[i] < 0.0f || randomNumbers[i] > 1.0f) {
            std::cerr << "Uniform random number generation test failed at index " << i << std::endl;
            return;
        }
    }

    std::cout << "Uniform random number generation test passed" << std::endl;
}

int main() {
    testUniformRandomNumbers();
    return 0;
}