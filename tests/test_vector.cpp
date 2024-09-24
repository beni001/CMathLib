// Unit tests for vector operations
#include <iostream>
#include "Vector.h"

void testVectorAddition() {
    const int N = 3;
    float a[N] = {1.0f, 2.0f, 3.0f};
    float b[N] = {4.0f, 5.0f, 6.0f};
    float c[N];

    Vector<float>::add(a, b, c, N);

    for (int i = 0; i < N; ++i) {
        if (c[i] != a[i] + b[i]) {
            std::cerr << "Vector addition test failed at index " << i << std::endl;
            return;
        }
    }

    std::cout << "Vector addition test passed" << std::endl;
}

int main() {
    testVectorAddition();
    return 0;
}