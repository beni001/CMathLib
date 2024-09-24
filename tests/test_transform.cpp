// Unit tests for transformation operations
#include <iostream>
#include "Transform.h"

void testTranslation() {
    float matrix[16] = {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f
    };

    Transform<float>::translate(matrix, 1.0f, 2.0f, 3.0f);

    float expected[16] = {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
        1.0f, 2.0f, 3.0f, 1.0f
    };

    for (int i = 0; i < 16; ++i) {
        if (matrix[i] != expected[i]) {
            std::cerr << "Translation test failed at index " << i << std::endl;
            return;
        }
    }

    std::cout << "Translation test passed" << std::endl;
}

int main() {
    testTranslation();
    return 0;
}