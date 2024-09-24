// Unit tests for quaternion operations
#include <iostream>
#include "Quaternion.h"

void testQuaternionMultiplication() {
    float q1[4] = {1.0f, 0.0f, 1.0f, 0.0f};
    float q2[4] = {1.0f, 0.5f, 0.5f, 0.75f};
    float result[4];

    Quaternion<float>::multiply(q1, q2, result);

    float expected[4] = {0.5f, 1.25f, 1.5f, 0.25f};
    for (int i = 0; i < 4; ++i) {
        if (result[i] != expected[i]) {
            std::cerr << "Quaternion multiplication test failed at index " << i << std::endl;
            return;
        }
    }

    std::cout << "Quaternion multiplication test passed" << std::endl;
}

int main() {
    testQuaternionMultiplication();
    return 0;
}