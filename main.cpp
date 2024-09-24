// Example usage of the library
// src/main.cpp
#include <iostream>
#include "MathOperationsFactory.h"
#include "Random.h"

void initializeWeights(float* weights, int size) {
    Random::generateUniform(weights, size, 1234);
}

void forwardPass(IMathOperations<float>* mathOps, float* input, float* weights, float* output, int inputSize, int outputSize) {
    mathOps->matrixMultiply(input, weights, output, inputSize, outputSize);
    // Apply activation function (e.g., ReLU)
    for (int i = 0; i < outputSize; ++i) {
        output[i] = std::max(0.0f, output[i]);
    }
}

void trainNeuralNetwork() {
    const int inputSize = 784; // Example: 28x28 images flattened
    const int hiddenSize = 128;
    const int outputSize = 10; // Example: 10 classes for classification

    float input[inputSize];
    float hiddenWeights[inputSize * hiddenSize];
    float outputWeights[hiddenSize * outputSize];
    float hiddenLayer[hiddenSize];
    float outputLayer[outputSize];

    // Initialize weights
    initializeWeights(hiddenWeights, inputSize * hiddenSize);
    initializeWeights(outputWeights, hiddenSize * outputSize);

    IMathOperations<float>* mathOps = MathOperationsFactory::createMathOperations(ImplementationType::GPU);

    // Example training loop (simplified)
    for (int epoch = 0; epoch < 10; ++epoch) {
        // Forward pass
        forwardPass(mathOps, input, hiddenWeights, hiddenLayer, inputSize, hiddenSize);
        forwardPass(mathOps, hiddenLayer, outputWeights, outputLayer, hiddenSize, outputSize);

        // Compute loss and backpropagate (not implemented in this example)
        // Update weights (not implemented in this example)

        std::cout << "Epoch " << epoch << " completed." << std::endl;
    }

    delete mathOps;
}

int main() {
    trainNeuralNetwork();
    return 0;
}
