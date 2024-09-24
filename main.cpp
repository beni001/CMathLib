// Example usage of the library
// src/main.cpp
#include <iostream>
#include <cmath>
#include "MathOperationsFactory.h"
#include "Random.h"

// Function to initialize the weights of the neural network
void initializeWeights(float* weights, int size) {
    // Generate random uniform values for the weights
    Random::generateUniform(weights, size, 1234);
}

// Function to perform a forward pass through the neural network
void forwardPass(IMathOperations<float>* mathOps, float* input, float* weights, float* output, int inputSize, int outputSize) {
    // Perform matrix multiplication: output = input * weights
    mathOps->matrixMultiply(input, weights, output, inputSize, outputSize);
    
    // Apply the ReLU activation function to the output
    for (int i = 0; i < outputSize; ++i) {
        output[i] = std::max(0.0f, output[i]);
    }
}

// Function to compute the loss (mean squared error)
float computeLoss(float* output, float* target, int size) {
    float loss = 0.0f;
    for (int i = 0; i < size; ++i) {
        float diff = output[i] - target[i];
        loss += diff * diff;
    }
    return loss / size;
}

// Function to perform backpropagation and update weights
void backpropagateAndUpdateWeights(IMathOperations<float>* mathOps, float* input, float* hiddenLayer, float* outputLayer, float* hiddenWeights, float* outputWeights, float* target, int inputSize, int hiddenSize, int outputSize, float learningRate) {
    // Compute output layer error (gradient of loss with respect to output)
    float outputError[outputSize];
    for (int i = 0; i < outputSize; ++i) {
        outputError[i] = 2.0f * (outputLayer[i] - target[i]) / outputSize;
    }

    // Compute hidden layer error
    float hiddenError[hiddenSize] = {0};
    for (int i = 0; i < hiddenSize; ++i) {
        for (int j = 0; j < outputSize; ++j) {
            hiddenError[i] += outputError[j] * outputWeights[i * outputSize + j];
        }
        hiddenError[i] *= (hiddenLayer[i] > 0) ? 1 : 0; // Derivative of ReLU
    }

    // Update output weights
    for (int i = 0; i < hiddenSize; ++i) {
        for (int j = 0; j < outputSize; ++j) {
            outputWeights[i * outputSize + j] -= learningRate * outputError[j] * hiddenLayer[i];
        }
    }

    // Update hidden weights
    for (int i = 0; i < inputSize; ++i) {
        for (int j = 0; j < hiddenSize; ++j) {
            hiddenWeights[i * hiddenSize + j] -= learningRate * hiddenError[j] * input[i];
        }
    }
}

// Function to train the neural network
void trainNeuralNetwork() {
    const int inputSize = 784; // Example: 28x28 images flattened into a 1D array
    const int hiddenSize = 128; // Number of neurons in the hidden layer
    const int outputSize = 10; // Number of classes for classification
    const float learningRate = 0.01f; // Learning rate for weight updates

    // Arrays to hold the input data, weights, and layer outputs
    float input[inputSize];
    float hiddenWeights[inputSize * hiddenSize];
    float outputWeights[hiddenSize * outputSize];
    float hiddenLayer[hiddenSize];
    float outputLayer[outputSize];
    float target[outputSize]; // Target output for training

    // Initialize the weights for the hidden and output layers
    initializeWeights(hiddenWeights, inputSize * hiddenSize);
    initializeWeights(outputWeights, hiddenSize * outputSize);

    // Create a math operations object for GPU computations
    IMathOperations<float>* mathOps = MathOperationsFactory::createMathOperations(ImplementationType::GPU);

    // Example training loop (simplified)
    for (int epoch = 0; epoch < 10; ++epoch) {
        // Perform a forward pass through the hidden layer
        forwardPass(mathOps, input, hiddenWeights, hiddenLayer, inputSize, hiddenSize);
        
        // Perform a forward pass through the output layer
        forwardPass(mathOps, hiddenLayer, outputWeights, outputLayer, hiddenSize, outputSize);

        // Compute the loss (mean squared error)
        float loss = computeLoss(outputLayer, target, outputSize);
        std::cout << "Epoch " << epoch << " - Loss: " << loss << std::endl;

        // Perform backpropagation and update weights
        backpropagateAndUpdateWeights(mathOps, input, hiddenLayer, outputLayer, hiddenWeights, outputWeights, target, inputSize, hiddenSize, outputSize, learningRate);
    }

    // Clean up and delete the math operations object
    delete mathOps;
}

// Main function to start the training process
int main() {
    // Train the neural network
    trainNeuralNetwork();
    return 0;
}
