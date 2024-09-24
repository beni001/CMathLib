# CMathLib

CMathLib is a high-performance, GPU-accelerated math library designed for large-scale computations. It offers a comprehensive set of mathematical operations, including vector operations, matrix operations, quaternion operations, and transformations. CMathLib is optimized for various use cases such as game development, 3D modeling, virtual reality (VR) and augmented reality (AR), scientific visualization, computer-aided design (CAD), machine learning, robotics, simulation software, graphics rendering engines, and animation software.

## Table of Contents

- [Features](#features)
- [File Structure](#file-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Sample Use Cases](#sample-use-cases)
- [Why Use CMathLib](#why-use-cmathlib)
- [Contributing](#contributing)
- [License](#license)

## Features

- **High-Performance Computations**: Leverages GPU acceleration for efficient large-scale computations.
- **Comprehensive Mathematical Operations**: Supports vector operations, matrix operations, quaternion operations, and transformations.
- **Auto-Tuning**: Dynamically selects the best kernel configurations based on input data and hardware.
- **Memory Management**: Efficient memory pooling and pinned memory for faster data transfers.
- **Random Number Generation**: High-quality random number generators for various distributions.
- **Scalability**: Designed to handle large datasets and complex computations.

## File Structure
mathlib/
├── include/
│ ├── IMathOperations.h // Interface for math operations
│ ├── MathOperationsFactory.h // Factory for creating math operations objects
│ ├── MemoryPool.h // Memory pool for efficient memory management
│ ├── Vector.h // Vector operations
│ ├── Matrix.h // Matrix operations
│ ├── Quaternion.h // Quaternion operations for 3D rotations
│ ├── Transform.h // Transformations (translation, rotation, scaling)
│ ├── Random.h // Random number generation
│ └── AutoTuner.h // Auto-tuning for kernel configurations
├── src/
│ ├── kernels/
│ │ ├── math_kernels.cu // CUDA kernels for basic math operations
│ │ ├── matrix_kernels.cu // CUDA kernels for matrix operations
│ │ ├── sparse_kernels.cu // CUDA kernels for sparse matrix operations
│ │ ├── vector_kernels.cu // CUDA kernels for vector operations
│ │ ├── quaternion_kernels.cu // CUDA kernels for quaternion operations
│ │ └── transform_kernels.cu // CUDA kernels for transformations
│ ├── CPUMathOperations.cpp // CPU implementation of math operations
│ ├── GPUMathOperations.cpp // GPU implementation of math operations
│ ├── MemoryPool.cpp // Implementation of memory pool
│ ├── AutoTuner.cpp // Implementation of auto-tuning
│ ├── Vector.cpp // Implementation of vector operations
│ ├── Matrix.cpp // Implementation of matrix operations
│ ├── Quaternion.cpp // Implementation of quaternion operations
│ ├── Transform.cpp // Implementation of transformations
│ └── Random.cpp // Implementation of random number generation
├── tests/
│ ├── test_math_operations.cpp // Unit tests for math operations
│ ├── test_performance.cpp // Performance tests and benchmarks
│ ├── test_vector.cpp // Unit tests for vector operations
│ ├── test_matrix.cpp // Unit tests for matrix operations
│ ├── test_quaternion.cpp // Unit tests for quaternion operations
│ ├── test_transform.cpp // Unit tests for transformations
│ └── test_random.cpp // Unit tests for random number generation
└── main.cpp // Example usage of the library


## Installation

1. **Clone the repository**:
    ```sh
    git clone https://github.com/beni001/CMathLib.git
    cd CMathLib
    ```

2. **Build the library**:
    ```sh
    mkdir build
    cd build
    cmake ..
    make
    ```

3. **Run the tests**:
    ```sh
    ./tests/test_math_operations
    ./tests/test_performance
    ./tests/test_vector
    ./tests/test_matrix
    ./tests/test_quaternion
    ./tests/test_transform
    ./tests/test_random
    ```

## Usage

### Example: Training a Neural Network

```cpp
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

Sample Use Cases
Game Development
Optimized math operations for game physics, character animation, and rendering.
3D Modeling Software
Fast calculations for mesh transformations and deformations.
Virtual Reality (VR) and Augmented Reality (AR)
Real-time calculations for spatial tracking and object placement.
Scientific Visualization
Efficient computations for large-scale data visualization.
Computer-Aided Design (CAD)
Precise calculations for engineering and architectural designs.
Machine Learning for Computer Vision
Optimized matrix operations for image processing and analysis.
Robotics
Real-time calculations for motion planning and kinematics.
Simulation Software
Efficient computations for physics-based simulations.
Graphics Rendering Engines
Core mathematical functions for rendering pipelines.
Animation Software
Predictive calculations for keyframe interpolation and physics-based animation.
Why Use CMathLib
Performance
CMathLib leverages GPU acceleration to perform large-scale computations efficiently. This makes it ideal for applications that require high computational power, such as training neural networks, real-time simulations, and rendering.
Comprehensive Functionality
The library provides a wide range of mathematical operations, including vector operations, matrix operations, quaternion operations, and transformations. This makes it versatile and suitable for various use cases.
Auto-Tuning
CMathLib includes an auto-tuning feature that dynamically selects the best kernel configurations based on input data and hardware. This ensures optimal performance for different use cases.
Memory Management
The library includes efficient memory pooling and pinned memory for faster data transfers. This reduces memory allocation overhead and improves performance.
Scalability
CMathLib is designed to handle large datasets and complex computations. This makes it suitable for applications that require processing large amounts of data, such as scientific visualization and machine learning.
Usability and Integration
CMathLib is easy to integrate into existing projects. The library provides a simple and intuitive interface for performing mathematical operations. The factory pattern is used to create math operations objects, making it easy to switch between CPU and GPU implementations.
Contributing
We welcome contributions to CMathLib! If you have any ideas, suggestions, or bug reports, please open an issue or submit a pull request on GitHub.
License
CMathLib is licensed under the MIT License. See the LICENSE file for more details.
