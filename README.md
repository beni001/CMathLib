# CMathLib

CMathLib is a high-performance, GPU-accelerated math library designed for large-scale computations. It offers a comprehensive set of mathematical operations, including vector operations, matrix operations, quaternion operations, and transformations. CMathLib is optimized for various use cases such as game development, 3D modeling, virtual reality (VR) and augmented reality (AR), scientific visualization, computer-aided design (CAD), machine learning, robotics, simulation software, graphics rendering engines, and animation software.

## Table of Contents

- [Features](#features)
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
// src/main.cpp//


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
