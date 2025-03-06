# Exploring Sparse Matrices

## Overview
This repository focuses on efficient sparse matrix multiplication using High-Performance Computing (HPC) techniques. Sparse matrices, which contain a high number of zero elements, are commonly used in scientific computing, machine learning, and engineering applications. The goal of this project is to optimize sparse matrix operations using parallel computing frameworks such as CUDA and OpenMP.

## Features
- **Sparse Matrix Multiplication**: Implementations using different parallel computing techniques.
- **CUDA Implementation**: GPU-accelerated computations using NVIDIA CUDA.
- **OpenMP Implementation**: Multi-threaded CPU-based sparse matrix operations.
- **Performance Benchmarking**: Compare execution times and efficiency of different approaches.

## Installation
### Prerequisites
- C++ compiler (GCC/Clang/MSVC)
- NVIDIA CUDA Toolkit (for CUDA implementation)
- OpenMP support (for CPU-based parallelism)
- CMake (for building the project)

### Build and Run
1. Clone the repository:
   ```bash
   git clone https://github.com/NotCleo/exploring_sparse_matrices.git
   cd exploring_sparse_matrices
   ```
2. Compile the project:
   ```bash
   mkdir build && cd build
   cmake ..
   make
   ```
3. Run the program:
   ```bash
   ./sparse_matrix_mult
   ```

## Usage
Modify the input matrices in the source code or load them from files to test different matrix operations. You can benchmark different implementations by comparing execution times.
