# An out-of-tree MLIR dialect for precision tuning
## Overview

This project provides an out-of-tree MLIR dialect designed for precision tuning. It includes custom operations, types, and transformations to facilitate floating-point to fixed-point transformation

## Features

- Custom MLIR dialect for precision tuning
- Operations for casting and arithmetic with precision control
- Integration with MLIR's pass infrastructure
- Example passes for lowering to arithmetic operations

## Getting Started

### Prerequisites

Install the following dependencies on you system:
- cmake
- ninja
- mold (or gold if you prefer. But mold is a faster linker, useful as llvm compilation is long)
- clang-format 19 or newer

Make sure that you have at least 110 GB free of disk space if you build llvm in debug mode.
You could also build llvm in release mode to occupy a lot less space, but it is not recommended if you will develop TAFFO itself.
The debug build itself occupies more than 60 GB and the debug installation occupies more than 40 GB.
Once llvm is installed, you can delete the debug build directory to reclaim its 60 GB of space.

### Cloning the repository

Clone and cd in the repository with:
  ```sh
  git clone https://github.com/your-repo/TAFFO-MLIR.git
  cd TAFFO-MLIR
  ```

### Building MLIR

Shallow clone the llvm repository at the specific know-working commit:
```
git clone --depth 1 --revision 1053047a4be7d1fece3adaf5e7597f838058c947 https://github.com/llvm/llvm-project.git
```

Build in debug mode with the commands below.
You can change it to release mode if you don't need to debug TAFFO, but it is not recommended if you will develop TAFFO itself

First create and cd the build directory:
```
cd llvm-project
mkdir build_debug
cd build_debug
```
Then compose the right cmake command based on your choices (so do not start next command straight up!):
```
cmake ../llvm -G Ninja \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_INSTALL_PREFIX=../install_debug \
    -DLLVM_ENABLE_PROJECTS="mlir;clang" \
    -DLLVM_TARGETS_TO_BUILD="host" \
    -DLLVM_INSTALL_UTILS=ON \
    -DLLVM_INCLUDE_BENCHMARKS=OFF \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++
```
Append this if you chose mold as linker:
```
    -DLLVM_USE_LINKER=mold
```
or this if you chose gold
```
  -DLLVM_USE_LINKER=gold
```
If your system doesn't have much available RAM, the safest way is to also append:
```
  -DLLVM_PARALLEL_LINK_JOBS=1
```
This last option limits the concurrent linker jobs to save RAM at the cost of a slower build.
You can also try with more link jobs like ```-DLLVM_PARALLEL_LINK_JOBS=2``` or omit it completely if you think you have enough RAM.
The worst it can happen is that the system kills your build because it has no more RAM and you have to rebuild by from scratch limiting the linker jobs more.
An llvm build with unlimited linker jobs can more than 16 GB of RAM depending on your linker (also quite a lot more).

Then start the build with:
```
ninja
```

And install with:
```
ninja install
```

Then cd back to the TAFFO repository root with:
```
cd ../..
```

### Building TAFFO

Configure and build TAFFO in debug mode.
Once again you could build TAFFO in release mode, but not recommended if you will develop TAFFO itself.
  ```sh
  mkdir build_debug
  cd build_debug
  cmake -G Ninja .. \
    -DCMAKE_BUILD_TYPE=Debug \
    -DLLVM_DIR=./llvm-project/install_debug/lib/cmake/llvm \
    -DMLIR_DIR=./llvm-project/install_debug/lib/cmake/mlir
  ninja
  cd ..
  ```

### Running Tests

To run the tests, use the following command:
```sh
cmake --build build_debug --target check
```
