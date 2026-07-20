# MLCC-GPT

A repository for training and running generative pre-trained transformers (GPTs) built in a custom domain-specific language.

This project implements a GPT architecture (inspired by nanoGPT) but, instead of relying on Python and PyTorch, it is written entirely in a custom language called **MLCC**. It compiles directly to a native executable for execution.

## What is MLCC?

MLCC (Machine Learning Compiler Collection) is a custom compiler infrastructure designed to parse a high-level, human-readable language (with `.mlcc` extensions) and compile it down to native machine code. 

**Under the hood, MLCC uses:**
- **Flex & Bison** for parsing the AST.
- **MLIR & LLVM** for lowering the high-level syntax into LLVM IR and then to a native executable.
- **LibTorch C++ Runtime** for dispatching heavy tensor operations. The runtime bridge seamlessly interfaces the compiled LLVM code with PyTorch's native C++ backend, automatically utilizing Apple Silicon MPS (Metal Performance Shaders) or CPU depending on availability.

## Repository Structure

- `src/` - The core MLCC source files defining the GPT architecture, training, generation, and inference logic.
- `Data/` - Datasets (OpenWebText, Shakespeare) and python scripts for tokenization and binary preparation.
- `compiler/` - **(Required Submodule)** The C++ source code for the MLCC compiler. You must fetch this separately (see below).

## Install

Before building the project, ensure you have the required dependencies:
- **LLVM & MLIR** (Required for the MLCC compiler backend)
- **LibTorch (C++)** (Required for the runtime bridge)
- **CMake** (For building the compiler)
- Python 3 & `tiktoken` (Only needed for dataset preparation)

### 1. Get the MLCC Compiler Submodule
Because the MLCC compiler is maintained in a separate repository, you need to pull it into the `compiler/` directory. If it's configured as a submodule:
```bash
git submodule update --init --recursive
```
*(If you do not have it as a submodule, you will need to clone the MLCC compiler repository directly into the `compiler/` directory).*

### 2. Build the Compiler
```bash
cd compiler
mkdir build && cd build
cmake ..
make -j
cd ../..
```

## Quick Start

### 1. Prepare the Dataset
We use the OpenWebText dataset (or Shakespeare for testing). The dataset needs to be downloaded, tokenized, and saved as a flat binary file that the MLCC runtime can stream.
```bash
python3 Data/openwebtext/prepare.py
```

### 2. Compile the GPT Model
Use the freshly built MLCC compiler to compile the `.mlcc` source files into a native executable.
```bash
./compiler/build/mlcc src/main.mlcc
```
This parses the model architecture, training loop, and inference loop into MLIR, lowers it to LLVM, and produces a native executable (typically `a.out`).

### 3. Run Training or Inference
Execute the native binary! The MLCC runtime will automatically allocate tensors, build the computational graph, and execute the model natively.
```bash
./a.out
```

## What it does

When you run the compiled binary, it:
1. Loads the tokenized dataset from disk (e.g., `train.bin` and `val.bin`).
2. Initializes the GPT model weights via the compiled MLCC binary.
3. Runs the training loop, automatically computing cross-entropy loss, applying gradients, and printing the step performance (loss and speed).
4. Periodically saves the model checkpoints.
5. In inference mode, it drops into an interactive loop where you can prompt the compiled GPT model and it streams tokens back natively.

## License

MIT
