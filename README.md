# Minimalist MNIST Inference Engine, Compare CPU vs GPU with CUDA and cuDNN Acceleration
**Project Purpose:** This project demonstrates the acceleration of a minimalist neural network for MNIST digit recognition using CUDA and the cuDNN library.

The goal is to illustrate the process and **advantages of integrating GPU acceleration** for computationally intensive tasks in a simple deep learning model, **as compared to executing the training and inference in the CPU**.

*Upon execution of each program, the test accuracy and execution time is shown, thus it's possible to compare accuracy and execution time results for the CPU and GPU.*

# Architecture
The project implements a two-layer fully connected neural network (MLP).
Note that computation layers are only one hidden and one output layer.

* **Input Layer:** Takes the flattened 28x28 MNIST images.
* **Hidden Layer:** A fully connected layer with a sigmoid activation function.
* **Output Layer:** A fully connected layer with a softmax activation for classification across 10 digits.

# GPU-oriented Implementation
The core computational parts of the network, specifically matrix multiplications and activation functions, are accelerated using:
* **CUDA:** For general-purpose GPU programming and memory management.
* **cuBLAS:** A library providing optimized basic linear algebra subprograms for matrix multiplications.
* **cuDNN:** A library providing highly tuned primitives for deep neural networks, used here for activation functions and bias addition.

Note that even in the GPU-oriented program the data loading and initial setup are handled on the host (CPU), while the forward and backward passes during training and the forward pass during inference are performed on the device (GPU).

# Building and Execution
The project is implemented in C++ with CUDA extensions. To build and run the project in a compatible environment, follow these steps:

1. **Download the MNIST Dataset:** The required MNIST dataset files (train-images-idx3-ubyte, train-labels-idx1-ubyte, t10k-images-idx3-ubyte, t10k-labels-idx1-ubyte) need to be downloaded. The provided code includes wget commands to download these from a public mirror.
```
wget https://storage.googleapis.com/tensorflow/tf-keras-datasets/train-images-idx3-ubyte.gz
wget https://storage.googleapis.com/tensorflow/tf-keras-datasets/train-labels-idx1-ubyte.gz
wget https://storage.googleapis.com/tensorflow/tf-keras-datasets/t10k-images-idx3-ubyte.gz
wget https://storage.googleapis.com/tensorflow/tf-keras-datasets/t10k-labels-idx1-ubyte.gz

gunzip train-images-idx3-ubyte.gz
gunzip train-labels-idx1-ubyte.gz
gunzip t10k-images-idx3-ubyte.gz
gunzip t10k-labels-idx1-ubyte.gz
```

**Note:** The programs expect to find these assets under the following path: `/content`.

2. **Compile the Code:**
Check the Makefile.
- Run `make mnist_titest` to build the program that does Training Inference and Test on the CPU.
- Run `make mnist_titestongpu` to build the program that does Training Inference and Test on the GPU.

3. **Execute the Programs:**
Run the compiled executables.
- Run `./mnist_titest` to run the program that uses the CPU.
- Run `./mnist_titest_on_gpu` to run the program that uses the GPU.

Both programs will load the dataset, initialize the neural network on the CPU/GPU, train the model for a specified number of epochs, and then evaluate its accuracy on the test set, printing the progress and final accuracy to the console as well execution time.

# System Requirements
Execution of the compiled code requires a CUDA-enabled GPU and compatible CUDA driver and runtime versions.
The GPU and NVIDIA driver used for the development are:
- NVIDIA GPU: GeForce RTX 5090.
- NVIDIA driver version: 575.64.03.
- CUDA version: 12.9.