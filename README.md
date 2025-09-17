# Minimalist MNIST Inference Engine, Compare CPU vs GPU with CUDA and cuDNN Acceleration
**Project Purpose:** This project demonstrates the acceleration of a minimalist neural network for MNIST digit recognition using CUDA and the cuDNN library.

Following the instructions of this README file, you will be able to download the MNIST digits dataset, and build two programs, one meant to run completely on the CPU, and another meant to do the heavy stuff run on the GPU.

After both programs are finished, you can compare the test results for accuracy, and the execution time, typically being both metrics remarkably better in the program that runs on the GPU.  

**Goals:**
A special goal was to try the creation and training of a NeuralNetwork.
Although what's demonstrated is the **advantages of integrating GPU acceleration** for computationally intensive tasks in a simple deep learning model, **as compared to executing the training and inference in the CPU**.

# Architecture
The project implements a two-layer fully connected neural network (MLP) (same for both programs).
Note that computation layers are only one hidden and one output layer.

* **Input Layer:** Takes the flattened 28x28 MNIST images.
* **Hidden Layer:** A fully connected layer with a sigmoid activation function.
* **Output Layer:** A fully connected layer with a softmax activation for classification across 10 digits.

Both programs have a similar flow:
1. Load the MNIST digits dataset.
2. Initialize the neural network on the target (be it CPU or GPU).
3. Train the model for a specified number of epochs.
4. Evaluate its accuracy on the test set, printing the progress and final accuracy to the console as well execution time.

# GPU-oriented Implementation
The core computational parts of the network, specifically matrix multiplications and activation functions, are accelerated using:
* **CUDA:** For general-purpose GPU programming and memory management.
* **cuBLAS:** A library providing optimized basic linear algebra subprograms for matrix multiplications.
* **cuDNN:** A library providing highly tuned primitives for deep neural networks, used here for activation functions and bias addition.

Note that even in the GPU-oriented program the data loading and initial setup are handled on the host (CPU), while the forward and backward passes during training and the forward pass during inference are performed on the host for the CPU-oriented program and on the GPU for the GPU-oriented program.

# Building and Execution
The project is implemented in C++ with CUDA extensions. To build and run the project in a compatible environment, follow these steps:

1. **Download the MNIST Dataset:** The required MNIST dataset files need to be downloaded from a public mirror. You can simply run the command `make download_mnist`.

- **Note:** The programs expect to find these assets under the following path: `/content`, but the Makefile rule already takes care of placing the dataset there, and it will also extract the data set.

2. **Compile the Code:**
Check the Makefile.
- Run `make all`, which will build both programs, the one for the CPU and the one for the GPU.

3. **Execute the Programs:**
Run the compiled executables.
- Run `make run` to run both programs, firstly the CPU-oriented program and secondly the GPU-oriented program.

# System Requirements
Execution of the compiled code requires a CUDA-enabled GPU and compatible CUDA driver and runtime versions.
The GPU and NVIDIA driver **used during the development** were:
- NVIDIA GPU: GeForce RTX 5090.
- NVIDIA driver version: 575.64.03.
- CUDA version: 12.9.

# Full Build, Compilation and Run Example
```
# Clone the project:
git clone git@github.com:jquk/CUDA-practice-capstone.git
cd CUDA-practice-capstone

# download and extracts dataset:
make download-mnist

# build both programs for the cpu and the gpu:
make build-all

# runs both programs, passing them the params = {epochs, hidden_layers_size, learning_rate}
make run-all ARGS="5 128 0.001"

# And the final **output** from the cpu-version and the gpu-version programs should look like this:
root@3001e1f044d3:/app/CUDA-practice-capstone# ./bin/mnist_titest_on_cpu 5 128 0.01
Parameters accepted:
 - epochs = 5
 - hidden_layer_size = 128
 - learning_rate = 0.010000
Starting training...
Epoch 0 completed. Average loss: 0.619917
Epoch 1 completed. Average loss: 0.417883
Epoch 2 completed. Average loss: 0.379473
Epoch 3 completed. Average loss: 0.356101
Epoch 4 completed. Average loss: 0.338258
Training finished.
Test accuracy: 86.61%
Total execution time: 294.405 seconds
root@3001e1f044d3:/app/CUDA-practice-capstone# ./bin/mnist_titest_on_gpu 5 128 0.01
Parameters accepted:
 - epochs = 5
 - hidden_layer_size = 128
 - learning_rate = 0.010000
Starting GPU training...
Epoch 0 completed. Average loss: 0.616009
Epoch 1 completed. Average loss: 0.416489
Epoch 2 completed. Average loss: 0.378392
Epoch 3 completed. Average loss: 0.35467
Epoch 4 completed. Average loss: 0.336927
GPU Training finished.
GPU Test accuracy: 86.65%
Total execution time: 16.1407 seconds
root@3001e1f044d3:/app/CUDA-practice-capstone#
```