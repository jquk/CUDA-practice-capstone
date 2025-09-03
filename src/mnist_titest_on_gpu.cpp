#include <chrono>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <numeric>
#include <random>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cudnn.h>
#include <cublas_v2.h>

// Helper function for checking CUDA errors
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__    \
                      << " code: " << cudaGetErrorString(err) << std::endl; \
            exit(1);                                                        \
        }                                                                   \
    } while (0)

// Helper function for checking cuDNN errors
#define CUDNN_CHECK(call)                                                     \
    do {                                                                      \
        cudnnStatus_t status = call;                                          \
        if (status != CUDNN_STATUS_SUCCESS) {                                 \
            std::cerr << "cuDNN error at " << __FILE__ << ":" << __LINE__     \
                      << " code: " << cudnnGetErrorString(status) << std::endl; \
            exit(1);                                                          \
        }                                                                     \
    } while (0)

// Helper function for checking cuBLAS errors
#define CUBLAS_CHECK(call)                                                    \
    do {                                                                      \
        cublasStatus_t status = call;                                         \
        if (status != CUBLAS_STATUS_SUCCESS) {                                \
            std::cerr << "cuBLAS error at " << __FILE__ << ":" << __LINE__    \
                      << " code: " << status << std::endl; \
            exit(1);                                                          \
        }                                                                     \
    } while (0)


// Function to read MNIST image file
std::vector<std::vector<double>> read_mnist_images(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error opening image file: " << filename << std::endl;
        return {};
    }

    int magic_number = 0;
    int number_of_images = 0;
    int n_rows = 0;
    int n_cols = 0;

    // Read magic number
    file.read((char*)&magic_number, sizeof(magic_number));
    magic_number = __builtin_bswap32(magic_number); // Convert big-endian to little-endian

    // Read number of images
    file.read((char*)&number_of_images, sizeof(number_of_images));
    number_of_images = __builtin_bswap32(number_of_images);

    // Read number of rows
    file.read((char*)&n_rows, sizeof(n_rows));
    n_rows = __builtin_bswap32(n_rows);

    // Read number of columns
    file.read((char*)&n_cols, sizeof(n_cols));
    n_cols = __builtin_bswap32(n_cols);

    std::vector<std::vector<double>> images;
    images.reserve(number_of_images);

    for (int i = 0; i < number_of_images; ++i) {
        std::vector<double> image(n_rows * n_cols);
        for (int j = 0; j < n_rows * n_cols; ++j) {
            unsigned char pixel = 0;
            file.read((char*)&pixel, sizeof(pixel));
            image[j] = static_cast<double>(pixel) / 255.0; // Normalize pixel values
        }
        images.push_back(image);
    }

    file.close();
    return images;
}

// Function to read MNIST label file
std::vector<int> read_mnist_labels(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error opening label file: " << filename << std::endl;
        return {};
    }

    int magic_number = 0;
    int number_of_items = 0;

    // Read magic number
    file.read((char*)&magic_number, sizeof(magic_number));
    magic_number = __builtin_bswap32(magic_number);

    // Read number of items
    file.read((char*)&number_of_items, sizeof(number_of_items));
    number_of_items = __builtin_bswap32(number_of_items);

    std::vector<int> labels;
    labels.reserve(number_of_items);

    for (int i = 0; i < number_of_items; ++i) {
        unsigned char label = 0;
        file.read((char*)&label, sizeof(label));
        labels.push_back(static_cast<int>(label));
    }

    file.close();
    return labels;
}

// Function to allocate device memory
template<typename T>
T* allocate_device_memory(size_t size) {
    T* device_ptr;
    CUDA_CHECK(cudaMalloc(&device_ptr, size * sizeof(T)));
    return device_ptr;
}

// Function to deallocate device memory
template<typename T>
void free_device_memory(T* device_ptr) {
    CUDA_CHECK(cudaFree(device_ptr));
}

// Function to transfer data from host to device
template<typename T>
void host_to_device(const T* host_ptr, T* device_ptr, size_t size) {
    CUDA_CHECK(cudaMemcpy(device_ptr, host_ptr, size * sizeof(T), cudaMemcpyHostToDevice));
}

// Function to transfer data from device to host
template<typename T>
void device_to_host(const T* device_ptr, T* host_ptr, size_t size) {
    CUDA_CHECK(cudaMemcpy(host_ptr, device_ptr, size * sizeof(T), cudaMemcpyDeviceToHost));
}

// Define the structure for a simple neural network model (MLP) with GPU support
struct NeuralNetworkGPU {
    // Device pointers for weights, biases, and intermediate results
    double* d_weights1;
    double* d_bias1;
    double* d_weights2;
    double* d_bias2;

    // Device pointers for intermediate results
    double* d_hidden_output;
    double* d_output;
    double* d_output_errors;
    double* d_hidden_errors;
    double* d_input; // Device pointer for input data

    int input_size;
    int hidden_size;
    int output_size;

    // cuDNN handle and descriptors
    cudnnHandle_t cudnn;
    cudnnTensorDescriptor_t input_descriptor;
    cudnnTensorDescriptor_t bias1_descriptor;
    cudnnTensorDescriptor_t hidden_output_descriptor;
    cudnnActivationDescriptor_t sigmoid_activation_descriptor;
    cudnnTensorDescriptor_t bias2_descriptor;
    cudnnTensorDescriptor_t output_descriptor;

    // cuBLAS handle
    cublasHandle_t cublas;


    // Constructor
    NeuralNetworkGPU(int input_size, int hidden_size, int output_size) :
        input_size(input_size), hidden_size(hidden_size), output_size(output_size) {

        // Allocate device memory
        d_weights1 = allocate_device_memory<double>(input_size * hidden_size);
        d_bias1 = allocate_device_memory<double>(hidden_size);
        d_weights2 = allocate_device_memory<double>(hidden_size * output_size);
        d_bias2 = allocate_device_memory<double>(output_size);

        // Allocate device memory for intermediate results
        d_hidden_output = allocate_device_memory<double>(hidden_size);
        d_output = allocate_device_memory<double>(output_size);
        d_output_errors = allocate_device_memory<double>(output_size);
        d_hidden_errors = allocate_device_memory<double>(hidden_size);
        d_input = allocate_device_memory<double>(input_size);


        // Initialize cuDNN
        CUDNN_CHECK(cudnnCreate(&cudnn));

        // Initialize cuBLAS
        CUBLAS_CHECK(cublasCreate(&cublas));

        // Initialize cuDNN descriptors
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&input_descriptor));
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&bias1_descriptor));
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&hidden_output_descriptor));
        CUDNN_CHECK(cudnnCreateActivationDescriptor(&sigmoid_activation_descriptor));
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&bias2_descriptor));
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&output_descriptor));


        // Set cuDNN tensor descriptors dimensions for AddTensor (batch size 1)
        CUDNN_CHECK(cudnnSetTensor4dDescriptor(input_descriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_DOUBLE, 1, input_size, 1, 1));
        CUDNN_CHECK(cudnnSetTensor4dDescriptor(bias1_descriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_DOUBLE, 1, hidden_size, 1, 1));
        CUDNN_CHECK(cudnnSetTensor4dDescriptor(hidden_output_descriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_DOUBLE, 1, hidden_size, 1, 1));
        CUDNN_CHECK(cudnnSetTensor4dDescriptor(bias2_descriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_DOUBLE, 1, output_size, 1, 1));
        CUDNN_CHECK(cudnnSetTensor4dDescriptor(output_descriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_DOUBLE, 1, output_size, 1, 1));


        // Set cuDNN activation descriptor
        CUDNN_CHECK(cudnnSetActivationDescriptor(sigmoid_activation_descriptor, CUDNN_ACTIVATION_SIGMOID, CUDNN_NOT_PROPAGATE_NAN, 0.0));


        // Initialize weights and biases on the device with random values
        std::vector<double> h_weights1(input_size * hidden_size);
        std::vector<double> h_bias1(hidden_size);
        std::vector<double> h_weights2(hidden_size * output_size);
        std::vector<double> h_bias2(output_size);

        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<> d(0, 0.01);

        for (int i = 0; i < input_size * hidden_size; ++i) h_weights1[i] = d(gen);
        for (int i = 0; i < hidden_size; ++i) h_bias1[i] = d(gen);
        for (int i = 0; i < hidden_size * output_size; ++i) h_weights2[i] = d(gen);
        for (int i = 0; i < output_size; ++i) h_bias2[i] = d(gen);

        host_to_device(h_weights1.data(), d_weights1, input_size * hidden_size);
        host_to_device(h_bias1.data(), d_bias1, hidden_size);
        host_to_device(h_weights2.data(), d_weights2, hidden_size * output_size);
        host_to_device(h_bias2.data(), d_bias2, output_size);
    }

    // Destructor to free device memory and cuDNN/cuBLAS handles and descriptors
    ~NeuralNetworkGPU() {
        free_device_memory(d_weights1);
        free_device_memory(d_bias1);
        free_device_memory(d_weights2);
        free_device_memory(d_bias2);
        free_device_memory(d_hidden_output);
        free_device_memory(d_output);
        free_device_memory(d_output_errors);
        free_device_memory(d_hidden_errors);
        free_device_memory(d_input);
        CUDNN_CHECK(cudnnDestroyTensorDescriptor(input_descriptor));
        CUDNN_CHECK(cudnnDestroyTensorDescriptor(bias1_descriptor));
        CUDNN_CHECK(cudnnDestroyTensorDescriptor(hidden_output_descriptor));
        CUDNN_CHECK(cudnnDestroyActivationDescriptor(sigmoid_activation_descriptor));
        CUDNN_CHECK(cudnnDestroyTensorDescriptor(bias2_descriptor));
        CUDNN_CHECK(cudnnDestroyTensorDescriptor(output_descriptor));
        CUDNN_CHECK(cudnnDestroy(cudnn));
        CUBLAS_CHECK(cublasDestroy(cublas));
    }

    // Forward pass on GPU using cuBLAS and cuDNN
    void forward_gpu(const double* d_input_data) {
        double alpha = 1.0, beta = 0.0;

        // Input to Hidden Layer (Matrix Multiplication)
        // d_hidden_output = d_weights1^T * d_input_data
        // Matrix A: d_weights1 (input_size x hidden_size), transposed -> (hidden_size x input_size)
        // Matrix B: d_input_data (input_size x 1)
        // Matrix C: d_hidden_output (hidden_size x 1)
        CUBLAS_CHECK(cublasDgemm(cublas,
                                 CUBLAS_OP_T, // Transpose A (weights1)
                                 CUBLAS_OP_N, // No transpose B (input_data)
                                 hidden_size, // M: rows of op(A) and C
                                 1,           // N: columns of op(B) and C
                                 input_size,  // K: columns of op(A) and rows of op(B)
                                 &alpha,      // alpha
                                 d_weights1,  // A
                                 input_size,  // lda: leading dimension of A
                                 d_input_data, // B
                                 input_size,  // ldb: leading dimension of B
                                 &beta,       // beta
                                 d_hidden_output, // C
                                 hidden_size)); // ldc: leading dimension of C


        // Add bias to hidden layer output
        CUDNN_CHECK(cudnnAddTensor(cudnn, &alpha, bias1_descriptor, d_bias1, &alpha, hidden_output_descriptor, d_hidden_output));

        // Sigmoid activation for hidden layer
        CUDNN_CHECK(cudnnActivationForward(cudnn, sigmoid_activation_descriptor, &alpha, hidden_output_descriptor, d_hidden_output, &beta, hidden_output_descriptor, d_hidden_output));

        // Hidden to Output Layer (Matrix Multiplication)
        // d_output = d_weights2^T * d_hidden_output
        // Matrix A: d_weights2 (hidden_size x output_size), transposed -> (output_size x hidden_size)
        // Matrix B: d_hidden_output (hidden_size x 1)
        // Matrix C: d_output (output_size x 1)
        CUBLAS_CHECK(cublasDgemm(cublas,
                                 CUBLAS_OP_T, // Transpose A (weights2)
                                 CUBLAS_OP_N, // No transpose B (hidden_output)
                                 output_size, // M
                                 1,           // N
                                 hidden_size, // K
                                 &alpha,      // alpha
                                 d_weights2,  // A
                                 hidden_size, // lda
                                 d_hidden_output, // B
                                 hidden_size, // ldb
                                 &beta,       // beta
                                 d_output, // C
                                 output_size)); // ldc


        // Add bias to output layer input
        CUDNN_CHECK(cudnnAddTensor(cudnn, &alpha, bias2_descriptor, d_bias2, &alpha, output_descriptor, d_output));

        // Softmax activation for output layer
        CUDNN_CHECK(cudnnSoftmaxForward(cudnn, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL, &alpha, output_descriptor, d_output, &beta, output_descriptor, d_output));
    }

    // Backward pass on GPU using cuBLAS and cuDNN
    void backward_gpu(const double* d_input_data, int target_label, double learning_rate) {
        double alpha = 1.0, beta = 0.0;
        double alpha_lr = learning_rate;

        // Calculate output errors (using a simple kernel or host logic and transfer)
        // For simplicity, we'll use host logic and transfer for the target label comparison
        std::vector<double> h_output(output_size);
        device_to_host(d_output, h_output.data(), output_size);

        std::vector<double> h_output_errors(output_size);
        for (int i = 0; i < output_size; ++i) {
            h_output_errors[i] = (i == target_label ? 1.0 : 0.0) - h_output[i];
        }
        host_to_device(h_output_errors.data(), d_output_errors, output_size);

        // Backpropagate errors to hidden layer
        // d_hidden_errors = d_weights2 * d_output_errors
        // Matrix A: d_weights2 (hidden_size x output_size)
        // Matrix B: d_output_errors (output_size x 1)
        // Matrix C: d_hidden_errors (hidden_size x 1)
        CUBLAS_CHECK(cublasDgemm(cublas,
                                 CUBLAS_OP_N, // No transpose A (weights2)
                                 CUBLAS_OP_N, // No transpose B (output_errors)
                                 hidden_size, // M
                                 1,           // N
                                 output_size, // K
                                 &alpha,      // alpha
                                 d_weights2,  // A
                                 hidden_size, // lda
                                 d_output_errors, // B
                                 output_size, // ldb
                                 &beta,       // beta
                                 d_hidden_errors, // C
                                 hidden_size)); // ldc


        // Apply sigmoid derivative to hidden errors
        // The sigmoid derivative requires the original hidden layer output
        CUDNN_CHECK(cudnnActivationBackward(cudnn, sigmoid_activation_descriptor, &alpha, hidden_output_descriptor, d_hidden_output, hidden_output_descriptor, d_hidden_errors, hidden_output_descriptor, d_hidden_errors, &beta, hidden_output_descriptor, d_hidden_errors));


        // Update weights and biases for output layer
        // Gradients for weights2: learning_rate * d_output_errors * d_hidden_output^T
        // Matrix A: d_output_errors (output_size x 1)
        // Matrix B: d_hidden_output (hidden_size x 1), transposed -> (1 x hidden_size)
        // Matrix C: d_weights2 (hidden_size x output_size), transposed -> (output_size x hidden_size) - accumulate here
        // We need to update d_weights2 (hidden_size x output_size), so A is d_hidden_output (hidden_size x 1), B is d_output_errors (output_size x 1), transposed -> (1 x output_size)
        // C = alpha * A * B + beta * C
        CUBLAS_CHECK(cublasDgemm(cublas,
                                 CUBLAS_OP_N, // No transpose A (hidden_output)
                                 CUBLAS_OP_T, // Transpose B (output_errors)
                                 hidden_size, // M
                                 output_size, // N
                                 1,           // K
                                 &alpha_lr,   // alpha (learning_rate)
                                 d_hidden_output, // A (hidden_size x 1)
                                 hidden_size, // lda
                                 d_output_errors, // B (output_size x 1)
                                 output_size, // ldb
                                 &alpha,      // beta (accumulate)
                                 d_weights2,  // C (hidden_size x output_size)
                                 hidden_size)); // ldc


        // Gradients for bias2: learning_rate * d_output_errors (summed over batch, but batch size is 1 here)
        CUDNN_CHECK(cudnnAddTensor(cudnn, &alpha_lr, output_descriptor, d_output_errors, &alpha, bias2_descriptor, d_bias2));


        // Update weights and biases for hidden layer
        // Gradients for weights1: learning_rate * d_hidden_errors * d_input_data^T
        // Matrix A: d_hidden_errors (hidden_size x 1)
        // Matrix B: d_input_data (input_size x 1), transposed -> (1 x input_size)
        // Matrix C: d_weights1 (input_size x hidden_size), transposed -> (hidden_size x input_size) - accumulate here
        // We need to update d_weights1 (input_size x hidden_size), so A is d_input_data (input_size x 1), B is d_hidden_errors (hidden_size x 1), transposed -> (1 x hidden_size)
         CUBLAS_CHECK(cublasDgemm(cublas,
                                 CUBLAS_OP_N, // No transpose A (input_data)
                                 CUBLAS_OP_T, // Transpose B (hidden_errors)
                                 input_size, // M
                                 hidden_size, // N
                                 1,           // K
                                 &alpha_lr,   // alpha (learning_rate)
                                 d_input_data, // A (input_size x 1)
                                 input_size, // lda
                                 d_hidden_errors, // B (hidden_size x 1)
                                 hidden_size, // ldb
                                 &alpha,      // beta (accumulate)
                                 d_weights1,  // C (input_size x hidden_size)
                                 input_size)); // ldc

        // Gradients for bias1: learning_rate * d_hidden_errors (summed over batch, but batch size is 1 here)
        CUDNN_CHECK(cudnnAddTensor(cudnn, &alpha_lr, hidden_output_descriptor, d_hidden_errors, &alpha, bias1_descriptor, d_bias1));
    }

    // Predict on GPU
    int predict_gpu(const double* d_image) {
        forward_gpu(d_image); // Perform forward pass

        std::vector<double> h_output(output_size);
        device_to_host(d_output, h_output.data(), output_size);

        int predicted_label = 0;
        double max_output = -1.0;
        for (int j = 0; j < output_size; ++j) {
            if (h_output[j] > max_output) {
                max_output = h_output[j];
                predicted_label = j;
            }
        }
        return predicted_label;
    }
};

int main() {
    // Start measuring time
    auto start = std::chrono::high_resolution_clock::now();

    // Load training data
    std::vector<std::vector<double>> train_images = read_mnist_images("/content/train-images-idx3-ubyte");
    std::vector<int> train_labels = read_mnist_labels("/content/train-labels-idx1-ubyte");

    // Load testing data
    std::vector<std::vector<double>> test_images = read_mnist_images("/content/t10k-images-idx3-ubyte");
    std::vector<int> test_labels = read_mnist_labels("/content/t10k-labels-idx1-ubyte");

    if (train_images.empty() || train_labels.empty() || test_images.empty() || test_labels.empty()) {
        std::cerr << "Failed to load MNIST data." << std::endl;
        return 1;
    }

    // Define network parameters
    int input_size = train_images[0].size();
    int hidden_size = 128; // Example hidden layer size
    int output_size = 10; // 0-9 digits

    // Create neural network on GPU
    NeuralNetworkGPU model(input_size, hidden_size, output_size);

    // Training parameters
    int epochs = 5; // Example number of epochs
    double learning_rate = 0.01; // Adjusted learning rate for GPU training

    // Training loop
    std::cout << "Starting GPU training..." << std::endl;
    for (int epoch = 0; epoch < epochs; ++epoch) {
        double total_loss = 0.0;
        for (size_t i = 0; i < train_images.size(); ++i) {
            // Transfer input data to device
            host_to_device(train_images[i].data(), model.d_input, input_size);

            // Forward pass on GPU
            model.forward_gpu(model.d_input);

            // Calculate loss (Cross-entropy) on host after transferring output
            std::vector<double> h_output(output_size);
            device_to_host(model.d_output, h_output.data(), output_size);
            total_loss -= log(h_output[train_labels[i]]);

            // Backward pass on GPU
            model.backward_gpu(model.d_input, train_labels[i], learning_rate);
        }
        std::cout << "Epoch " << epoch + 1 << " completed. Average loss: " << total_loss / train_images.size() << std::endl;
    }
    std::cout << "GPU Training finished." << std::endl;

    // Inference and evaluation on GPU
    int correct_predictions = 0;
    for (size_t i = 0; i < test_images.size(); ++i) {
        // Transfer test image to device
        host_to_device(test_images[i].data(), model.d_input, input_size);

        // Predict on GPU
        int predicted_label = model.predict_gpu(model.d_input);

        if (predicted_label == test_labels[i]) {
            correct_predictions++;
        }
    }

    double accuracy = (double)correct_predictions / test_images.size();
    std::cout << "GPU Test accuracy: " << accuracy * 100.0 << "%" << std::endl;

    // End measuring time, calculate duration and print it on the terminal
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Total execution time: " << duration.count() << " seconds" << std::endl;

    return 0;
}