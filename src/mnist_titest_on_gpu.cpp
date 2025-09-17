#include <chrono>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <numeric>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cudnn.h>
#include <cublas_v2.h>


#include "../lib/nn_gpu.h"
#include "../lib/gpu_helpers.cuh"
#include "../lib/helpers.h"

int main(int argc, char *argv[]) {
    // Start measuring time
    auto start = std::chrono::high_resolution_clock::now();

    // Declare training parameters, some are defined now, some (input_layer_size) later
    int input_layer_size; // This parameter will be defined later on, based on the size of each of the images in the training data set
    int hidden_layer_size;
    int output_layer_size = 10; // 0-9 digits
    int epochs;
    double learning_rate; // Example learning rate (0.1 might be too high, better try with 0.01 or even 0.001), or implement 'learning rate decay'.

    // Validate program input parameters given (0 if there are NO ERRORS)
    if(validate_program_input_parameters(argc, (const char**)argv, &epochs, &hidden_layer_size, &learning_rate)) {
        return 1;
    }

    // Load training data
    std::vector<std::vector<double>> train_images = read_mnist_images(DATA_DIR "train-images-idx3-ubyte");
    std::vector<int> train_labels = read_mnist_labels(DATA_DIR "train-labels-idx1-ubyte");

    // Load testing data
    std::vector<std::vector<double>> test_images = read_mnist_images(DATA_DIR "t10k-images-idx3-ubyte");
    std::vector<int> test_labels = read_mnist_labels(DATA_DIR "t10k-labels-idx1-ubyte");

    if (train_images.empty() || train_labels.empty() || test_images.empty() || test_labels.empty()) {
        std::cerr << "Failed to load MNIST data." << std::endl;
        return 1;
    }

    // Define network parameter input_layer_size, based on the expected size for each training image
    input_layer_size = train_images[0].size();

    // Create neural network on GPU
    NeuralNetworkGPU model(input_layer_size, hidden_layer_size, output_layer_size);

    // Training loop
    std::cout << "Starting GPU training..." << std::endl;
    for (int epoch = 0; epoch < epochs; ++epoch) {
        double total_loss = 0.0;
        for (size_t i = 0; i < train_images.size(); ++i) {
            // Transfer input data to device
            host_to_device(train_images[i].data(), model.d_input, input_layer_size);

            // Forward pass on GPU
            model.forward_gpu(model.d_input);

            // Calculate loss (Cross-entropy) on host after transferring output
            std::vector<double> h_output(output_layer_size);
            device_to_host(model.d_output, h_output.data(), output_layer_size);
            total_loss -= log(h_output[train_labels[i]]);

            // Backward pass on GPU
            model.backward_gpu(model.d_input, train_labels[i], learning_rate);
        }
        std::cout << "Epoch " << epoch << " completed. Average loss: " << total_loss / train_images.size() << std::endl;
    }
    std::cout << "GPU Training finished." << std::endl;

    // Inference and evaluation on GPU
    int correct_predictions = 0;
    for (size_t i = 0; i < test_images.size(); ++i) {
        // Transfer test image to device
        host_to_device(test_images[i].data(), model.d_input, input_layer_size);

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