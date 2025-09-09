#include <chrono>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <numeric>
#include <random> // For better random initialization

#include "../lib/nn_cpu.h"
#include "../lib/helpers.h"

int main() {
    // Start measuring time
    auto start = std::chrono::high_resolution_clock::now();

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

    // Define network parameters
    int input_size = train_images[0].size();
    int hidden_size = 128; // Example hidden layer size
    int output_size = 10; // 0-9 digits

    // Create neural network
    NeuralNetworkCPU model(input_size, hidden_size, output_size);

    // Training parameters
    int epochs = 5; // Example number of epochs
    double learning_rate = 0.1; // Example learning rate

    // Training loop
    std::cout << "Starting training..." << std::endl;
    for (int epoch = 0; epoch < epochs; ++epoch) {
        double total_loss = 0.0;
        for (size_t i = 0; i < train_images.size(); ++i) {
            // Forward pass
            std::vector<double> hidden_output(model.bias1.size());
            for (size_t j = 0; j < model.bias1.size(); ++j) {
                double sum = model.bias1[j];
                for (size_t k = 0; k < train_images[i].size(); ++k) {
                    sum += train_images[i][k] * model.weights1[k][j];
                }
                hidden_output[j] = model.sigmoid(sum);
            }

            std::vector<double> output = model.forward(train_images[i]);

            // Calculate loss (Cross-entropy)
            total_loss -= log(output[train_labels[i]]);

            // Backward pass and weight update
            model.backward(train_images[i], hidden_output, output, train_labels[i], learning_rate);
        }
        std::cout << "Epoch " << epoch << " completed. Average loss: " << total_loss / train_images.size() << std::endl;
    }
    std::cout << "Training finished." << std::endl;

    // Inference and evaluation
    int correct_predictions = 0;
    for (size_t i = 0; i < test_images.size(); ++i) {
        int predicted_label = model.predict(test_images[i]);
        if (predicted_label == test_labels[i]) {
            correct_predictions++;
        }
    }

    double accuracy = (double)correct_predictions / test_images.size();
    std::cout << "Test accuracy: " << accuracy * 100.0 << "%" << std::endl;
    
    // End measuring time, calculate duration and print it on the terminal
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Total execution time: " << duration.count() << " seconds" << std::endl;

    return 0;
}