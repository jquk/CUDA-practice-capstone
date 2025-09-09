#include <chrono>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <numeric>
#include <random> // For better random initialization

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

// Define the structure for a simple neural network model (MLP)
struct NeuralNetwork {
    std::vector<std::vector<double>> weights1;
    std::vector<double> bias1;
    std::vector<std::vector<double>> weights2;
    std::vector<double> bias2;

    // Constructor
    NeuralNetwork(int input_size, int hidden_size, int output_size) {
        // Initialize weights and biases
        weights1.resize(input_size, std::vector<double>(hidden_size));
        bias1.resize(hidden_size);
        weights2.resize(hidden_size, std::vector<double>(output_size));
        bias2.resize(output_size);

        // Initialize weights and biases with small random values using a better approach
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<> d(0, 0.01); // Using a normal distribution with small standard deviation

        for (int i = 0; i < input_size; ++i) {
            for (int j = 0; j < hidden_size; ++j) {
                weights1[i][j] = d(gen);
            }
        }
        for (int i = 0; i < hidden_size; ++i) {
            bias1[i] = d(gen);
        }
        for (int i = 0; i < hidden_size; ++i) {
            for (int j = 0; j < output_size; ++j) {
                weights2[i][j] = d(gen);
            }
        }
        for (int i = 0; i < output_size; ++i) {
            bias2[i] = d(gen);
        }
    }

    // Sigmoid activation function
    double sigmoid(double x) {
        return 1.0 / (1.0 + exp(-x));
    }

    // Derivative of sigmoid
    double sigmoid_derivative(double x) {
        return x * (1.0 - x);
    }

    // Softmax activation function
    std::vector<double> softmax(const std::vector<double>& input) {
        std::vector<double> output(input.size());
        double sum_exp = 0.0;
        double max_val = input[0];
        for (size_t i = 1; i < input.size(); ++i) {
            if (input[i] > max_val) {
                max_val = input[i];
            }
        }

        for (double val : input) {
            sum_exp += exp(val - max_val); // Subtract max for numerical stability
        }
        for (size_t i = 0; i < input.size(); ++i) {
            output[i] = exp(input[i] - max_val) / sum_exp;
        }
        return output;
    }

    // Forward pass
    std::vector<double> forward(const std::vector<double>& input) {
        std::vector<double> hidden_layer_output(bias1.size());
        for (size_t i = 0; i < bias1.size(); ++i) {
            double sum = bias1[i];
            for (size_t j = 0; j < input.size(); ++j) {
                sum += input[j] * weights1[j][i];
            }
            hidden_layer_output[i] = sigmoid(sum);
        }

        std::vector<double> output_layer_input(bias2.size());
        for (size_t i = 0; i < bias2.size(); ++i) {
            double sum = bias2[i];
            for (size_t j = 0; j < hidden_layer_output.size(); ++j) {
                sum += hidden_layer_output[j] * weights2[j][i];
            }
            output_layer_input[i] = sum;
        }

        return softmax(output_layer_input);
    }

    // Backward pass and weight update (simplified SGD)
    void backward(const std::vector<double>& input, const std::vector<double>& hidden_output, const std::vector<double>& output, int target_label, double learning_rate) {
        std::vector<double> output_errors(output.size());
        for (size_t i = 0; i < output.size(); ++i) {
            output_errors[i] = (i == static_cast<size_t>(target_label) ? 1.0 : 0.0) - output[i];
        }

        std::vector<double> hidden_errors(hidden_output.size());
        for (size_t i = 0; i < hidden_output.size(); ++i) {
            double sum = 0.0;
            for (size_t j = 0; j < output.size(); ++j) {
                sum += output_errors[j] * weights2[i][j];
            }
            hidden_errors[i] = sum * sigmoid_derivative(hidden_output[i]);
        }

        // Update weights and biases for output layer
        for (size_t i = 0; i < hidden_output.size(); ++i) {
            for (size_t j = 0; j < output.size(); ++j) {
                weights2[i][j] += learning_rate * output_errors[j] * hidden_output[i];
            }
        }
        for (size_t i = 0; i < output.size(); ++i) {
            bias2[i] += learning_rate * output_errors[i];
        }

        // Update weights and biases for hidden layer
        for (size_t i = 0; i < input.size(); ++i) {
            for (size_t j = 0; j < hidden_output.size(); ++j) {
                weights1[i][j] += learning_rate * hidden_errors[j] * input[i];
            }
        }
        for (size_t i = 0; i < hidden_output.size(); ++i) {
            bias1[i] += learning_rate * hidden_errors[i];
        }
    }

    // Function to perform inference on a single image
    int predict(const std::vector<double>& image) {
        std::vector<double> output = forward(image);
        int predicted_label = 0;
        double max_output = -1.0;
        for (size_t j = 0; j < output.size(); ++j) {
            if (output[j] > max_output) {
                max_output = output[j];
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

    // Create neural network
    NeuralNetwork model(input_size, hidden_size, output_size);

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