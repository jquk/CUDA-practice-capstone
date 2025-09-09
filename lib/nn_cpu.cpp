#include "nn_cpu.h"
#include <random> // For better random initialization

// Define the structure for a simple neural network model (MLP)
// Constructor
NeuralNetworkCPU::NeuralNetworkCPU(int input_size, int hidden_size, int output_size) {
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
double NeuralNetworkCPU::sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

// Derivative of sigmoid
double NeuralNetworkCPU::sigmoid_derivative(double x) {
    return x * (1.0 - x);
}

// Softmax activation function
std::vector<double> NeuralNetworkCPU::softmax(const std::vector<double>& input) {
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
std::vector<double> NeuralNetworkCPU::forward(const std::vector<double>& input) {
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
void NeuralNetworkCPU::backward(const std::vector<double>& input, const std::vector<double>& hidden_output, const std::vector<double>& output, int target_label, double learning_rate) {
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
int NeuralNetworkCPU::predict(const std::vector<double>& image) {
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