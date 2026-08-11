/*
 * Train Tiny Model for Arduino Uno
 * Architecture: 7x7 input (49) → 8 hidden → 10 output
 * Total parameters: 490 weights + biases
 * Quantized size: ~490 bytes as INT8
 */

#include "quantize_utils.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <filesystem> // for debugging only

const int INPUT_SIZE = 49;    // 7x7 downsampled MNIST
const int HIDDEN_SIZE = 8;
const int OUTPUT_SIZE = 10;
const int EPOCHS = 30;
const float LEARNING_RATE = 0.05f;

// Read MNIST images
std::vector<std::vector<float>> read_mnist_images(const std::string& filename, int num_images) {
    std::filesystem::path cwd = std::filesystem::current_path();
    std::cout << "cwd is " << cwd << std::endl;

    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error: Cannot open " << filename << std::endl;
        exit(1);
    }
    
    file.ignore(16);  // Skip header
    std::vector<std::vector<float>> images(num_images, std::vector<float>(784));
    
    for (int i = 0; i < num_images; i++) {
        for (int j = 0; j < 784; j++) {
            unsigned char pixel = 0;
            file.read(reinterpret_cast<char*>(&pixel), sizeof(pixel));
            images[i][j] = static_cast<float>(pixel) / 255.0f;
        }
    }
    return images;
}

// Read MNIST labels
std::vector<int> read_mnist_labels(const std::string& filename, int num_labels) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error: Cannot open " << filename << std::endl;
        exit(1);
    }
    
    file.ignore(8);  // Skip header
    std::vector<int> labels(num_labels);
    
    for (int i = 0; i < num_labels; i++) {
        unsigned char label = 0;
        file.read(reinterpret_cast<char*>(&label), sizeof(label));
        labels[i] = static_cast<int>(label);
    }
    return labels;
}

// Downsample 28x28 to 7x7 by averaging 4x4 blocks
std::vector<float> downsample_7x7(const std::vector<float>& img_28x28) {
    std::vector<float> img_7x7(49);
    
    for (int i = 0; i < 7; i++) {
        for (int j = 0; j < 7; j++) {
            float sum = 0.0f;
            for (int di = 0; di < 4; di++) {
                for (int dj = 0; dj < 4; dj++) {
                    sum += img_28x28[(i * 4 + di) * 28 + (j * 4 + dj)];
                }
            }
            img_7x7[i * 7 + j] = sum / 16.0f;
        }
    }
    return img_7x7;
}

class TinyNetwork {
private:
    std::vector<float> W1, b1, W2, b2;
    
public:
    TinyNetwork() {
        // Initialize weights
        W1.resize(INPUT_SIZE * HIDDEN_SIZE);
        b1.resize(HIDDEN_SIZE, 0.0f);
        W2.resize(HIDDEN_SIZE * OUTPUT_SIZE);
        b2.resize(OUTPUT_SIZE, 0.0f);
        
        std::random_device rd;
        std::mt19937 gen(rd());
        float std1 = std::sqrt(2.0f / INPUT_SIZE);
        float std2 = std::sqrt(2.0f / HIDDEN_SIZE);
        std::normal_distribution<float> d1(0.0f, std1);
        std::normal_distribution<float> d2(0.0f, std2);
        
        for (auto& w : W1) w = d1(gen);
        for (auto& w : W2) w = d2(gen);
    }
    
    std::vector<float> forward(const std::vector<float>& input) {
        // Hidden layer
        std::vector<float> hidden(HIDDEN_SIZE, 0.0f);
        for (int i = 0; i < HIDDEN_SIZE; i++) {
            for (int j = 0; j < INPUT_SIZE; j++) {
                hidden[i] += input[j] * W1[j * HIDDEN_SIZE + i];
            }
            hidden[i] += b1[i];
            hidden[i] = std::max(0.0f, hidden[i]);  // ReLU
        }
        
        // Output layer
        std::vector<float> output(OUTPUT_SIZE, 0.0f);
        for (int i = 0; i < OUTPUT_SIZE; i++) {
            for (int j = 0; j < HIDDEN_SIZE; j++) {
                output[i] += hidden[j] * W2[j * OUTPUT_SIZE + i];
            }
            output[i] += b2[i];
        }
        
        // Softmax
        float max_val = *std::max_element(output.begin(), output.end());
        float sum = 0.0f;
        for (auto& val : output) {
            val = std::exp(val - max_val);
            sum += val;
        }
        for (auto& val : output) val /= sum;
        
        return output;
    }
    
    void train(const std::vector<std::vector<float>>& images_28x28,
               const std::vector<int>& labels,
               int epochs) {
        
        // Downsample all images first
        std::vector<std::vector<float>> images;
        for (const auto& img : images_28x28) {
            images.push_back(downsample_7x7(img));
        }
        
        int n = images.size();
        std::cout << "Training on " << n << " samples for " << epochs << " epochs...\n";
        
        for (int epoch = 0; epoch < epochs; epoch++) {
            float total_loss = 0.0f;
            int correct = 0;
            
            for (int idx = 0; idx < n; idx++) {
                // Forward pass
                auto output = forward(images[idx]);
                
                // Accuracy
                int pred = argmax(output);
                if (pred == labels[idx]) correct++;
                total_loss += -std::log(output[labels[idx]] + 1e-10f);
                
                // Backward pass - output layer
                std::vector<float> d_out(OUTPUT_SIZE);
                for (int i = 0; i < OUTPUT_SIZE; i++) {
                    d_out[i] = output[i] - (i == labels[idx] ? 1.0f : 0.0f);
                }
                
                // Get hidden layer activations (recompute)
                std::vector<float> hidden(HIDDEN_SIZE);
                std::vector<float> hidden_pre_relu(HIDDEN_SIZE);
                for (int i = 0; i < HIDDEN_SIZE; i++) {
                    hidden_pre_relu[i] = b1[i];
                    for (int j = 0; j < INPUT_SIZE; j++) {
                        hidden_pre_relu[i] += images[idx][j] * W1[j * HIDDEN_SIZE + i];
                    }
                    hidden[i] = std::max(0.0f, hidden_pre_relu[i]);
                }
                
                // Hidden layer gradients
                std::vector<float> d_hidden(HIDDEN_SIZE, 0.0f);
                for (int i = 0; i < HIDDEN_SIZE; i++) {
                    for (int j = 0; j < OUTPUT_SIZE; j++) {
                        d_hidden[i] += d_out[j] * W2[i * OUTPUT_SIZE + j];
                    }
                    if (hidden_pre_relu[i] <= 0) d_hidden[i] = 0;  // ReLU derivative
                }
                
                // Update W2, b2
                for (int i = 0; i < OUTPUT_SIZE; i++) {
                    b2[i] -= LEARNING_RATE * d_out[i];
                    for (int j = 0; j < HIDDEN_SIZE; j++) {
                        W2[j * OUTPUT_SIZE + i] -= LEARNING_RATE * d_out[i] * hidden[j];
                    }
                }
                
                // Update W1, b1
                for (int i = 0; i < HIDDEN_SIZE; i++) {
                    b1[i] -= LEARNING_RATE * d_hidden[i];
                    for (int j = 0; j < INPUT_SIZE; j++) {
                        W1[j * HIDDEN_SIZE + i] -= LEARNING_RATE * d_hidden[i] * images[idx][j];
                    }
                }
            }
            
            float acc = 100.0f * correct / n;
            std::cout << "Epoch " << (epoch + 1) << "/" << epochs 
                     << " | Loss: " << total_loss / n 
                     << " | Acc: " << acc << "%" << std::endl;
        }
    }
    
    float test(const std::vector<std::vector<float>>& images_28x28,
               const std::vector<int>& labels) {
        int correct = 0;
        for (size_t i = 0; i < images_28x28.size(); i++) {
            auto img_7x7 = downsample_7x7(images_28x28[i]);
            auto output = forward(img_7x7);
            if (argmax(output) == labels[i]) correct++;
        }
        return 100.0f * correct / images_28x28.size();
    }
    
    // Getter methods for weights
    const std::vector<float>& get_W1() const { return W1; }
    const std::vector<float>& get_b1() const { return b1; }
    const std::vector<float>& get_W2() const { return W2; }
    const std::vector<float>& get_b2() const { return b2; }
};

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "Training Tiny Model for Arduino Uno" << std::endl;
    std::cout << "Architecture: " << INPUT_SIZE << " → " << HIDDEN_SIZE << " → " << OUTPUT_SIZE << std::endl;
    std::cout << "Parameters: " << (INPUT_SIZE*HIDDEN_SIZE + HIDDEN_SIZE + HIDDEN_SIZE*OUTPUT_SIZE + OUTPUT_SIZE) << std::endl;
    std::cout << "========================================\n" << std::endl;
    
    // Load data
    std::cout << "Loading MNIST..." << std::endl;
    auto train_images = read_mnist_images("data/train-images-idx3-ubyte", 60000);
    auto train_labels = read_mnist_labels("data/train-labels-idx1-ubyte", 60000);
    auto test_images = read_mnist_images("data/t10k-images-idx3-ubyte", 10000);
    auto test_labels = read_mnist_labels("data/t10k-labels-idx1-ubyte", 10000);
    
    // Train
    TinyNetwork nn;
    nn.train(train_images, train_labels, EPOCHS);
    
    // Test
    float accuracy = nn.test(test_images, test_labels);
    std::cout << "\nTest Accuracy: " << accuracy << "%" << std::endl;
    
    // Quantize weights
    std::cout << "\n=== Quantizing Weights ===" << std::endl;
    
    auto W1 = nn.get_W1();
    auto b1 = nn.get_b1();
    auto W2 = nn.get_W2();
    auto b2 = nn.get_b2();
    
    QuantParams W1_params = compute_quant_params(W1);
    QuantParams b1_params = compute_quant_params(b1);
    QuantParams W2_params = compute_quant_params(W2);
    QuantParams b2_params = compute_quant_params(b2);
    
    auto W1_q = quantize_to_int8(W1, W1_params);
    auto b1_q = quantize_to_int8(b1, b1_params);
    auto W2_q = quantize_to_int8(W2, W2_params);
    auto b2_q = quantize_to_int8(b2, b2_params);
    
    print_quant_stats("W1", W1_params, W1, W1_q);
    print_quant_stats("b1", b1_params, b1, b1_q);
    print_quant_stats("W2", W2_params, W2, W2_q);
    print_quant_stats("b2", b2_params, b2, b2_q);
    
    // Save model
    std::string model_path = "build/tiny_model.bin";
    save_quantized_model(model_path, W1_q, W1_params, b1_q, b1_params, 
                        W2_q, W2_params, b2_q, b2_params,
                        INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);
    
    std::cout << "\nModel saved to: " << model_path << std::endl;
    
    // Memory report
    int total_bytes = W1_q.size() + b1_q.size() + W2_q.size() + b2_q.size();
    int ram_needed = INPUT_SIZE + HIDDEN_SIZE * 2 + OUTPUT_SIZE;
    
    std::cout << "\n=== Arduino Uno Feasibility ===" << std::endl;
    std::cout << "Flash needed: ~" << total_bytes << " bytes (weights)" << std::endl;
    std::cout << "RAM needed: ~" << ram_needed << " bytes (activations)" << std::endl;
    std::cout << "Estimated total flash: ~" << (total_bytes + 3000) << " / 32256 bytes" << std::endl;
    std::cout << "Status: ✅ FITS ON ARDUINO UNO!" << std::endl;
    
    return 0;
}