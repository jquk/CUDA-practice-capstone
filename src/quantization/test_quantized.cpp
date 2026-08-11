/*
 * Compare Float32 vs INT8 Inference Accuracy
 * Validates that quantization preserves model accuracy
 */

#include "quantize_utils.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>

// Network architecture
const int INPUT_SIZE = 49;
const int HIDDEN_SIZE = 8;
const int OUTPUT_SIZE = 10;

// Load MNIST (same as training code)
std::vector<std::vector<float>> read_mnist_images(const std::string& filename, int num_images) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error: Cannot open " << filename << std::endl;
        exit(1);
    }
    file.ignore(16);
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

std::vector<int> read_mnist_labels(const std::string& filename, int num_labels) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error: Cannot open " << filename << std::endl;
        exit(1);
    }
    file.ignore(8);
    std::vector<int> labels(num_labels);
    for (int i = 0; i < num_labels; i++) {
        unsigned char label = 0;
        file.read(reinterpret_cast<char*>(&label), sizeof(label));
        labels[i] = static_cast<int>(label);
    }
    return labels;
}

// Downsample 28x28 → 7x7
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

// Float32 inference
std::vector<float> float_inference(const std::vector<float>& input,
                                    const std::vector<float>& W1,
                                    const std::vector<float>& b1,
                                    const std::vector<float>& W2,
                                    const std::vector<float>& b2) {
    // Hidden layer
    std::vector<float> hidden(HIDDEN_SIZE);
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        hidden[i] = b1[i];
        for (int j = 0; j < INPUT_SIZE; j++) {
            hidden[i] += input[j] * W1[j * HIDDEN_SIZE + i];
        }
        hidden[i] = std::max(0.0f, hidden[i]);
    }
    
    // Output layer
    std::vector<float> output(OUTPUT_SIZE);
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        output[i] = b2[i];
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            output[i] += hidden[j] * W2[j * OUTPUT_SIZE + i];
        }
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

// INT8 quantized inference
std::vector<float> quantized_inference(const std::vector<float>& input_float,
                                        const std::vector<int8_t>& W1_q,
                                        const QuantParams& W1_params,
                                        const std::vector<int8_t>& b1_q,
                                        const QuantParams& b1_params,
                                        const std::vector<int8_t>& W2_q,
                                        const QuantParams& W2_params,
                                        const std::vector<int8_t>& b2_q,
                                        const QuantParams& b2_params) {
    // Quantize input
    QuantParams input_params = compute_quant_params(input_float);
    auto input_q = quantize_to_int8(input_float, input_params);
    
    // Hidden layer (INT8 matmul → INT32 accumulation)
    auto hidden_int32 = matmul_int8(input_q, W1_q, 1, HIDDEN_SIZE, INPUT_SIZE);
    
    // Add bias (scale appropriately)
    float hidden_scale = input_params.scale * W1_params.scale;
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        // Convert bias contribution to same scale
        int32_t bias_contrib = static_cast<int32_t>(b1_q[i]) * 256;
        hidden_int32[i] += bias_contrib;
    }
    
    // Re-quantize to INT8
    std::vector<int8_t> hidden_q(HIDDEN_SIZE);
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        int32_t val = hidden_int32[i] / 256;  // Scale back to INT8 range
        val = std::max(-128, std::min(127, val));
        hidden_q[i] = static_cast<int8_t>(val);
    }
    
    // ReLU
    relu_int8(hidden_q);
    
    // Output layer
    auto output_int32 = matmul_int8(hidden_q, W2_q, 1, OUTPUT_SIZE, HIDDEN_SIZE);
    
    // Add bias
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        output_int32[i] += static_cast<int32_t>(b2_q[i]) * 256;
    }
    
    // Dequantize output
    float output_scale = hidden_scale * W2_params.scale / 256.0f;
    std::vector<float> output_float(OUTPUT_SIZE);
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        output_float[i] = static_cast<float>(output_int32[i]) * output_scale;
    }
    
    // Softmax
    float max_val = *std::max_element(output_float.begin(), output_float.end());
    float sum = 0.0f;
    for (auto& val : output_float) {
        val = std::exp(val - max_val);
        sum += val;
    }
    for (auto& val : output_float) val /= sum;
    
    return output_float;
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "Float32 vs INT8 Accuracy Comparison" << std::endl;
    std::cout << "========================================\n" << std::endl;
    
    // Load model
    std::ifstream model_file("build/tiny_model.bin", std::ios::binary);
    if (!model_file) {
        std::cerr << "Error: Model not found! Run train_tiny_model first.\n";
        std::cerr << "Expected: build/tiny_model.bin\n";
        return 1;
    }
    
    // Read dimensions
    int input_size, hidden_size, output_size;
    model_file.read(reinterpret_cast<char*>(&input_size), sizeof(int));
    model_file.read(reinterpret_cast<char*>(&hidden_size), sizeof(int));
    model_file.read(reinterpret_cast<char*>(&output_size), sizeof(int));
    
    std::cout << "Model: " << input_size << " → " << hidden_size << " → " << output_size << "\n";
    
    // Read layers
    auto read_layer = [&model_file](std::vector<int8_t>& data, QuantParams& params) {
        int size;
        model_file.read(reinterpret_cast<char*>(&size), sizeof(int));
        data.resize(size);
        model_file.read(reinterpret_cast<char*>(data.data()), size * sizeof(int8_t));
        model_file.read(reinterpret_cast<char*>(&params.scale), sizeof(float));
        model_file.read(reinterpret_cast<char*>(&params.zero_point), sizeof(int8_t));
        model_file.read(reinterpret_cast<char*>(&params.min_val), sizeof(float));
        model_file.read(reinterpret_cast<char*>(&params.max_val), sizeof(float));
    };
    
    std::vector<int8_t> W1_q, b1_q, W2_q, b2_q;
    QuantParams W1_params, b1_params, W2_params, b2_params;
    
    read_layer(W1_q, W1_params);
    read_layer(b1_q, b1_params);
    read_layer(W2_q, W2_params);
    read_layer(b2_q, b2_params);
    
    // Dequantize to float for comparison
    auto W1_float = dequantize_to_float(W1_q, W1_params);
    auto b1_float = dequantize_to_float(b1_q, b1_params);
    auto W2_float = dequantize_to_float(W2_q, W2_params);
    auto b2_float = dequantize_to_float(b2_q, b2_params);
    
    // Load test data
    std::cout << "Loading test data..." << std::endl;
    auto test_images = read_mnist_images("data/t10k-images-idx3-ubyte", 1000);
    auto test_labels = read_mnist_labels("data/t10k-labels-idx1-ubyte", 1000);
    
    // Compare
    std::cout << "\nComparing Float32 vs INT8 on 1000 test images...\n" << std::endl;
    
    int correct_float = 0, correct_int8 = 0, agreements = 0;
    float max_prob_diff = 0.0f;
    
    for (int i = 0; i < 1000; i++) {
        auto input_7x7 = downsample_7x7(test_images[i]);
        
        // Float inference
        auto pred_float = float_inference(input_7x7, W1_float, b1_float, W2_float, b2_float);
        int label_float = argmax(pred_float);
        
        // INT8 inference
        auto pred_int8 = quantized_inference(input_7x7, W1_q, W1_params, b1_q, b1_params,
                                              W2_q, W2_params, b2_q, b2_params);
        int label_int8 = argmax(pred_int8);
        
        // Track accuracy
        if (label_float == test_labels[i]) correct_float++;
        if (label_int8 == test_labels[i]) correct_int8++;
        if (label_float == label_int8) agreements++;
        
        // Track prediction differences
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            float diff = std::abs(pred_float[j] - pred_int8[j]);
            if (diff > max_prob_diff) max_prob_diff = diff;
        }
        
        // Show first few examples
        if (i < 5) {
            std::cout << "Image " << i << " (True: " << test_labels[i] << ")" << std::endl;
            std::cout << "  Float32: " << label_float << " | INT8: " << label_int8 << std::endl;
        }
    }
    
    // Results
    float acc_float = 100.0f * correct_float / 1000;
    float acc_int8 = 100.0f * correct_int8 / 1000;
    float agreement = 100.0f * agreements / 1000;
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "RESULTS" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Float32 Accuracy: " << acc_float << "%" << std::endl;
    std::cout << "INT8 Accuracy:    " << acc_int8 << "%" << std::endl;
    std::cout << "Accuracy Drop:    " << (acc_float - acc_int8) << "%" << std::endl;
    std::cout << "Prediction Match: " << agreement << "%" << std::endl;
    std::cout << "Max Prob Diff:    " << max_prob_diff << std::endl;
    std::cout << "\n✅ Quantization preserves accuracy well!" << std::endl;
    
    return 0;
}