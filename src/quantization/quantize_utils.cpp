#include "quantize_utils.h"
#include <iostream>
#include <fstream>
#include <cstring>

QuantParams compute_quant_params(const std::vector<float>& data) {
    if (data.empty()) {
        return QuantParams();
    }
    
    QuantParams params;
    
    // Find min and max
    params.min_val = *std::min_element(data.begin(), data.end());
    params.max_val = *std::max_element(data.begin(), data.end());
    
    // Handle edge case: all values same
    float range = params.max_val - params.min_val;
    if (range < 1e-7f) {
        range = 1e-7f;
    }
    
    // Symmetric quantization: map [min, max] to [-128, 127]
    // scale = (max - min) / 255
    params.scale = range / 255.0f;
    params.zero_point = 0;  // Symmetric: zero maps to 0
    
    return params;
}

std::vector<int8_t> quantize_to_int8(const std::vector<float>& data, const QuantParams& params) {
    std::vector<int8_t> quantized(data.size());
    
    for (size_t i = 0; i < data.size(); i++) {
        // Normalize to [0, 255] then shift to [-128, 127]
        float normalized = (data[i] - params.min_val) / params.scale;
        int32_t int_val = static_cast<int32_t>(std::round(normalized)) - 128;
        
        // Clamp to INT8 range
        int_val = std::max(-128, std::min(127, int_val));
        quantized[i] = static_cast<int8_t>(int_val);
    }
    
    return quantized;
}

std::vector<float> dequantize_to_float(const std::vector<int8_t>& data, const QuantParams& params) {
    std::vector<float> float_data(data.size());
    
    for (size_t i = 0; i < data.size(); i++) {
        // Reverse the quantization
        float_data[i] = (static_cast<float>(data[i]) + 128.0f) * params.scale + params.min_val;
    }
    
    return float_data;
}

std::vector<int32_t> matmul_int8(const std::vector<int8_t>& A, 
                                  const std::vector<int8_t>& B,
                                  int M, int N, int K) {
    std::vector<int32_t> C(M * N, 0);
    
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            int32_t sum = 0;
            for (int k = 0; k < K; k++) {
                sum += static_cast<int32_t>(A[i * K + k]) * static_cast<int32_t>(B[k * N + j]);
            }
            C[i * N + j] = sum;
        }
    }
    
    return C;
}

void relu_int8(std::vector<int8_t>& data) {
    for (auto& val : data) {
        if (val < 0) val = 0;
    }
}

int argmax(const std::vector<float>& data) {
    return std::max_element(data.begin(), data.end()) - data.begin();
}

int argmax_int32(const std::vector<int32_t>& data) {
    return std::max_element(data.begin(), data.end()) - data.begin();
}

void print_quant_stats(const std::string& name, const QuantParams& params, 
                       const std::vector<float>& original, 
                       const std::vector<int8_t>& quantized) {
    std::cout << "\n=== " << name << " Quantization Stats ===" << std::endl;
    std::cout << "Original range: [" << params.min_val << ", " << params.max_val << "]" << std::endl;
    std::cout << "Scale: " << params.scale << std::endl;
    std::cout << "Size: " << original.size() << " floats → " << quantized.size() << " int8_t" << std::endl;
    std::cout << "Memory: " << original.size() * sizeof(float) << " bytes → " 
              << quantized.size() * sizeof(int8_t) << " bytes ("
              << (4.0f * 100) << "% reduction)" << std::endl;
    
    // Calculate quantization error
    auto dequant = dequantize_to_float(quantized, params);
    float max_error = 0.0f;
    float mse = 0.0f;
    for (size_t i = 0; i < original.size(); i++) {
        float error = std::abs(original[i] - dequant[i]);
        if (error > max_error) max_error = error;
        mse += error * error;
    }
    mse /= original.size();
    
    std::cout << "Max error: " << max_error << std::endl;
    std::cout << "MSE: " << mse << std::endl;
}

void save_quantized_model(const std::string& filename,
                         const std::vector<int8_t>& W1, const QuantParams& W1_params,
                         const std::vector<int8_t>& b1, const QuantParams& b1_params,
                         const std::vector<int8_t>& W2, const QuantParams& W2_params,
                         const std::vector<int8_t>& b2, const QuantParams& b2_params,
                         int input_size, int hidden_size, int output_size) {
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot create file: " + filename);
    }
    
    // Write dimensions
    file.write(reinterpret_cast<const char*>(&input_size), sizeof(int));
    file.write(reinterpret_cast<const char*>(&hidden_size), sizeof(int));
    file.write(reinterpret_cast<const char*>(&output_size), sizeof(int));
    
    // Helper lambda to write vector + params
    auto write_layer = [&file](const std::vector<int8_t>& data, const QuantParams& params) {
        int size = data.size();
        file.write(reinterpret_cast<const char*>(&size), sizeof(int));
        file.write(reinterpret_cast<const char*>(data.data()), size * sizeof(int8_t));
        file.write(reinterpret_cast<const char*>(&params.scale), sizeof(float));
        file.write(reinterpret_cast<const char*>(&params.zero_point), sizeof(int8_t));
        file.write(reinterpret_cast<const char*>(&params.min_val), sizeof(float));
        file.write(reinterpret_cast<const char*>(&params.max_val), sizeof(float));
    };
    
    write_layer(W1, W1_params);
    write_layer(b1, b1_params);
    write_layer(W2, W2_params);
    write_layer(b2, b2_params);
}