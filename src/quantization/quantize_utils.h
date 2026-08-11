/*
 * Generic Quantization Utilities
 * Portable to any CPU architecture
 * Reference implementation for microcontroller deployment
 */

#pragma once
#include <vector>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <stdexcept>

struct QuantParams {
    float scale;
    int8_t zero_point;
    float min_val;
    float max_val;
    
    QuantParams() : scale(1.0f), zero_point(0), min_val(0.0f), max_val(0.0f) {}
};

// Calculate quantization parameters from float data
// Uses symmetric quantization: zero_point = 0, range = [-128, 127]
QuantParams compute_quant_params(const std::vector<float>& data);

// Quantize float vector to INT8
std::vector<int8_t> quantize_to_int8(const std::vector<float>& data, const QuantParams& params);

// Dequantize INT8 vector back to float
std::vector<float> dequantize_to_float(const std::vector<int8_t>& data, const QuantParams& params);

// INT8 matrix multiplication (reference implementation)
// C = A * B where A is MxK, B is KxN, C is MxN
std::vector<int32_t> matmul_int8(const std::vector<int8_t>& A, 
                                  const std::vector<int8_t>& B,
                                  int M, int N, int K);

// ReLU activation for INT8 data (in-place)
void relu_int8(std::vector<int8_t>& data);

// Find index of maximum value (argmax)
int argmax(const std::vector<float>& data);
int argmax_int32(const std::vector<int32_t>& data);

// Print quantization statistics
void print_quant_stats(const std::string& name, const QuantParams& params, 
                       const std::vector<float>& original, 
                       const std::vector<int8_t>& quantized);

// Save/Load quantized model
void save_quantized_model(const std::string& filename,
                         const std::vector<int8_t>& W1, const QuantParams& W1_params,
                         const std::vector<int8_t>& b1, const QuantParams& b1_params,
                         const std::vector<int8_t>& W2, const QuantParams& W2_params,
                         const std::vector<int8_t>& b2, const QuantParams& b2_params,
                         int input_size, int hidden_size, int output_size);