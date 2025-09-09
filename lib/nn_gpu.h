#ifndef NEURALNETWORKGPU_H
#define NEURALNETWORKGPU_H

class NeuralNetworkGPU {
private:

public:
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

    NeuralNetworkGPU(int input_size, int hidden_size, int output_size);
    ~NeuralNetworkGPU();
    void forward_gpu(const double* d_input_data);
    void backward_gpu(const double* d_input_data, int target_label, double learning_rate);
    int predict_gpu(const double* d_image);
};

#endif // NEURALNETWORKGPU_H