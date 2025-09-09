#ifndef NEURALNETWORKCPU_H
#define NEURALNETWORKCPU_H

#include <vector>

class NeuralNetworkCPU {
private:
    std::vector<std::vector<double>> weights2;
    std::vector<double> bias2;
public:
    std::vector<double> bias1;
    std::vector<std::vector<double>> weights1;
    NeuralNetworkCPU(int input_size, int hidden_size, int output_size);
    double sigmoid(double x);
    double sigmoid_derivative(double x);
    std::vector<double> softmax(const std::vector<double>& input);
    std::vector<double> forward(const std::vector<double>& input);
    void backward(const std::vector<double>& input, const std::vector<double>& hidden_output, const std::vector<double>& output, int target_label, double learning_rate);
    int predict(const std::vector<double>& image);
};

#endif // NEURALNETWORKCPU_H