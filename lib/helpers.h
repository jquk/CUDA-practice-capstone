#ifndef HELPERS_H
#define HELPERS_H

#include <vector>
#include <iostream>

std::vector<int> read_mnist_labels(const std::string& filename);
std::vector<std::vector<double>> read_mnist_images(const std::string& filename);

// Function to validate the program's input parameters given by the user
int validate_program_input_parameters(int argc, const char *argv[], int *epochs, int *hidden_layer_size, double *learning_rate);

#endif // HELPERS_H