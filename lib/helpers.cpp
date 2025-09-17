
#include "helpers.h"
#include <iostream>
#include <fstream>
#include <vector>
// #include <stdio.h>
// #include <stdlib.h>
#include "../cfg/cfg.h"
#include <errno.h>
#include <ctype.h>

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

// Function to verify that the given parameter is an integer
int is_valid_int(const char *str, int *out) {
    char *endptr;
    errno = 0;
    long val = strtol(str, &endptr, 10);
    if (errno != 0 || *endptr != '\0') return 0;
    *out = (int)val;
    return 1;
}

// Function to verify that the given parameter is a float
int is_valid_float(const char *str, double *out) {
    char *endptr;
    errno = 0;
    double val = strtod(str, &endptr);
    if (errno != 0 || *endptr != '\0') return 0;
    *out = val;
    return 1;
}

// Function to validate the program's input parameters given by the user
int validate_program_input_parameters(int argc, const char *argv[], int *epochs, int *hidden_layer_size, double *learning_rate) {
    if (argc != 4) {
        printf("Usage: %s <training epochs:1-100> <neural network's hidden_layer:100-500> <neural network's learning_rate:0.00001-0.1>\n", argv[0]);
        return 1;
    }

    if (!is_valid_int(argv[1], epochs) || *epochs < EPOCHS_MIN || *epochs > EPOCHS_MAX) {
        printf("Error: epochs must be an integer between 1 and 100.\n");
        printf("Usage: %s <training epochs:%d-%d> <neural network's hidden_layer:%d-%d> <neural network's learning_rate:%f-%f\n", argv[1], EPOCHS_MIN, EPOCHS_MAX, HIDDEN_LAYERS_SIZE_MIN, HIDDEN_LAYERS_SIZE_MAX, LEARNING_RATE_MIN, LEARNING_RATE_MAX );
        return 1;
    }

    if (!is_valid_int(argv[2], hidden_layer_size) || *hidden_layer_size < HIDDEN_LAYERS_SIZE_MIN || *hidden_layer_size > HIDDEN_LAYERS_SIZE_MAX) {
        printf("Error: hidden_layer_size must be an integer between 100 and 500.\n");
        printf("Usage: %s <training epochs:%d-%d> <neural network's hidden_layer:%d-%d> <neural network's learning_rate:%f-%f\n", argv[1], EPOCHS_MIN, EPOCHS_MAX, HIDDEN_LAYERS_SIZE_MIN, HIDDEN_LAYERS_SIZE_MAX, LEARNING_RATE_MIN, LEARNING_RATE_MAX );
        return 1;
    }

    if (!is_valid_float(argv[3], learning_rate) || *learning_rate < LEARNING_RATE_MIN || *learning_rate > LEARNING_RATE_MAX) {
        printf("Error: learning_rate must be a float between 0.00001 and 0.1.\n");
        printf("Usage: %s <training epochs:%d-%d> <neural network's hidden_layer:%d-%d> <neural network's learning_rate:%f-%f\n", argv[1], EPOCHS_MIN, EPOCHS_MAX, HIDDEN_LAYERS_SIZE_MIN, HIDDEN_LAYERS_SIZE_MAX, LEARNING_RATE_MIN, LEARNING_RATE_MAX );
        return 1;
    }

    printf("Parameters accepted:\n");
    printf(" - epochs = %d\n", *epochs);
    printf(" - hidden_layer_size = %d\n", *hidden_layer_size);
    printf(" - learning_rate = %f\n", *learning_rate);

    return 0;
}
