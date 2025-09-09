
#include "helpers.h"
#include <iostream>
#include <fstream>
#include <vector>


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