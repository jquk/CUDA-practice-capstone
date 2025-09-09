#ifndef HELPERS_H
#define HELPERS_H

#include <vector>
#include <iostream>

std::vector<int> read_mnist_labels(const std::string& filename);
std::vector<std::vector<double>> read_mnist_images(const std::string& filename);

#endif // HELPERS_H