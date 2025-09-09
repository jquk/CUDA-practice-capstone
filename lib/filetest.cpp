#include <iostream>
#include <fstream>
#include <filesystem>

int main() {
    std::string test_file = "~/content/train-labels-idx1-ubyte";
    
    // Try to open with relative path first
    std::ifstream file("~/content/train-labels-idx1-ubyte");
    if (file.is_open()) {
        std::cout << "File opened successfully with relative path" << std::endl;
        file.close();
    } else {
        std::cout << "Could not open with relative path" << std::endl;
    }

    test_file = "home/jose/content/train-labels-idx1-ubyte";
    
    // Try to open with absolute path first
    std::ifstream file2("/home/jose/content/train-labels-idx1-ubyte");
    if (file2.is_open()) {
        std::cout << "File opened successfully with absolute path" << std::endl;
        file2.close();
    } else {
        std::cout << "Could not open with absolute path" << std::endl;
    }
    
    return 0;
}