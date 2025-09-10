#ifndef GPU_HELPERS_H
#define GPU_HELPERS_H

#include <cstddef>
#include <iostream>
#include <cudnn.h>

// Helper function for checking CUDA errors
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__    \
                      << " code: " << cudaGetErrorString(err) << std::endl; \
            exit(1);                                                        \
        }                                                                   \
    } while (0)

// Helper function for checking cuDNN errors
#define CUDNN_CHECK(call)                                                     \
    do {                                                                      \
        cudnnStatus_t status = call;                                          \
        if (status != CUDNN_STATUS_SUCCESS) {                                 \
            std::cerr << "cuDNN error at " << __FILE__ << ":" << __LINE__     \
                      << " code: " << cudnnGetErrorString(status) << std::endl; \
            exit(1);                                                          \
        }                                                                     \
    } while (0)

// Helper function for checking cuBLAS errors
#define CUBLAS_CHECK(call)                                                    \
    do {                                                                      \
        cublasStatus_t status = call;                                         \
        if (status != CUBLAS_STATUS_SUCCESS) {                                \
            std::cerr << "cuBLAS error at " << __FILE__ << ":" << __LINE__    \
                      << " code: " << status << std::endl; \
            exit(1);                                                          \
        }                                                                     \
    } while (0)

// Template functions must be defined in header files when they're used across multiple compilation units.

// Function to allocate device memory
// template<typename T> T* allocate_device_memory(size_t size) {
//     T* device_ptr;
//     CUDA_CHECK(cudaMalloc(&device_ptr, size * sizeof(T)));
//     return device_ptr;
// }

template <typename T> T* allocate_device_memory(size_t size) {
    T* device_ptr = nullptr;
    // Cast to void** to fix the type conversion error
    cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&device_ptr), size * sizeof(T));
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate device memory: " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
    return device_ptr;
}

// Function to deallocate device memory
template<typename T> void free_device_memory(T* device_ptr) {
    CUDA_CHECK(cudaFree(device_ptr));
}

// Function to transfer data from host to device
template<typename T> void host_to_device(const T* host_ptr, T* device_ptr, size_t size) {
    CUDA_CHECK(cudaMemcpy(device_ptr, host_ptr, size * sizeof(T), cudaMemcpyHostToDevice));
}

// Function to transfer data from device to host
template<typename T> void device_to_host(const T* device_ptr, T* host_ptr, size_t size) {
    CUDA_CHECK(cudaMemcpy(host_ptr, device_ptr, size * sizeof(T), cudaMemcpyDeviceToHost));
}

#endif // GPU_HELPERS_H