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

// Function to allocate device memory
template<typename T> T* allocate_device_memory(size_t size);

    // Function to deallocate device memory
template<typename T> void free_device_memory(T* device_ptr);

// Function to transfer data from host to device
template<typename T> void host_to_device(const T* host_ptr, T* device_ptr, size_t size);

// Function to transfer data from device to host
template<typename T> void device_to_host(const T* device_ptr, T* host_ptr, size_t size);