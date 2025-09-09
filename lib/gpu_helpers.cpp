#include <cstddef>
#include <cuda_runtime.h>

// Function to allocate device memory
template<typename T> T* allocate_device_memory(size_t size) {
    T* device_ptr;
    CUDA_CHECK(cudaMalloc(&device_ptr, size * sizeof(T)));
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
