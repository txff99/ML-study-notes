#include <cuda_runtime.h>
#include <stdio.h>

extern "C" void* alloc_on_gpu(int size_bytes) {
    void* d_ptr = nullptr;
    cudaError_t err = cudaMalloc(&d_ptr, size_bytes);
    if (err != cudaSuccess) {
        printf("cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return nullptr; 
    }
    return d_ptr;
}

extern "C" void free_on_gpu(void* ptr) {
    cudaError_t err = cudaFree(ptr);
    if (err != cudaSuccess) {
        printf("cudaFree failed: %s\n", cudaGetErrorString(err));
    }
}

extern "C" void cuda_copy_to_device(void* dst, const void* src, size_t size) {
    cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
}

extern "C" void cuda_copy_to_host(void* dst, const void* src, size_t size) {
    cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
}