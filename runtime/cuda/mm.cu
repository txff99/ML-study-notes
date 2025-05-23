#include <cuda_runtime.h>
#include <stdio.h>

extern "C" {

__global__ void matmul_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

__global__ void add_kernel(const float* A, const float* B, float* C, int M, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        C[row * N + col] = A[row * N + col] + B[row * N + col];
    }
}

__global__ void sub_kernel(const float* A, const float* B, float* C, int M, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        C[row * N + col] = A[row * N + col] - B[row * N + col];
    }
}

__global__ void expand_kernel(const float* A, float* B, int M, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        B[row * N + col] = A[col];
    }
}

void matmul(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 block(32, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    matmul_kernel<<<grid, block>>>(A, B, C, M, N, K);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch error: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();
}

void add(const float* A, const float* B, float* C, int M, int N){
    dim3 block(32, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

    add_kernel<<<grid, block>>>(A, B, C, M, N);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch error: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();
}

void sub(const float* A, const float* B, float* C, int M, int N){
    dim3 block(32, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

    sub_kernel<<<grid, block>>>(A, B, C, M, N);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch error: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();
}

void expand(const float* A, float* B, int M, int N){
    dim3 block(32, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

    expand_kernel<<<grid, block>>>(A, B, M, N);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch error: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();
}
}