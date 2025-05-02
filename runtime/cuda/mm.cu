#include <cuda_runtime.h>
#include <stdio.h>

extern "C" {

__global__ void matmul_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    // A: M x K
    // B: K x N
    // C: M x N
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < K; ++i) {
            sum += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }
}

void matmul(const float* A, const float* B, float* C, int M, int N, int K) {
    // Grid/block config
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

    matmul_kernel<<<grid, block>>>(A, B, C, M, N, K);

    cudaDeviceSynchronize(); // make sure it's done
}

}
