#include <cstdio>
#include <iostream>

#include "data_manager.h"
#include "gpu_util.h"

#include "gpu_benchmark.h"

#define BLOCK_SIZE 32

__global__
void naiveMulKernel(size_t pitch_A, size_t pitch_B, size_t pitch_C,
                  float* cuda_A, float* cuda_B, float* cuda_C,
                  int k);

void naiveGPU_gemm_execute(GemmRun* run) {
    size_t pitch_A, pitch_B, pitch_C, cuda_lda, cuda_ldb, cuda_ldc;
    float* cuda_A;
    float* cuda_B;
    float* cuda_C;

    init_cuda_matrices(run, &pitch_A, &pitch_B, &pitch_C,
                       &cuda_A, &cuda_B, &cuda_C);
    cuda_lda = pitch_A / sizeof(float);
    cuda_ldb = pitch_B / sizeof(float);
    cuda_ldc = pitch_C / sizeof(float);

    // Invoke kernel
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(run->n / dimBlock.x, run->m / dimBlock.y);

    naiveMulKernel<<<dimGrid, dimBlock>>>(cuda_lda, cuda_ldb, cuda_ldc,
                                        cuda_A, cuda_B, cuda_C, run->k);
    cudaDeviceSynchronize();

    deinit_cuda_matrices(run, pitch_C, cuda_A, cuda_B, cuda_C);
}


__global__ void naiveMulKernel(size_t pitch_A, size_t pitch_B, size_t pitch_C,
                               float* cuda_A, float* cuda_B, float* cuda_C,
                               int k) {
    /*
     Taken from the CUDA Developer Documentation. Section 3.2.3 Shared Memory
    */
    // Each thread computes one element of C
    // by accumulating results into Cvalue
    float Cvalue = 0;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    for (int e = 0; e < k; ++e)
        Cvalue += cuda_A[row * pitch_A + e] * cuda_B[e * pitch_B + col];
    cuda_C[row * pitch_C + col] = Cvalue;
}
