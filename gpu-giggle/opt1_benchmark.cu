#include <cstdio>
#include <iostream>

#include "data_manager.h"
#include "gpu_util.h"

#include "gpu_benchmark.h"

#define BLOCK_SIZE 32

__global__
void opt1MulKernel(size_t pitch_A, size_t pitch_B, size_t pitch_C,
                  float* cuda_A, float* cuda_B, float* cuda_C,
                  int k);

void opt1GPU_gemm_execute(GemmRun* run) {
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

    opt1MulKernel<<<dimGrid, dimBlock>>>(cuda_lda, cuda_ldb, cuda_ldc,
                                         cuda_A, cuda_B, cuda_C, run->k);
    cudaDeviceSynchronize();
    // printf("%s\n", cudaGetErrorString(cudaGetLastError()));

    deinit_cuda_matrices(run, pitch_C, cuda_A, cuda_B, cuda_C);
}


__global__ void opt1MulKernel(size_t pitch_A, size_t pitch_B, size_t pitch_C,
                             float* cuda_A, float* cuda_B, float* cuda_C,
                             int k) {
    /*
     Taken from the CUDA Developer Documentation. Section 3.2.3 Shared Memory
    */

    unsigned int A_thread_offset = (threadIdx.y * pitch_A) + threadIdx.x;
    unsigned int B_thread_offset = (threadIdx.y * pitch_B) + threadIdx.x;
    unsigned int C_thread_offset = (threadIdx.y * pitch_C) + threadIdx.x;

    unsigned int A_block_offset = blockIdx.y * BLOCK_SIZE * pitch_A;
    unsigned int B_block_offset = blockIdx.x * BLOCK_SIZE;
    unsigned int C_block_offset = blockIdx.y * pitch_C * BLOCK_SIZE;


    float Cvalue = 0;

    for (unsigned int block_idx = 0; block_idx < (k / BLOCK_SIZE); block_idx++) {
        __shared__ float A_shared[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float B_shared[BLOCK_SIZE][BLOCK_SIZE];

        // load the block from A and B
        A_shared[threadIdx.y][threadIdx.x] = cuda_A[A_block_offset + (block_idx * BLOCK_SIZE) + A_thread_offset];
        B_shared[threadIdx.y][threadIdx.x] = cuda_B[B_block_offset + (block_idx * BLOCK_SIZE * pitch_B) + B_thread_offset];

        __syncthreads();

        // Do the block multiplication
        for (unsigned int ele_idx = 0; ele_idx < BLOCK_SIZE; ele_idx++) {
            Cvalue += A_shared[threadIdx.y][ele_idx] * B_shared[ele_idx][threadIdx.x];
        }
        __syncthreads();

    }
    cuda_C[C_block_offset + (blockIdx.x * BLOCK_SIZE) + C_thread_offset] = Cvalue;
}
