#include <cstdio>
#include <iostream>

#include "data_manager.h"
#include "gpu_util.h"

#include "gpu_benchmark.h"

#define VECTOR_LENGTH 512
#define M_BLOCK 16
#define K_BLOCK 32
#define N_BLOCK 512

__global__
void opt3MulKernel(size_t pitch_A, size_t pitch_B, size_t pitch_C,
                  float* cuda_A, float* cuda_B, float* cuda_C,
                  int k);

void opt3GPU_gemm_execute(GemmRun* run) {
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
    dim3 dimBlock(K_BLOCK, VECTOR_LENGTH / K_BLOCK);
    dim3 dimGrid(run->n / N_BLOCK, run->m / M_BLOCK);

    opt3MulKernel<<<dimGrid, dimBlock>>>(cuda_lda, cuda_ldb, cuda_ldc,
                                         cuda_A, cuda_B, cuda_C, run->k);
    cudaDeviceSynchronize();
    printf("%s\n", cudaGetErrorString(cudaGetLastError()));

    deinit_cuda_matrices(run, pitch_C, cuda_A, cuda_B, cuda_C);
}


__global__ void opt3MulKernel(size_t pitch_A, size_t pitch_B, size_t pitch_C,
                             float* cuda_A, float* cuda_B, float* cuda_C,
                             int k) {

    unsigned int A_block_offset = (blockIdx.y * M_BLOCK) * pitch_A;
    unsigned int B_block_offset = blockIdx.x * N_BLOCK;
    unsigned int C_block_offset = (blockIdx.y * M_BLOCK) * pitch_C + blockIdx.x * N_BLOCK;


    float Cvalue[M_BLOCK];

    #pragma unroll 16
    for (unsigned int i = 0; i < M_BLOCK; i++) {
        Cvalue[i] = 0;
    }

    for (unsigned int block_idx = 0; block_idx < (k / K_BLOCK); block_idx++) {
        __shared__ float A_shared[M_BLOCK][K_BLOCK];

        // load the block from A
        for (int j = 0; j < M_BLOCK; j += blockDim.y) {
            A_shared[j + threadIdx.y][threadIdx.x] = cuda_A[A_block_offset + (block_idx * K_BLOCK) + ((threadIdx.y + j) * pitch_A) + threadIdx.x];
        }
        __syncthreads();

        for (int i = 0; i < K_BLOCK; i ++) {
            float b = cuda_B[B_block_offset + ((block_idx * K_BLOCK + i) * pitch_B) + threadIdx.y * blockDim.x + threadIdx.x];
            // float b1 = cuda_B[B_block_offset + ((block_idx * K_BLOCK + i + 1) * pitch_B) + threadIdx.y * blockDim.x + threadIdx.x];

            #pragma unroll 16
            for (int k = 0; k < M_BLOCK; k++) {
                // Cvalue[k] += b * A_shared[k][i];
                Cvalue[k] = fmaf(b, A_shared[k][i], Cvalue[k]);
                // Cvalue[k] = fmaf(b1, A_shared[k][i+1], Cvalue[k]);
            }
        }
        __syncthreads();
    }

    #pragma unroll 16
    for (unsigned int i = 0; i < M_BLOCK; i++) {
        cuda_C[C_block_offset + i * pitch_C + (threadIdx.y * blockDim.x + threadIdx.x)] = Cvalue[i];
    }
}
