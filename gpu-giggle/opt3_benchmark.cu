#include <cstdio>
#include <iostream>

#include "data_manager.h"
#include "gpu_util.h"

#include "gpu_benchmark.h"

#define VECTOR_LENGTH 512
#define M_BLOCK 32
#define K_BLOCK 32
#define N_BLOCK 128

__global__
void opt3MulKernel(size_t pitch_A, size_t pitch_B, size_t pitch_C,
                  float* cuda_A, float* cuda_B, float* cuda_C,
                  int k);

void opt3GPU_gemm_execute(GemmRun<float>* run) {
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
    // dim3 dimBlock(K_BLOCK, 32);
    dim3 dimBlock(K_BLOCK, VECTOR_LENGTH / K_BLOCK);
    dim3 dimGrid(run->n / N_BLOCK, run->m / M_BLOCK);

    opt3MulKernel<<<dimGrid, dimBlock>>>(cuda_lda, cuda_ldb, cuda_ldc,
                                         cuda_A, cuda_B, cuda_C, run->k);
    // cudaDeviceSynchronize();
    // printf("%s\n", cudaGetErrorString(cudaGetLastError()));

    deinit_cuda_matrices(run, pitch_C, cuda_A, cuda_B, cuda_C);
}


__global__ void opt3MulKernel(size_t pitch_A, size_t pitch_B, size_t pitch_C,
                             float* cuda_A, float* cuda_B, float* cuda_C,
                             int k) {

    unsigned int A_block_offset = (blockIdx.y * M_BLOCK) * pitch_A;
    unsigned int B_block_offset = blockIdx.x * N_BLOCK;
    unsigned int C_block_offset = (blockIdx.y * M_BLOCK) * pitch_C + blockIdx.x * N_BLOCK;

    // printf("Grid x %d, y %d\n", gridDim.x, gridDim.y);
    // printf("Block x %d, y %d\n", blockDim.x, blockDim.y);


    float Cvalue[8];

    #pragma unroll 8
    for (unsigned int i = 0; i < 8; i++) {
        Cvalue[i] = 0;
    }

    for (unsigned int block_idx = 0; block_idx < (k / K_BLOCK); block_idx++) {
        __shared__ float A_shared[M_BLOCK][K_BLOCK];
        __shared__ float B_shared[K_BLOCK][N_BLOCK];

        // load the block from A
        for (int j = 0; j < M_BLOCK; j += blockDim.y) {
            A_shared[j + threadIdx.y][threadIdx.x] = cuda_A[A_block_offset + (block_idx * K_BLOCK) + ((threadIdx.y + j) * pitch_A) + threadIdx.x];
        }

        for (int i = 0; i < K_BLOCK; i += blockDim.y) {
            B_shared[i + threadIdx.y][threadIdx.x + 0 * blockDim.x] = cuda_B[B_block_offset + ((block_idx * K_BLOCK + i + threadIdx.y) * pitch_B) + 0 * blockDim.x + threadIdx.x];
            B_shared[i + threadIdx.y][threadIdx.x + 1 * blockDim.x] = cuda_B[B_block_offset + ((block_idx * K_BLOCK + i + threadIdx.y) * pitch_B) + 1 * blockDim.x + threadIdx.x];
            B_shared[i + threadIdx.y][threadIdx.x + 2 * blockDim.x] = cuda_B[B_block_offset + ((block_idx * K_BLOCK + i + threadIdx.y) * pitch_B) + 2 * blockDim.x + threadIdx.x];
            B_shared[i + threadIdx.y][threadIdx.x + 3 * blockDim.x] = cuda_B[B_block_offset + ((block_idx * K_BLOCK + i + threadIdx.y) * pitch_B) + 3 * blockDim.x + threadIdx.x];
        }

        // load the block from B
        __syncthreads();

        for (int i = 0; i < K_BLOCK; i++) {
            float b0 = B_shared[i][0 * blockDim.x + threadIdx.x];
            float b1 = B_shared[i][1 * blockDim.x + threadIdx.x];
            float b2 = B_shared[i][2 * blockDim.x + threadIdx.x];
            float b3 = B_shared[i][3 * blockDim.x + threadIdx.x];

            float a0 = A_shared[threadIdx.y + 0][i];
            float a1 = A_shared[threadIdx.y + blockDim.y][i];

            Cvalue[0] = fmaf(a0, b0, Cvalue[0]);
            Cvalue[1] = fmaf(a1, b0, Cvalue[1]);

            Cvalue[2] = fmaf(a0, b1, Cvalue[2]);
            Cvalue[3] = fmaf(a1, b1, Cvalue[3]);

            Cvalue[4] = fmaf(a0, b2, Cvalue[4]);
            Cvalue[5] = fmaf(a1, b2, Cvalue[5]);

            Cvalue[6] = fmaf(a0, b3, Cvalue[6]);
            Cvalue[7] = fmaf(a1, b3, Cvalue[7]);

        }
        __syncthreads();

    }


    cuda_C[C_block_offset + (threadIdx.y + 0 * blockDim.y) * pitch_C + (0 * blockDim.x + threadIdx.x)] = Cvalue[0];
    cuda_C[C_block_offset + (threadIdx.y + 1 * blockDim.y) * pitch_C + (0 * blockDim.x + threadIdx.x)] = Cvalue[1];

    cuda_C[C_block_offset + (threadIdx.y + 0 * blockDim.y) * pitch_C + (1 * blockDim.x + threadIdx.x)] = Cvalue[2];
    cuda_C[C_block_offset + (threadIdx.y + 1 * blockDim.y) * pitch_C + (1 * blockDim.x + threadIdx.x)] = Cvalue[3];

    cuda_C[C_block_offset + (threadIdx.y + 0 * blockDim.y) * pitch_C + (2 * blockDim.x + threadIdx.x)] = Cvalue[4];
    cuda_C[C_block_offset + (threadIdx.y + 1 * blockDim.y) * pitch_C + (2 * blockDim.x + threadIdx.x)] = Cvalue[5];

    cuda_C[C_block_offset + (threadIdx.y + 0 * blockDim.y) * pitch_C + (3 * blockDim.x + threadIdx.x)] = Cvalue[6];
    cuda_C[C_block_offset + (threadIdx.y + 1 * blockDim.y) * pitch_C + (3 * blockDim.x + threadIdx.x)] = Cvalue[7];
}
