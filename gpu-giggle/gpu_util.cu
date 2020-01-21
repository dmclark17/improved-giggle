#include <cstdio>
#include <cuda_runtime.h>
#include "cublas_v2.h"

#include "data_manager.h"

#include "gpu_util.h"

void init_cuda_matrices(GemmRun<float>* run,
                        size_t* pitch_A, size_t* pitch_B, size_t* pitch_C,
                        float** cuda_A, float** cuda_B, float** cuda_C) {

    cublasStatus_t stat;
    cudaError_t err;

    cudaMallocPitch(cuda_A, pitch_A, run->k * sizeof(float), run->m);
    cudaMallocPitch(cuda_B, pitch_B, run->n * sizeof(float), run->k);
    cudaMallocPitch(cuda_C, pitch_C, run->n * sizeof(float), run->m);
    if ((err = cudaGetLastError()) != cudaSuccess) {
        printf("%s\n", cudaGetErrorString(err));
        return;
    }


    stat = cublasSetMatrix(run->m, run->k, sizeof(float), run->a, run->lda, *cuda_A, *pitch_A / sizeof(float));
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("Set A matrix failed\n");
        return;
    }

    stat = cublasSetMatrix(run->k, run->n, sizeof(float), run->b, run->ldb, *cuda_B, *pitch_B / sizeof(float));
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("Set B matrix failed\n");
        return;
    }
}


void deinit_cuda_matrices(GemmRun<float>* run, size_t pitch_C,
                          float* cuda_A, float* cuda_B, float* cuda_C) {

    cublasStatus_t stat;

    stat = cublasGetMatrix(run->m, run->n, sizeof(float), cuda_C, pitch_C / sizeof(float),
              run->c, run->ldc);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("Get matrix failed\n");
        return;
    }

    cudaFree(cuda_A);
    cudaFree(cuda_B);
    cudaFree(cuda_C);
}
