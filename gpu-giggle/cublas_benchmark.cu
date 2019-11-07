#include <cstdio>
#include <iostream>
#include <cuda_runtime.h>
#include "cublas_v2.h"

#include "data_manager.h"

#include "gpu_benchmark.h"

void cublass_gemm_execute(GemmRun* run) {

    cublasOperation_t transa, transb;

    transa = CUBLAS_OP_T;
    transb = CUBLAS_OP_T;

    size_t pitch_A, pitch_B, pitch_C;
    float* cuda_A;
    float* cuda_B;
    float* cuda_C;

    cublasStatus_t stat;
    cublasHandle_t handle;
    cudaError_t err;

    stat = cublasCreate(&handle);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("CUBLAS initialization failed\n");
        return;
    }

    cudaMallocPitch(&cuda_A, &pitch_A, run->k * sizeof(float), run->m);
    cudaMallocPitch(&cuda_B, &pitch_B, run->n * sizeof(float), run->k);
    cudaMallocPitch(&cuda_C, &pitch_C, run->n * sizeof(float), run->m);
    if ((err = cudaGetLastError()) != cudaSuccess) {
        printf("%s\n", cudaGetErrorString(err));
        return;
    }

    stat = cublasSetMatrix(run->m, run->k, sizeof(float), run->a, run->lda, cuda_A, pitch_A / sizeof(float));
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("Set A matrix failed\n");
        return;
    }

    stat = cublasSetMatrix(run->k, run->n, sizeof(float), run->b, run->ldb, cuda_B, pitch_B / sizeof(float));
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("Set B matrix failed\n");
        return;
    }


    stat = cublasSgemm(handle, transa, transb,
                       run->m, run->n, run->k,
                       &(run->alpha),
                       cuda_A, pitch_A / sizeof(float),
                       cuda_B, pitch_B / sizeof(float), &(run->beta),
                       cuda_C, pitch_C / sizeof(float));
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("Multiplication failed\n");
        return;
    }

    cublasGetMatrix(run->m, run->n, sizeof(float), cuda_C, pitch_C / sizeof(float),
                    run->c, run->ldc);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("Get matrix failed\n");
        return;
    }

    cudaFree(cuda_A);
    cudaFree(cuda_B);
    cudaFree(cuda_C);
    // std::cout << "Pitch " << pitch_A << " size " << run->k * sizeof(float) << std::endl;
}
