#include <cstdio>
#include <iostream>
#include <cuda_runtime.h>
#include "cublas_v2.h"

#include "data_manager.h"
#include "gpu_util.h"

#include "gpu_benchmark.h"

void cublass_gemm_execute(GemmRun<float>* run) {

    cublasOperation_t transa, transb;

    transa = CUBLAS_OP_T;
    transb = CUBLAS_OP_T;

    size_t pitch_A, pitch_B, pitch_C;
    float* cuda_A;
    float* cuda_B;
    float* cuda_C;

    cublasStatus_t stat;
    cublasHandle_t handle;

    stat = cublasCreate(&handle);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("CUBLAS initialization failed\n");
        return;
    }

    init_cuda_matrices(run, &pitch_A, &pitch_B, &pitch_C, &cuda_A, &cuda_B, &cuda_C);

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

    deinit_cuda_matrices(run, pitch_C, cuda_A, cuda_B, cuda_C);
}
