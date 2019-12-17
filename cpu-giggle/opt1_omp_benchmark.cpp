#include <stdio.h>
#include <stdlib.h>

#include <immintrin.h>
#include <omp.h>

#include "data_manager.h"

#include "cpu_benchmark.h"

#define NUM_THREADS 4


void opt1OMP_CPU_gemm_execute(GemmRun* run) {
    #pragma omp parallel for num_threads(NUM_THREADS)
    for (unsigned int i = 0; i < 4; i++) {
        GemmRun subrun;
        subrun.alpha = run->alpha;
        subrun.beta = run->beta;

        subrun.lda = run->lda;
        subrun.ldb = run->ldb;
        subrun.ldc = run->ldc;

        subrun.m = run->m / 2;
        subrun.n = run->n / 2;
        subrun.k = run->k;

        unsigned int x = i / 2;
        unsigned int y = i % 2;

        subrun.a = run->a + (y * subrun.m * subrun.lda);
        subrun.b = run->b + (x * subrun.n);
        subrun.c = run->c + (x * subrun.n) + (y * subrun.m * subrun.ldc);

        opt3CPU_gemm_execute(&subrun);
    }
}
