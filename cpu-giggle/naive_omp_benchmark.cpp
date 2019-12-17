#include <stdio.h>
#include <stdlib.h>

#include <omp.h>

#include "data_manager.h"

#include "cpu_benchmark.h"


void naiveOMP_CPU_gemm_execute(GemmRun* run) {
    #pragma omp parallel for
    for (unsigned int i = 0; i < run->m; i++) {
        for (unsigned int j = 0; j < run->n; j++) {
            float dot_prod = 0;
            for (unsigned int z = 0; z < run->k; z++) {
                dot_prod += run->a[i * run->lda + z] *
                            run->b[z * run->ldb + j];
            }
            run->c[i* run->ldc + j] *= run->beta;
            run->c[i* run->ldc + j] += run->alpha * dot_prod;
            dot_prod = 0;
        }
    }
}
