#include <stdio.h>
#include <stdlib.h>

#include "data_manager.h"

#include "cpu_benchmark.h"


void naive_gemm_execute(GemmRun* run) {
    float dot_prod = 0;
    for (int i = 0; i < run->m; i++) {
        for (int j = 0; j < run->n; j++) {
            for (int z = 0; z < run->k; z++) {
                dot_prod += run->a[i * run->lda + z] *
                            run->b[z * run->ldb + j];
            }
            run->c[i* run->ldc + j] *= run->beta;
            run->c[i* run->ldc + j] += run->alpha * dot_prod;
            dot_prod = 0;
        }
    }
}
