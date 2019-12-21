#include <stdio.h>
#include <stdlib.h>

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#endif

#include "data_manager.h"

#include "cpu_benchmark.h"

void accelerate_gemm_execute(GemmRun* run) {
#ifdef __APPLE__
    CBLAS_ORDER layout;
    CBLAS_TRANSPOSE transa, transb;

    layout = CblasRowMajor;
    transa = CblasNoTrans;
    transb = CblasNoTrans;

    cblas_sgemm(layout, transa, transb,
                run->m,  run->n,  run->k,
                run->alpha,
                run->a, run->lda,
                run->b, run->ldb, run->beta,
                run->c, run->ldc);
#else
    (void) run;
#endif
}
