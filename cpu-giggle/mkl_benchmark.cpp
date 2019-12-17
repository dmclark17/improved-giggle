#include <stdio.h>
#include <stdlib.h>

#ifdef MKL_ILP64
#include "mkl.h"
#endif

#include "data_manager.h"

#include "cpu_benchmark.h"

void mkl_gemm_execute(GemmRun* run) {
#ifdef MKL_ILP64
    mkl_set_num_threads(4);
    CBLAS_LAYOUT layout;
    CBLAS_TRANSPOSE transa, transb;

    layout = CblasRowMajor;
    transa = CblasNoTrans;
    transb = CblasNoTrans;

    cblas_sgemm(layout, transa, transb,
                (MKL_INT) run->m, (MKL_INT) run->n, (MKL_INT) run->k,
                run->alpha,
                run->a, (MKL_INT) run->lda,
                run->b, (MKL_INT) run->ldb, run->beta,
                run->c, (MKL_INT) run->ldc);
#else
    (void) run;
    printf("MKL not supported\n");
#endif
}
