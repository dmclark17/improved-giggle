#include <stdio.h>
#include <stdlib.h>

#include "mkl.h"

#include "data_manager.h"

#include "mkl_benchmark.h"

void gemm_execute(GemmRun* run) {
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

}
