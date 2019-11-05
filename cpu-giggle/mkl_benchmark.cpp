#include <stdio.h>
#include <stdlib.h>

#include "mkl.h"


void gemm_execute() {
    CBLAS_LAYOUT layout;
    CBLAS_TRANSPOSE transa, transb;
    MKL_INT         m, n, k;
    MKL_INT         lda, ldb, ldc;
    float           alpha, beta;
    float          *a, *b, *c;

    layout = CblasRowMajor;
    transa = CblasNoTrans;
    transb = CblasNoTrans;

    m = 4;
    n = 4;
    k = 4;

    // let's assume row major format for now
    lda = k;
    ldb = n;
    ldc = n;

    alpha = 1.0;
    beta = 0.0;

    a = (float *)calloc( lda*m, sizeof(float) );
    b = (float *)calloc( ldb*k, sizeof(float) );
    c = (float *)calloc( ldc*m, sizeof(float) );

    if ( a == NULL || b == NULL || c == NULL ) {
        printf("\n Can't allocate memory arrays");
        return;
    }

    for (MKL_INT i = 0; i < m; i++) {
        for (MKL_INT j = 0; j < k; j++) {
            a[i * lda + j] = i * j;
        }
    }

    for (MKL_INT i = 0; i < k; i++) {
        for (MKL_INT j = 0; j < n; j++) {
            b[i * ldb + j] = i - j;
        }
    }

    cblas_sgemm(layout, transa, transb, m, n, k, alpha,
                a, lda,
                b, ldb, beta,
                c, ldc);

    printf("OUTPUT DATA\n");
    for (MKL_INT i = 0; i < m; i++) {
        for (MKL_INT j = 0; j < n; j++) {
            printf("%f ", c[i * ldc + j]);
        }
        printf("\n");
    }

    free(a);
    free(b);
    free(c);
    }
