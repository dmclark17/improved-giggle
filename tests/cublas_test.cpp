#include "data_manager.h"
#include "cublas_benchmark.h"

#include "gtest/gtest.h"

#define IDX2C(i,j,ld) (((j)*(ld))+(i))

TEST(CUBLASTests, ThreeByThree) {
    GemmRun* run;
    allocate_run(&run, 3);

    float* expected = (float *)calloc(run->ldc * run->m, sizeof(float) );
    expected[0] = 0.0;
    expected[1] = 0.0;
    expected[2] = 0.0;
    expected[1*run->ldc + 0] = 5.0;
    expected[1*run->ldc + 1] = 2.0;
    expected[1*run->ldc + 2] = -1.0;
    expected[2*run->ldc + 0] = 10.0;
    expected[2*run->ldc + 1] = 4.0;
    expected[2*run->ldc + 2] = -2.0;

    generate_matrix_prod(run->a, run->lda, run->m);
    generate_matrix_diff(run->b, run->ldb, run->k);

    cublass_gemm_execute(run);

    for (int i = 0; i < run->m; i++) {
        for (int j = 0; j < run->n; j++) {
            printf("%f\t", run->c[i * run->ldc + j]);
        }
        printf("\n");
    }

    for (int i = 0; i < run->m; i++) {
        for (int j = 0; j < run->n; j++) {
            ASSERT_TRUE(run->c[IDX2C(i, j, run->ldc)] == expected[i * run->ldc + j]);
        }
    }

    free(expected);
    deallocate_run(run);
}
