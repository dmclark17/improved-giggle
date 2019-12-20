#include "data_manager.h"
#include "cpu_benchmark.h"

#include "gtest/gtest.h"

TEST(NaiveTests, ThreeByThree) {
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

    naiveCPU_gemm_execute(run);

    for (unsigned int i = 0; i < run->m; i++) {
        for (unsigned int j = 0; j < run->n; j++) {
            ASSERT_NEAR(run->c[i * run->ldc + j], expected[i * run->ldc + j],
                        1e-10);
        }
    }

    free(expected);
    deallocate_run(run);
}

TEST(NaiveTests, BigTest) {
    GemmRun* run;
    GemmRun* run_mkl;
    allocate_run(&run, 512);
    allocate_run(&run_mkl, 512);

    generate_matrix_random(run->a, run->lda, run->m);
    generate_matrix_random(run->b, run->ldb, run->k);

    // to free laters
    float* temp_a = run_mkl->a;
    float* temp_b = run_mkl->b;

    run_mkl->a = run->a;
    run_mkl->b = run->b;

    mkl_gemm_execute(run_mkl);
    naiveCPU_gemm_execute(run);

    for (unsigned int i = 0; i < run->m; i++) {
        for (unsigned int j = 0; j < run->n; j++) {
            ASSERT_NEAR(run->c[i * run->ldc + j], run_mkl->c[i * run->ldc + j],
                        1e-3);
        }
    }

    run_mkl->a = temp_a;
    run_mkl->b = temp_b;

    deallocate_run(run);
    deallocate_run(run_mkl);
}
