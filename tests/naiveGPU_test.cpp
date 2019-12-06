#include "data_manager.h"

#include "cpu_benchmark.h"
#include "gpu_benchmark.h"

#include "gtest/gtest.h"

TEST(NaiveGPUTests, BigTest) {
    GemmRun* run;
    GemmRun* run_mkl;
    allocate_run(&run, 1024);
    allocate_run(&run_mkl, 1024);

    generate_matrix_random(run->a, run->lda, run->m);
    generate_matrix_random(run->b, run->ldb, run->k);

    // to free laters
    float* temp_a = run_mkl->a;
    float* temp_b = run_mkl->b;

    run_mkl->a = run->a;
    run_mkl->b = run->b;

    mkl_gemm_execute(run_mkl);
    naiveGPU_gemm_execute(run);

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
