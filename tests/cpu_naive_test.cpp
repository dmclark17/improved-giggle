#include "data_manager.h"
#include "cpu_benchmark.h"
#include "test_helper.h"

#include "gtest/gtest.h"

#define MATRIX_SIZE 256
#define THRESHOLD 1e-10


using MyTypes = ::testing::Types<float,double>;
TYPED_TEST_SUITE(MatrixTest, MyTypes);

TEST(NaiveTests, ThreeByThree) {
    GemmRun<float>* run;
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


#ifdef __APPLE__
TYPED_TEST(MatrixTest, RandomNaiveCPUAccelerate) {
    this->MySetUp(MATRIX_SIZE, RANDOM);

    naiveCPU_gemm_execute(this->run_truth);
    accelerate_gemm_execute(this->run);

    this->verify_correctness(THRESHOLD);
}


TYPED_TEST(MatrixTest, FixedNaiveCPUAccelerate) {
    this->MySetUp(MATRIX_SIZE, FIXED);

    naiveCPU_gemm_execute(this->run_truth);
    accelerate_gemm_execute(this->run);

    this->verify_correctness(THRESHOLD);
}
#endif
