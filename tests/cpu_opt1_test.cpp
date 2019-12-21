#include "iostream"

#include "data_manager.h"
#include "cpu_benchmark.h"
#include "test_helper.h"

#include "gtest/gtest.h"

#define MATRIX_SIZE 128
#define THRESHOLD 1e-10


TEST_F(MatrixTest, RandomOpt1CPU) {
    MySetUp(MATRIX_SIZE, RANDOM);

    naiveCPU_gemm_execute(run_truth);
    opt1CPU_gemm_execute(run);

    verify_correctness(THRESHOLD);
}


TEST_F(MatrixTest, FixedOpt1CPU) {
    MySetUp(MATRIX_SIZE, FIXED);

    opt1CPU_gemm_execute(run);
    naiveCPU_gemm_execute(run_truth);

    verify_correctness(THRESHOLD);
}


#ifdef __APPLE__
TEST_F(MatrixTest, AccelerateRandomOpt1CPU) {
    MySetUp(MATRIX_SIZE, RANDOM);

    accelerate_gemm_execute(run_truth);
    opt1CPU_gemm_execute(run);

    verify_correctness(THRESHOLD);
}
#endif


#ifdef __APPLE__
TEST_F(MatrixTest, AccelerateFixedOpt1CPU) {
    MySetUp(MATRIX_SIZE, FIXED);

    accelerate_gemm_execute(run_truth);
    opt1CPU_gemm_execute(run);

    verify_correctness(THRESHOLD);
}
#endif
