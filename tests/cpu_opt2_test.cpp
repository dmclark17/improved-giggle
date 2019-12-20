#include "iostream"

#include "data_manager.h"
#include "cpu_benchmark.h"
#include "test_helper.h"

#include "gtest/gtest.h"

#define MATRIX_SIZE 512
#define THRESHOLD 1e-3


TEST_F(MatrixTest, RandomOpt2CPU) {
    MySetUp(MATRIX_SIZE, RANDOM);

    naiveCPU_gemm_execute(run_truth);
    opt2CPU_gemm_execute(run);

    verify_correctness(THRESHOLD);
}


TEST_F(MatrixTest, FixedOpt2CPU) {
    MySetUp(MATRIX_SIZE, FIXED);

    naiveCPU_gemm_execute(run_truth);
    opt2CPU_gemm_execute(run);

    verify_correctness(THRESHOLD);
}
