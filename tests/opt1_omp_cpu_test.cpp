#include "iostream"

#include "data_manager.h"
#include "cpu_benchmark.h"
#include "test_helper.h"

#include "gtest/gtest.h"

#define MATRIX_SIZE 2048
#define THRESHOLD 1e-3


TEST_F(MatrixTest, RandomOpt1OMPCPU) {
    MySetUp(MATRIX_SIZE, RANDOM);

    opt3CPU_gemm_execute(run_truth);
    opt1OMP_CPU_gemm_execute(run);

    verify_correctness(THRESHOLD);
}


TEST_F(MatrixTest, FixedOpt1OMPCPU) {
    MySetUp(MATRIX_SIZE, FIXED);

    opt3CPU_gemm_execute(run_truth);
    opt1OMP_CPU_gemm_execute(run);

    verify_correctness(THRESHOLD);
}
