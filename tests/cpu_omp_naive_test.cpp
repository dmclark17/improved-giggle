#include "iostream"

#include "data_manager.h"
#include "cpu_benchmark.h"
#include "test_helper.h"

#include "gtest/gtest.h"

#define MATRIX_SIZE 512
#define THRESHOLD 1e-3


TEST_F(MatrixTest, RandomNaiveOMPCPU) {
    MySetUp(MATRIX_SIZE, RANDOM);

    naiveCPU_gemm_execute(run_truth);
    naiveOMP_CPU_gemm_execute(run);

    verify_correctness(THRESHOLD);
}


TEST_F(MatrixTest, FixedNaiveOMPCPU) {
    MySetUp(MATRIX_SIZE, FIXED);

    naiveCPU_gemm_execute(run_truth);
    naiveOMP_CPU_gemm_execute(run);

    verify_correctness(THRESHOLD);
}
