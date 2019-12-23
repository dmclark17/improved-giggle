#include "iostream"

#include "data_manager.h"
#include "cpu_benchmark.h"
#include "test_helper.h"

#include "gtest/gtest.h"

#define MATRIX_SIZE 512
#define THRESHOLD 1e-3


using MyTypes = ::testing::Types<float,double>;
TYPED_TEST_SUITE(MatrixTest, MyTypes);

TYPED_TEST(MatrixTest, RandomNaiveOMPCPU) {
    this->MySetUp(MATRIX_SIZE, RANDOM);

    naiveCPU_gemm_execute(this->run_truth);
    naiveOMP_CPU_gemm_execute(this->run);

    this->verify_correctness(THRESHOLD);
}


TYPED_TEST(MatrixTest, FixedNaiveOMPCPU) {
    this->MySetUp(MATRIX_SIZE, FIXED);

    naiveCPU_gemm_execute(this->run_truth);
    naiveOMP_CPU_gemm_execute(this->run);

    this->verify_correctness(THRESHOLD);
}

#ifdef __APPLE__
TYPED_TEST(MatrixTest, RandomNaiveOMPCPUAccelerate) {
    this->MySetUp(MATRIX_SIZE, RANDOM);

    naiveOMP_CPU_gemm_execute(this->run_truth);
    accelerate_gemm_execute(this->run);

    this->verify_correctness(THRESHOLD);
}


TYPED_TEST(MatrixTest, FixedNaiveOMPCPUAccelerate) {
    this->MySetUp(MATRIX_SIZE, FIXED);

    naiveOMP_CPU_gemm_execute(this->run_truth);
    accelerate_gemm_execute(this->run);

    this->verify_correctness(THRESHOLD);
}
#endif
