#include "iostream"

#include "data_manager.h"
#include "cpu_benchmark.h"
#include "test_helper.h"

#include "gtest/gtest.h"

#define MATRIX_SIZE 1024
#define THRESHOLD 1e-3


using MyTypes = ::testing::Types<float>;
TYPED_TEST_SUITE(MatrixTest, MyTypes);

TYPED_TEST(MatrixTest, RandomOpt3CPU) {
    this->MySetUp(MATRIX_SIZE, RANDOM);

    naiveCPU_gemm_execute(this->run_truth);
    opt3CPU_gemm_execute(this->run);

    this->verify_correctness(THRESHOLD);
}


TYPED_TEST(MatrixTest, FixedOpt3CPU) {
    this->MySetUp(MATRIX_SIZE, FIXED);

    naiveCPU_gemm_execute(this->run_truth);
    opt3CPU_gemm_execute(this->run);

    this->verify_correctness(THRESHOLD);
}


#ifdef __APPLE__
TYPED_TEST(MatrixTest, AccelerateRandomOpt3CPU) {
    this->MySetUp(MATRIX_SIZE, RANDOM);

    accelerate_gemm_execute(this->run_truth);
    opt3CPU_gemm_execute(this->run);

    this->verify_correctness(THRESHOLD);
}


TYPED_TEST(MatrixTest, AccelerateFixedOpt3CPU) {
    this->MySetUp(MATRIX_SIZE, FIXED);

    accelerate_gemm_execute(this->run_truth);
    opt3CPU_gemm_execute(this->run);

    this->verify_correctness(THRESHOLD);
}
#endif
