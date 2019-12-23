#include "iostream"

#include "data_manager.h"
#include "cpu_benchmark.h"
#include "test_helper.h"

#include "gtest/gtest.h"

#define MATRIX_SIZE 512
#define THRESHOLD 1e-10


using MyTypes = ::testing::Types<double>;
TYPED_TEST_SUITE(MatrixTest, MyTypes);

TYPED_TEST(MatrixTest, RandomOpt1CPU) {
    this->MySetUp(MATRIX_SIZE, RANDOM);

    naiveCPU_gemm_execute(this->run_truth);
    opt1CPU_gemm_execute(this->run);

    this->verify_correctness(THRESHOLD);
}


TYPED_TEST(MatrixTest, FixedOpt1CPU) {
    this->MySetUp(MATRIX_SIZE, FIXED);

    opt1CPU_gemm_execute(this->run);
    naiveCPU_gemm_execute(this->run_truth);

    this->verify_correctness(THRESHOLD);
}


#ifdef __APPLE__
TYPED_TEST(MatrixTest, AccelerateRandomOpt1CPU) {
    this->MySetUp(MATRIX_SIZE, RANDOM);

    accelerate_gemm_execute(this->run_truth);
    opt1CPU_gemm_execute(this->run);

    this->verify_correctness(THRESHOLD);
}


TYPED_TEST(MatrixTest, AccelerateFixedOpt1CPU) {
    this->MySetUp(MATRIX_SIZE, FIXED);

    accelerate_gemm_execute(this->run_truth);
    opt1CPU_gemm_execute(this->run);

    this->verify_correctness(THRESHOLD);
}
#endif
