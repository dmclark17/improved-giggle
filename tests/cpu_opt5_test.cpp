#include "iostream"

#include "data_manager.h"
#include "cpu_benchmark.h"
#include "test_helper.h"

#include "gtest/gtest.h"

#define MATRIX_SIZE 1024
#define THRESHOLD 1e-3


using MyTypes = ::testing::Types<float>;
TYPED_TEST_SUITE(MatrixTest, MyTypes);

TYPED_TEST(MatrixTest, RandomOpt5CPU) {
    this->MySetUp(MATRIX_SIZE, RANDOM);

    opt2CPU_gemm_execute(this->run_truth);
    opt5CPU_gemm_execute(this->run);

    this->verify_correctness(THRESHOLD);
}


TYPED_TEST(MatrixTest, FixedOpt5CPU) {
    this->MySetUp(MATRIX_SIZE, FIXED);

    opt2CPU_gemm_execute(this->run_truth);
    opt5CPU_gemm_execute(this->run);

    this->verify_correctness(THRESHOLD);
}


#ifdef __APPLE__
TYPED_TEST(MatrixTest, AccelerateRandomOpt5CPU) {
    this->MySetUp(MATRIX_SIZE, RANDOM);

    opt5CPU_gemm_execute(this->run);
    accelerate_gemm_execute(this->run_truth);


    this->verify_correctness(THRESHOLD);
}


TYPED_TEST(MatrixTest, AccelerateFixedOpt5CPU) {
    this->MySetUp(MATRIX_SIZE, FIXED);

    opt5CPU_gemm_execute(this->run);
    accelerate_gemm_execute(this->run_truth);


    this->verify_correctness(THRESHOLD);
}
#endif
