#include "iostream"

#include "data_manager.h"
#include "cpu_benchmark.h"
#include "test_helper.h"

#include "gtest/gtest.h"

#define MATRIX_SIZE 1024
#define THRESHOLD 1e-3


using MyTypes = ::testing::Types<float>;
TYPED_TEST_SUITE(MatrixTest, MyTypes);

TYPED_TEST(MatrixTest, RandomOpt4CPU) {
    this->MySetUp(MATRIX_SIZE, RANDOM);

    opt2CPU_gemm_execute(this->run_truth);
    opt4CPU_gemm_execute(this->run);

    this->verify_correctness(THRESHOLD);
}


TYPED_TEST(MatrixTest, FixedOpt4CPU) {
    this->MySetUp(MATRIX_SIZE, FIXED);

    opt2CPU_gemm_execute(this->run_truth);
    opt4CPU_gemm_execute(this->run);

    this->verify_correctness(THRESHOLD);
}


#ifdef __APPLE__
TYPED_TEST(MatrixTest, AccelerateRandomOpt4CPU) {
    this->MySetUp(MATRIX_SIZE, RANDOM);

    opt4CPU_gemm_execute(this->run);
    accelerate_gemm_execute(this->run_truth);


    this->verify_correctness(THRESHOLD);
}


TYPED_TEST(MatrixTest, AccelerateFixedOpt4CPU) {
    this->MySetUp(MATRIX_SIZE, FIXED);

    opt4CPU_gemm_execute(this->run);
    accelerate_gemm_execute(this->run_truth);


    this->verify_correctness(THRESHOLD);
}
#endif
