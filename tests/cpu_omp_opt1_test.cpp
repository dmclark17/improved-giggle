#include "iostream"

#include "data_manager.h"
#include "cpu_benchmark.h"
#include "test_helper.h"

#include "gtest/gtest.h"

#define MATRIX_SIZE 2048
#define THRESHOLD 1e-10


using MyTypes = ::testing::Types<float>;
TYPED_TEST_SUITE(MatrixTest, MyTypes);

TYPED_TEST(MatrixTest, RandomOpt1OMPCPU) {
    this->MySetUp(MATRIX_SIZE, RANDOM);

    opt3CPU_gemm_execute(this->run_truth);
    opt1OMP_CPU_gemm_execute(this->run);

    this->verify_correctness(THRESHOLD);
}


TYPED_TEST(MatrixTest, FixedOpt1OMPCPU) {
    this->MySetUp(MATRIX_SIZE, FIXED);

    opt3CPU_gemm_execute(this->run_truth);
    opt1OMP_CPU_gemm_execute(this->run);

    this->verify_correctness(THRESHOLD);
}


#ifdef __APPLE__
TYPED_TEST(MatrixTest, RandomOpt1OMPCPUAccelerate) {
    this->MySetUp(MATRIX_SIZE, RANDOM);

    accelerate_gemm_execute(this->run_truth);
    opt1OMP_CPU_gemm_execute(this->run);

    this->verify_correctness(THRESHOLD);
}


TYPED_TEST(MatrixTest, FixedOpt1OMPCPUAccelerate) {
    this->MySetUp(MATRIX_SIZE, FIXED);

    accelerate_gemm_execute(this->run_truth);
    opt1OMP_CPU_gemm_execute(this->run);

    this->verify_correctness(THRESHOLD);
}
#endif
