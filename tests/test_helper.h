#ifndef _IGIGGLE_TEST_HELPER_H_
#define _IGIGGLE_TEST_HELPER_H_

#include "gtest/gtest.h"
#include "data_manager.h"


typedef enum MatrixType {
    RANDOM,
    FIXED,
} MatrixType;


class MatrixTest : public ::testing::Test {
  protected:
    void MySetUp(unsigned int matrix_size, MatrixType matrix_type);
    void TearDown() override;
    void verify_correctness(float threshold);

    GemmRun<float>* run;
    GemmRun<float>* run_truth;
    float* temp_a;
    float* temp_b;
};

#endif /* end of include guard: _IGIGGLE_TEST_HELPER_H_ */
