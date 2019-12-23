#ifndef _IGIGGLE_TEST_HELPER_H_
#define _IGIGGLE_TEST_HELPER_H_

#include "gtest/gtest.h"
#include "data_manager.h"


typedef enum MatrixType {
    RANDOM,
    FIXED,
} MatrixType;

template <class T>
class MatrixTest : public ::testing::Test {
  protected:
    void MySetUp(unsigned int matrix_size, MatrixType matrix_type);
    void TearDown() override;
    void verify_correctness(T threshold);

    GemmRun<T>* run;
    GemmRun<T>* run_truth;
    T* temp_a;
    T* temp_b;
};

#endif /* end of include guard: _IGIGGLE_TEST_HELPER_H_ */
