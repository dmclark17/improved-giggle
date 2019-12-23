#include "gtest/gtest.h"
#include "data_manager.h"

#include "test_helper.h"


template <class T>
void MatrixTest<T>::MySetUp(unsigned int matrix_size, MatrixType matrix_type) {
    allocate_run(&run, matrix_size);
    allocate_run(&run_truth, matrix_size);

    if (matrix_type == RANDOM) {
        generate_matrix_random(run->a, run->lda, run->m);
        generate_matrix_random(run->b, run->ldb, run->k);
    } else {
        generate_matrix_prod(run->a, run->lda, run->m);
        generate_matrix_diff(run->b, run->ldb, run->k);
    }


    temp_a = run_truth->a;
    temp_b = run_truth->b;

    run_truth->a = run->a;
    run_truth->b = run->b;
}


template <class T>
void MatrixTest<T>::TearDown() {
    run_truth->a = temp_a;
    run_truth->b = temp_b;

    deallocate_run(run);
    deallocate_run(run_truth);
}


template <class T>
void MatrixTest<T>::verify_correctness(T threshold) {
    for (unsigned int i = 0; i < run->m; i++) {
        for (unsigned int j = 0; j < run->n; j++) {
            ASSERT_NEAR(run->c[i * run->ldc + j],
                        run_truth->c[i * run->ldc + j],
                        threshold);
        }
    }
}

template class MatrixTest<float>;
template class MatrixTest<double>;
