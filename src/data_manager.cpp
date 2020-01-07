#include <cstdio>
#include <cstdlib>

#include "data_manager.h"


template <typename T>
int allocate_run(GemmRun<T>** run, unsigned int size) {
    (*run) = new GemmRun<T>;

    (*run)->m = size;
    (*run)->n = size;
    (*run)->k = size;

    // let's assume row major format for now. We need to look into padding
    (*run)->lda = (*run)->k;
    (*run)->ldb = (*run)->n;
    (*run)->ldc = (*run)->n;

    (*run)->alpha = 1.0;
    (*run)->beta = 0.0;

    (*run)->a = (T *) aligned_alloc( 64, (*run)->lda * (*run)->m * sizeof(T) );
    (*run)->b = (T *) aligned_alloc( 64, (*run)->ldb * (*run)->k * sizeof(T) );
    (*run)->c = (T *) aligned_alloc( 64, (*run)->ldc * (*run)->m * sizeof(T) );

    if ( (*run)->a == nullptr || (*run)->b == nullptr || (*run)->c == nullptr ) {
        printf("\nCan't allocate aligned memory arrays\n");
        (*run)->a = (T *)malloc( (*run)->ldc * (*run)->m * sizeof(T) );
        (*run)->b = (T *)malloc( (*run)->ldc * (*run)->m * sizeof(T) );
        (*run)->c = (T *)malloc( (*run)->ldc * (*run)->m * sizeof(T) );
    }

    if ( (*run)->a == nullptr || (*run)->b == nullptr || (*run)->c == nullptr ) {
        printf("\nCan't allocate memory arrays\n");
        return 1;
    }

    for (unsigned int i = 0; i < (*run)->m; i++) {
        for (unsigned int j = 0; j < (*run)->n; j++) {
            (*run)->c[i * (*run)->ldc + j] = 0;
        }
    }

    return 0;
}

template <typename T>
void generate_matrix_prod(T* mat, unsigned int ld, unsigned int n) {
    for (unsigned int i = 0; i < n; i++) {
        for (unsigned int j = 0; j < n; j++) {
            mat[i * ld + j] = ((T) i) * ((T) j);
        }
    }
}

template <typename T>
void generate_matrix_diff(T* mat, unsigned int ld, unsigned int n) {
    for (unsigned int i = 0; i < n; i++) {
        for (unsigned int j = 0; j < n; j++) {
            mat[i * ld + j] = ((T) i) - ((T) j);
        }
    }
}

template <typename T>
void generate_matrix_random(T* mat, unsigned int ld, unsigned int n) {
    for (unsigned int i = 0; i < n; i++) {
        for (unsigned int j = 0; j < n; j++) {
            mat[i * ld + j] = static_cast <T> (rand()) / static_cast <T> (RAND_MAX);
        }
    }
}

void print_matrix(float* mat, unsigned int ld, unsigned int n) {
    for (unsigned int i = 0; i < n; i++) {
        for (unsigned int j = 0; j < n; j++) {
            printf("%f\t", mat[i * ld + j]);
        }
        printf("\n");
    }
}

void print_panel(float* mat, unsigned int ld, unsigned int m, unsigned int n) {
    for (unsigned int i = 0; i < n; i++) {
        for (unsigned int j = 0; j < m; j++) {
            printf("%f\t", mat[i * ld + j]);
        }
        printf("\n");
    }
}

template <typename T>
void deallocate_run(GemmRun<T>* run) {
    free(run->a);
    free(run->b);
    free(run->c);

    delete run;
}


template int allocate_run<float>(GemmRun<float>**, unsigned int);
template int allocate_run<double>(GemmRun<double>**, unsigned int);

template void generate_matrix_prod<float>(float*, unsigned int, unsigned int);
template void generate_matrix_prod<double>(double*, unsigned int, unsigned int);

template void generate_matrix_diff<float>(float*, unsigned int, unsigned int);
template void generate_matrix_diff<double>(double*, unsigned int, unsigned int);

template void generate_matrix_random<float>(float*, unsigned int, unsigned int);
template void generate_matrix_random<double>(double*, unsigned int, unsigned int);

template void deallocate_run<float>(GemmRun<float>* run);
template void deallocate_run<double>(GemmRun<double>* run);
