#include <stdio.h>
#include <stdlib.h>

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

    if ( (*run)->a == NULL || (*run)->b == NULL || (*run)->c == NULL ) {
        printf("\nCan't allocate alligned memory arrays\n");
        (*run)->a = (T *)malloc( (*run)->ldc * (*run)->m * sizeof(T) );
        (*run)->b = (T *)malloc( (*run)->ldc * (*run)->m * sizeof(T) );
        (*run)->c = (T *)malloc( (*run)->ldc * (*run)->m * sizeof(T) );
    }

    if ( (*run)->a == NULL || (*run)->b == NULL || (*run)->c == NULL ) {
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

void generate_matrix_prod(float* mat, unsigned int ld, unsigned int n) {
    for (unsigned int i = 0; i < n; i++) {
        for (unsigned int j = 0; j < n; j++) {
            mat[i * ld + j] = ((float) i) * ((float) j);
        }
    }
}

void generate_matrix_diff(float* mat, unsigned int ld, unsigned int n) {
    for (unsigned int i = 0; i < n; i++) {
        for (unsigned int j = 0; j < n; j++) {
            mat[i * ld + j] = ((float) i) - ((float) j);
        }
    }
}

void generate_matrix_random(float* mat, unsigned int ld, unsigned int n) {
    for (unsigned int i = 0; i < n; i++) {
        for (unsigned int j = 0; j < n; j++) {
            mat[i * ld + j] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
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
template void deallocate_run<float>(GemmRun<float>* run);
