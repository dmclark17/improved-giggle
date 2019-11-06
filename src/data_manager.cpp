#include <stdio.h>
#include <stdlib.h>

#include "data_manager.h"


int allocate_run(GemmRun** run, int size) {
    (*run) = new GemmRun;

    (*run)->m = size;
    (*run)->n = size;
    (*run)->k = size;

    // let's assume row major format for now. We need to look into padding
    (*run)->lda = (*run)->k;
    (*run)->ldb = (*run)->n;
    (*run)->ldc = (*run)->n;

    (*run)->alpha = 1.0;
    (*run)->beta = 0.0;

    (*run)->a = (float *)calloc( (*run)->lda * (*run)->m, sizeof(float) );
    (*run)->b = (float *)calloc( (*run)->ldb * (*run)->k, sizeof(float) );
    (*run)->c = (float *)calloc( (*run)->ldc * (*run)->m, sizeof(float) );

    if ( (*run)->a == NULL || (*run)->b == NULL || (*run)->c == NULL ) {
        printf("\n Can't allocate memory arrays");
        return 1;
    }

    return 0;
}

void generate_matrix_prod(float* mat, int ld, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            mat[i * ld + j] = i * j;
        }
    }
}

void generate_matrix_diff(float* mat, int ld, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            mat[i * ld + j] = i - j;
        }
    }
}

void generate_matrix_random(float* mat, int ld, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            mat[i * ld + j] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
        }
    }
}

void print_matrix(float* mat, int ld, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%f\t", mat[i * ld + j]);
        }
        printf("\n");
    }
}

void deallocate_run(GemmRun* run) {
    free(run->a);
    free(run->b);
    free(run->c);

    delete run;
}
