#ifndef _IGIGGLE_DM_H_
#define _IGIGGLE_DM_H_

typedef struct GemmRun {
    int m;
    int n;
    int k;
    float* a;
    float* b;
    float* c;
    int lda;
    int ldb;
    int ldc;
    float alpha;
    float beta;
} GemmRun;

int allocate_run(GemmRun** run, int size);

void generate_matrix_prod(float* mat, int ld, int n);

void generate_matrix_diff(float* mat, int ld, int n);

void generate_matrix_random(float* mat, int ld, int n);

void deallocate_run(GemmRun* run);

#endif /* end of include guard: _IGIGGLE_DM_H_ */
