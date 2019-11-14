#ifndef _IGIGGLE_DM_H_
#define _IGIGGLE_DM_H_

typedef struct GemmRun {
    unsigned int m;
    unsigned int n;
    unsigned int k;
    float* a;
    float* b;
    float* c;
    unsigned int lda;
    unsigned int ldb;
    unsigned int ldc;
    float alpha;
    float beta;
} GemmRun;

int allocate_run(GemmRun** run, unsigned int size);

void generate_matrix_prod(float* mat, unsigned int ld, unsigned int n);

void generate_matrix_diff(float* mat, unsigned int ld, unsigned int n);

void generate_matrix_random(float* mat, unsigned int ld, unsigned int n);

void print_matrix(float* mat, unsigned int ld, unsigned int n);

void print_panel(float* mat, unsigned int ld, unsigned int m, unsigned int n);

void deallocate_run(GemmRun* run);

#endif /* end of include guard: _IGIGGLE_DM_H_ */
