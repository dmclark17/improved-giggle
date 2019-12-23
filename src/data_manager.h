#ifndef _IGIGGLE_DM_H_
#define _IGIGGLE_DM_H_


template <typename T>
struct GemmRun {
    unsigned int m;
    unsigned int n;
    unsigned int k;
    T* a;
    T* b;
    T* c;
    unsigned int lda;
    unsigned int ldb;
    unsigned int ldc;
    T alpha;
    T beta;
};

template <typename T>
int allocate_run(GemmRun<T>** run, unsigned int size);

void generate_matrix_prod(float* mat, unsigned int ld, unsigned int n);

void generate_matrix_diff(float* mat, unsigned int ld, unsigned int n);

void generate_matrix_random(float* mat, unsigned int ld, unsigned int n);

void print_matrix(float* mat, unsigned int ld, unsigned int n);

void print_panel(float* mat, unsigned int ld, unsigned int m, unsigned int n);

template <typename T>
void deallocate_run(GemmRun<T>* run);

#endif /* end of include guard: _IGIGGLE_DM_H_ */
