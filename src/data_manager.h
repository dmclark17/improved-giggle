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

template <typename T>
void generate_matrix_prod(T* mat, unsigned int ld, unsigned int n);

template <typename T>
void generate_matrix_diff(T* mat, unsigned int ld, unsigned int n);

template <typename T>
void generate_matrix_random(T* mat, unsigned int ld, unsigned int n);

void print_matrix(float* mat, unsigned int ld, unsigned int n);

void print_panel(float* mat, unsigned int ld, unsigned int m, unsigned int n);

template <typename T>
void deallocate_run(GemmRun<T>* run);

#endif /* end of include guard: _IGIGGLE_DM_H_ */
