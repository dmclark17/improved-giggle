#include <cstdio>
#include <cstdlib>

#include <immintrin.h>

// #include <xmmintrin.h>
// _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);

#include "data_manager.h"

#include "cpu_benchmark.h"


#define NC 64
#define KC 512
#define MR 4
#define NR 8


inline void opt4CPU_packA(GemmRun<float>* run, unsigned int p, float* a_pack) {
    /*
      utility function for packing A
      - Packs the A matrix into a panel of size m by k_c
      - Within the packed array, the data should be in row major format
    */
    for (unsigned int i = 0; i < run->m; i += 4) {
        for (unsigned int j = 0; j < KC; j += 2) {

            for (unsigned int inner_j = 0; inner_j < 2; inner_j++) {
                for (unsigned int inner_i = 0; inner_i < 4; inner_i++) {
                    a_pack[i * KC + (j + inner_j) * 4 + inner_i] = run->a[p + run->lda * (i + inner_i) + (j + inner_j)];
                }
            }
        }
    }
}


inline void opt4CPU_packB(GemmRun<float>* run, unsigned int p, unsigned int j, float* b_pack) {
    /*
      utility function for packing B
      - Packs B into a block of size n_c and k_c
      - Within the packed array, the data should be in column major format
      maybe this should be some sort of block form, but we can start with pure
      column major format for now?
    */
    unsigned int offset = run->ldb * p + j;
    unsigned int block_index, global_index;
    for (unsigned int pack_j = 0; pack_j < NC; pack_j += NR) {
        for (unsigned int pack_i = 0; pack_i < KC; pack_i += 2) {

            for (unsigned int other_j = 0; other_j < NR; other_j += 4) {

                // These 8 values will be in one register
                for (unsigned int inner_i = 0; inner_i < 2; inner_i++) {
                    for (unsigned int inner_j = 0; inner_j < 4; inner_j++) {
                        block_index = (pack_j * KC) + (pack_i * NR) + (inner_i * 4) + other_j * 2 + inner_j;
                        global_index = offset + (pack_i + inner_i) * run->ldb + (pack_j + other_j + inner_j);
                        b_pack[block_index] = run->b[global_index];
                    }
                }
            }
        }
    }
}


inline void opt4CPU_unpackC(GemmRun<float>* run, unsigned int j, unsigned int i, float* c_pack) {
    /*
      utility function for unpacking C
      - Unpacks a m_r by n_c submatrix into C
    */
    unsigned int offset = i * run->ldc + j;
    for (unsigned int pack_i = 0; pack_i < MR; pack_i += MR) {
        for (unsigned int pack_j = 0; pack_j < NC; pack_j += 4) {

            run->c[offset + (pack_i * run->ldc) + pack_j + (0 * run->ldc) + 0] += c_pack[pack_j * MR + 0];
            run->c[offset + (pack_i * run->ldc) + pack_j + (1 * run->ldc) + 1] += c_pack[pack_j * MR + 1];
            run->c[offset + (pack_i * run->ldc) + pack_j + (2 * run->ldc) + 2] += c_pack[pack_j * MR + 2];
            run->c[offset + (pack_i * run->ldc) + pack_j + (3 * run->ldc) + 3] += c_pack[pack_j * MR + 3];

            run->c[offset + (pack_i * run->ldc) + pack_j + (3 * run->ldc) + 0] += c_pack[pack_j * MR + 4];
            run->c[offset + (pack_i * run->ldc) + pack_j + (0 * run->ldc) + 1] += c_pack[pack_j * MR + 5];
            run->c[offset + (pack_i * run->ldc) + pack_j + (1 * run->ldc) + 2] += c_pack[pack_j * MR + 6];
            run->c[offset + (pack_i * run->ldc) + pack_j + (2 * run->ldc) + 3] += c_pack[pack_j * MR + 7];

            run->c[offset + (pack_i * run->ldc) + pack_j + (2 * run->ldc) + 0] += c_pack[pack_j * MR + 8];
            run->c[offset + (pack_i * run->ldc) + pack_j + (3 * run->ldc) + 1] += c_pack[pack_j * MR + 9];
            run->c[offset + (pack_i * run->ldc) + pack_j + (0 * run->ldc) + 2] += c_pack[pack_j * MR + 10];
            run->c[offset + (pack_i * run->ldc) + pack_j + (1 * run->ldc) + 3] += c_pack[pack_j * MR + 11];

            run->c[offset + (pack_i * run->ldc) + pack_j + (1 * run->ldc) + 0] += c_pack[pack_j * MR + 12];
            run->c[offset + (pack_i * run->ldc) + pack_j + (2 * run->ldc) + 1] += c_pack[pack_j * MR + 13];
            run->c[offset + (pack_i * run->ldc) + pack_j + (3 * run->ldc) + 2] += c_pack[pack_j * MR + 14];
            run->c[offset + (pack_i * run->ldc) + pack_j + (0 * run->ldc) + 3] += c_pack[pack_j * MR + 15];
        }
    }
}


inline void opt4CPU_aux_simd(float* a_pack, float* b_pack, float* c_pack) {
    /*
      - a_pack should be in row major format
      - b_pack should be in column major
      - c_pack will be in row major?!
    */
    __m256 a0, a0_p;
    __m256 b0, b1;
    __m256 c0, c1, c2, c3, c4, c5, c6, c7;
    __m128 sum0, sum1, sum2, sum3, sum4, sum5, sum6, sum7;


    const __m256i perm1 = _mm256_setr_epi32(3, 0, 1, 2, 3, 0, 1, 2);
    const __m256i perm2 = _mm256_setr_epi32(2, 3, 0, 1, 2, 3, 0, 1);
    const __m256i perm3 = _mm256_setr_epi32(1, 2, 3, 0, 1, 2, 3, 0);

    unsigned int pack_n, pack_z;

    for (pack_n = 0; pack_n < NC; pack_n += NR) {

        c0 =  _mm256_setzero_ps();
        c1 =  _mm256_setzero_ps();
        c2 =  _mm256_setzero_ps();
        c3 =  _mm256_setzero_ps();

        c4 =  _mm256_setzero_ps();
        c5 =  _mm256_setzero_ps();
        c6 =  _mm256_setzero_ps();
        c7 =  _mm256_setzero_ps();

        for (pack_z = 0; pack_z < KC; pack_z += 2) {
            a0_p = _mm256_load_ps(a_pack + (pack_z) * 4 + (0));
            b0 = _mm256_load_ps(b_pack + (pack_n * KC) + (pack_z) * NR + (0));
            b1 = _mm256_load_ps(b_pack + (pack_n * KC) + (pack_z) * NR + (8));

            c0 = _mm256_fmadd_ps(a0_p, b0, c0);
            c4 = _mm256_fmadd_ps(a0_p, b1, c4);

            a0 = _mm256_permutevar_ps(a0_p, perm1);
            c1 = _mm256_fmadd_ps(a0, b0, c1);
            c5 = _mm256_fmadd_ps(a0, b1, c5);


            a0 = _mm256_permutevar_ps(a0_p, perm2);
            c2 = _mm256_fmadd_ps(a0, b0, c2);
            c6 = _mm256_fmadd_ps(a0, b1, c6);

            a0 = _mm256_permutevar_ps(a0_p, perm3);
            c3 = _mm256_fmadd_ps(a0, b0, c3);
            c7 = _mm256_fmadd_ps(a0, b1, c7);
        }

        sum0 = _mm256_extractf128_ps(c0, 1);
        sum1 = _mm256_extractf128_ps(c1, 1);
        sum2 = _mm256_extractf128_ps(c2, 1);
        sum3 = _mm256_extractf128_ps(c3, 1);

        sum4 = _mm256_extractf128_ps(c0, 0);
        sum5 = _mm256_extractf128_ps(c1, 0);
        sum6 = _mm256_extractf128_ps(c2, 0);
        sum7 = _mm256_extractf128_ps(c3, 0);

        sum0 = _mm_add_ps(sum0, sum4);
        sum1 = _mm_add_ps(sum1, sum5);
        sum2 = _mm_add_ps(sum2, sum6);
        sum3 = _mm_add_ps(sum3, sum7);

        _mm_store_ps(c_pack + pack_n * MR + (0) * 4, sum0);
        _mm_store_ps(c_pack + pack_n * MR + (1) * 4, sum1);
        _mm_store_ps(c_pack + pack_n * MR + (2) * 4, sum2);
        _mm_store_ps(c_pack + pack_n * MR + (3) * 4, sum3);

        // Round two
        sum0 = _mm256_extractf128_ps(c4, 1);
        sum1 = _mm256_extractf128_ps(c5, 1);
        sum2 = _mm256_extractf128_ps(c6, 1);
        sum3 = _mm256_extractf128_ps(c7, 1);

        sum4 = _mm256_extractf128_ps(c4, 0);
        sum5 = _mm256_extractf128_ps(c5, 0);
        sum6 = _mm256_extractf128_ps(c6, 0);
        sum7 = _mm256_extractf128_ps(c7, 0);

        sum0 = _mm_add_ps(sum0, sum4);
        sum1 = _mm_add_ps(sum1, sum5);
        sum2 = _mm_add_ps(sum2, sum6);
        sum3 = _mm_add_ps(sum3, sum7);

        _mm_store_ps(c_pack + pack_n * MR + (4) * 4, sum0);
        _mm_store_ps(c_pack + pack_n * MR + (5) * 4, sum1);
        _mm_store_ps(c_pack + pack_n * MR + (6) * 4, sum2);
        _mm_store_ps(c_pack + pack_n * MR + (7) * 4, sum3);
    }
}


inline void opt4CPU_gepb(GemmRun<float>* run, unsigned int p, unsigned int j, float* a_pack, float* b_pack, float* c_pack) {
    /*
      This should
      - pack B
      - iterate over A and C
    */
    opt4CPU_packB(run, p, j, b_pack);
    unsigned int i;
    for (i = 0; i < run->m; i += MR) {
        opt4CPU_aux_simd(a_pack + (i * KC), b_pack, c_pack);
        opt4CPU_unpackC(run, j, i, c_pack);
    }
}


inline void opt4CPU_gepp(GemmRun<float>* run, unsigned int p, float* a_pack, float* b_pack, float* c_pack) {
    /*
      This should
      - pack A: A is in row major format,
      - iterate over B
    */
    opt4CPU_packA(run, p, a_pack);
    for (unsigned int j = 0; j < run->n; j += NC) {
        opt4CPU_gepb(run, p, j, a_pack, b_pack, c_pack);
    }
}


template <typename T>
void opt4CPU_gemm_execute(GemmRun<T>* run) {
    /*
      This should call gepp iteration over panels. Panel A_p and B_p will
      make a contribution to all of C?
    */
    float* a_pack = (float *)aligned_alloc( 64, KC * run->m * sizeof(float) );
    float* b_pack = (float *)aligned_alloc( 64, KC * NC * sizeof(float) );
    float* c_pack = (float *)aligned_alloc( 64, MR * NC * sizeof(float) );

    __m256 zero_vec = _mm256_setzero_ps();
    for (unsigned int i = 0; i < MR * NC; i += 8) {
        _mm256_store_ps(c_pack + i, zero_vec);
    }

    for (unsigned int p = 0; p < run->k; p += KC) {
        opt4CPU_gepp(run, p, a_pack, b_pack, c_pack);
    }

    free(a_pack);
    free(b_pack);
    free(c_pack);
}

template void opt4CPU_gemm_execute<float>(GemmRun<float>*);
