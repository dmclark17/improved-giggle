#include <stdio.h>
#include <stdlib.h>

#include <immintrin.h>

#include "data_manager.h"

#include "cpu_benchmark.h"


#define NC 128
#define KC 128
#define MR 32
#define NR 1


inline void opt2CPU_packA(GemmRun* run, unsigned int p, float* a_pack) {
    /*
      utility function for packing A
      - Packs the A matrix into a panel of size m by k_c
      - Within the packed array, the data should be in row major format
    */
    __m256 src;
    for (unsigned int i = 0; i < run->m; i++) {
        for (unsigned int j = 0; j < KC; j += 8) {
            src = _mm256_load_ps(run->a + (p + run->lda * i + j));
            _mm256_store_ps(a_pack + (i * KC + j), src);
        }
    }
}


inline void opt2CPU_packB(GemmRun* run, unsigned int p, unsigned int j, float* b_pack) {
    /*
      utility function for packing B
      - Packs B into a block of size n_c and k_c
      - Within the packed array, the data should be in column major format
      maybe this should be some sort of block form, but we can start with pure
      column major format for now?
    */
    unsigned int offset = run->ldb * p + j;
    __m256 src;
    for (unsigned int pack_i = 0; pack_i < KC; pack_i++) {
        for (unsigned int pack_j = 0; pack_j < NC; pack_j += 8) {
            src = _mm256_load_ps(run->b + offset + pack_i * run->ldb + pack_j);
            _mm256_store_ps(b_pack + pack_i * NC + pack_j, src);
        }
    }
}


inline void opt2CPU_unpackC(GemmRun* run, unsigned int j, unsigned int i, float* c_pack) {
    /*
      utility function for unpacking C
      - Unpacks a m_r by n_c submatrix into C
    */
    unsigned int offset = i * run->ldc + j;
    __m256 m_cpack, m_c, m_sum, zero_vec;
    zero_vec = _mm256_setzero_ps();
    for (unsigned int pack_i = 0; pack_i < MR; pack_i++) {
        for (unsigned int pack_j = 0; pack_j < NC; pack_j += 8) {
            m_cpack = _mm256_load_ps(c_pack + pack_i * NC + pack_j);
            m_c = _mm256_load_ps(run->c + offset + pack_i * run->ldc + pack_j);

            m_sum = _mm256_add_ps(m_cpack, m_c);

            _mm256_store_ps(run->c + offset + pack_i * run->ldc + pack_j, m_sum);
            _mm256_store_ps(c_pack + pack_i * NC + pack_j, zero_vec);
        }
    }
}


inline void opt2CPU_aux(float* a_pack, float* b_pack, float* c_pack) {
    /*
      - a_pack should be in row major format
      - b_pack should be in column major
      - c_pack will be in row major?!
    */
    for (unsigned int pack_i = 0; pack_i < MR; pack_i++) {
        for (unsigned int pack_j = 0; pack_j < NC; pack_j++) {
            for (unsigned int pack_z = 0; pack_z < KC; pack_z++) {
                c_pack[pack_i * NC + pack_j] += a_pack[pack_i * KC + pack_z] * b_pack[pack_j + NC * pack_z];
            }
        }
    }
}


inline void opt2CPU_aux_simd(float* a_pack, float* b_pack, float* c_pack) {
    /*
      - a_pack should be in row major format
      - b_pack should be in column major
      - c_pack will be in row major?!
    */
    for (unsigned int pack_i = 0; pack_i < MR; pack_i++) {
        for (unsigned int pack_j = 0; pack_j < NC; pack_j++) {
            for (unsigned int pack_z = 0; pack_z < KC; pack_z++) {
                c_pack[pack_i * NC + pack_j] += a_pack[pack_i * KC + pack_z] * b_pack[pack_j + NC * pack_z];
            }
        }
    }
}


inline void opt2CPU_gepb(GemmRun* run, unsigned int p, unsigned int j, float* a_pack, float* b_pack, float* c_pack) {
    /*
      This should
      - pack B
      - iterate over A and C
    */
    opt2CPU_packB(run, p, j, b_pack);
    for (unsigned int i = 0; i < run->m; i += MR) {
        opt2CPU_aux(a_pack + (i * KC), b_pack, c_pack);
        opt2CPU_unpackC(run, j, i, c_pack);
    }
}


inline void opt2CPU_gepp(GemmRun* run, unsigned int p, float* a_pack, float* b_pack, float* c_pack) {
    /*
      This should
      - pack A: A is in row major format,
      - iterate over B
    */
    opt2CPU_packA(run, p, a_pack);
    for (unsigned int j = 0; j < run->n; j += NC) {
        opt2CPU_gepb(run, p, j, a_pack, b_pack, c_pack);
    }
}


void opt2CPU_gemm_execute(GemmRun* run) {
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
        opt2CPU_gepp(run, p, a_pack, b_pack, c_pack);
    }

    free(a_pack);
    free(b_pack);
    free(c_pack);
}
