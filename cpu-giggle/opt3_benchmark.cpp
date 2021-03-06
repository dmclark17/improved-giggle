#include <cstdlib>

#include <immintrin.h>

#include "data_manager.h"


#define NC 64
#define KC 512
#define MR 16
#define NR 16


inline void opt3CPU_packA(GemmRun<float>* run, unsigned int p, float* a_pack) {
    /*
      utility function for packing A
      - Packs the A matrix into a panel of size m by k_c
      - Within the packed array, the data should be in row major format
    */
    #ifdef __AVX512F__
    __m512 src;
    for (unsigned int i = 0; i < run->m; i++) {
        for (unsigned int j = 0; j < KC; j += 16) {
            src = _mm512_load_ps(run->a + (p + run->lda * i + j));
            _mm512_store_ps(a_pack + (i * KC + j), src);
        }
    }
    #else
    #ifdef __AVX__
    __m256 src;
    for (unsigned int i = 0; i < run->m; i++) {
        for (unsigned int j = 0; j < KC; j += 8) {
            src = _mm256_load_ps(run->a + (p + run->lda * i + j));
            _mm256_store_ps(a_pack + (i * KC + j), src);
        }
    }
    #endif
    #endif
}


inline void opt3CPU_packB(GemmRun<float>* run, unsigned int p, unsigned int j, float* b_pack) {
    /*
      utility function for packing B
      - Packs B into a block of size n_c and k_c
      - Within the packed array, the data should be in column major format
      maybe this should be some sort of block form, but we can start with pure
      column major format for now?
    */
    unsigned int offset = run->ldb * p + j;
    #ifdef __AVX512F__
    __m512 src;
    for (unsigned int pack_j = 0; pack_j < NC; pack_j += NR) {
        for (unsigned int pack_i = 0; pack_i < KC; pack_i++) {
            for (unsigned int pack_n = 0; pack_n < NR; pack_n += 16) {
                src = _mm512_load_ps(run->b + offset + pack_i * run->ldb + pack_j + pack_n);
                _mm512_store_ps(b_pack + (pack_j * KC) + (pack_i * NR) + pack_n, src);
            }

        }
    }
    #else
    #ifdef __AVX__
    __m256 src;
    for (unsigned int pack_j = 0; pack_j < NC; pack_j += NR) {
        for (unsigned int pack_i = 0; pack_i < KC; pack_i++) {
            for (unsigned int pack_n = 0; pack_n < NR; pack_n += 8) {
                src = _mm256_load_ps(run->b + offset + pack_i * run->ldb + pack_j + pack_n);
                _mm256_store_ps(b_pack + (pack_j * KC) + (pack_i * NR) + pack_n, src);
            }

        }
    }
    #endif
    #endif
}


inline void opt3CPU_unpackC(GemmRun<float>* run, unsigned int j, unsigned int i, float* c_pack) {
    /*
      utility function for unpacking C
      - Unpacks a m_r by n_c submatrix into C
    */
    unsigned int offset = i * run->ldc + j;
    #ifdef __AVX512F__
    __m512 m_cpack, m_c, m_sum, zero_vec;
    zero_vec = _mm512_setzero_ps();
    for (unsigned int pack_i = 0; pack_i < MR; pack_i++) {
        for (unsigned int pack_j = 0; pack_j < NC; pack_j += 16) {
            m_cpack = _mm512_load_ps(c_pack + pack_i * NC + pack_j);
            m_c = _mm512_load_ps(run->c + offset + pack_i * run->ldc + pack_j);

            m_sum = _mm512_add_ps(m_cpack, m_c);

            _mm512_store_ps(run->c + offset + pack_i * run->ldc + pack_j, m_sum);
            _mm512_store_ps(c_pack + pack_i * NC + pack_j, zero_vec);
        }
    }
    #else
    #ifdef __AVX__
    __m256 m_cpack, m_c;
    for (unsigned int pack_i = 0; pack_i < MR; pack_i++) {
        for (unsigned int pack_j = 0; pack_j < NC; pack_j += 8) {
            m_cpack = _mm256_load_ps(c_pack + pack_i * NC + pack_j);
            m_c = _mm256_load_ps(run->c + offset + pack_i * run->ldc + pack_j);
            m_c = _mm256_add_ps(m_cpack, m_c);
            _mm256_store_ps(run->c + offset + pack_i * run->ldc + pack_j, m_c);
        }
    }
    #endif
    #endif
}


inline void opt3CPU_aux_simd(float* a_pack, float* b_pack, float* c_pack) {
    /*
      - a_pack should be in row major format
      - b_pack should be in column major
      - c_pack will be in row major?!
    */
    #ifdef __AVX512F__
    __m512 a, b, c;
    unsigned int pack_n, pack_i, pack_j, pack_z;
    for (pack_n = 0; pack_n < NC; pack_n += NR) {
        for (pack_j = 0; pack_j < NR; pack_j += 16) {
            for (pack_z = 0; pack_z < KC; pack_z++) {
                b = _mm512_load_ps(b_pack + (pack_n * KC) + (NR * pack_z) + pack_j);
                for (pack_i = 0; pack_i < MR; pack_i++) {
                    c = _mm512_load_ps(c_pack + pack_n + (pack_i * NC) + pack_j);
                    a = _mm512_broadcast_f32x4(_mm_broadcast_ss(a_pack + (pack_i * KC) + pack_z));
                    c = _mm512_fmadd_ps(a, b, c);
                    _mm512_store_ps(c_pack + pack_n + pack_i * NC + pack_j, c);
                }
            }
        }
    }
    #else
    #ifdef __AVX__
    __m256 a0, a1, a2;
    __m256 b0, b1;
    __m256 c0, c1, c2, c3, c4, c5;
    unsigned int pack_n, pack_i, pack_j, pack_z;

    const unsigned int mr_3 = (MR / 3) * 3;
    for (pack_n = 0; pack_n < NC; pack_n += NR) {
        for (pack_j = 0; pack_j < NR; pack_j += 8 * 2) {
            for (pack_i = 0; pack_i < mr_3; pack_i += 3) {
                c0 =  _mm256_setzero_ps();
                c1 =  _mm256_setzero_ps();
                c2 =  _mm256_setzero_ps();
                c3 =  _mm256_setzero_ps();
                c4 =  _mm256_setzero_ps();
                c5 =  _mm256_setzero_ps();

                for (pack_z = 0; pack_z < KC; pack_z++) {
                    b0 = _mm256_load_ps(b_pack + (pack_n * KC) + (NR * pack_z) + (pack_j+0));
                    b1 = _mm256_load_ps(b_pack + (pack_n * KC) + (NR * pack_z) + (pack_j+8));


                    a0 = _mm256_broadcast_ss(a_pack + ((pack_i+0) * KC) + pack_z);
                    a1 = _mm256_broadcast_ss(a_pack + ((pack_i+1) * KC) + pack_z);
                    a2 = _mm256_broadcast_ss(a_pack + ((pack_i+2) * KC) + pack_z);


                    c0 = _mm256_fmadd_ps(a0, b0, c0);
                    c1 = _mm256_fmadd_ps(a1, b0, c1);
                    c2 = _mm256_fmadd_ps(a0, b1, c2);
                    c3 = _mm256_fmadd_ps(a1, b1, c3);

                    c4 = _mm256_fmadd_ps(a2, b0, c4);
                    c5 = _mm256_fmadd_ps(a2, b1, c5);
                }
                _mm256_store_ps(c_pack + pack_n + (pack_i+0) * NC + (pack_j+0), c0);
                _mm256_store_ps(c_pack + pack_n + (pack_i+1) * NC + (pack_j+0), c1);
                _mm256_store_ps(c_pack + pack_n + (pack_i+0) * NC + (pack_j+8), c2);
                _mm256_store_ps(c_pack + pack_n + (pack_i+1) * NC + (pack_j+8), c3);

                _mm256_store_ps(c_pack + pack_n + (pack_i+2) * NC + (pack_j+0), c4);
                _mm256_store_ps(c_pack + pack_n + (pack_i+2) * NC + (pack_j+8), c5);
            }

            for ( ; pack_i < MR; pack_i++) {
                c0 =  _mm256_setzero_ps();
                c2 =  _mm256_setzero_ps();

                for (pack_z = 0; pack_z < KC; pack_z++) {
                    b0 = _mm256_load_ps(b_pack + (pack_n * KC) + (NR * pack_z) + (pack_j+0));
                    b1 = _mm256_load_ps(b_pack + (pack_n * KC) + (NR * pack_z) + (pack_j+8));

                    a0 = _mm256_broadcast_ss(a_pack + ((pack_i+0) * KC) + pack_z);

                    c0 = _mm256_fmadd_ps(a0, b0, c0);
                    c2 = _mm256_fmadd_ps(a0, b1, c2);
                }
                _mm256_store_ps(c_pack + pack_n + (pack_i+0) * NC + (pack_j+0), c0);
                _mm256_store_ps(c_pack + pack_n + (pack_i+0) * NC + (pack_j+8), c2);
            }
        }
    }
    #endif
    #endif
}


inline void opt3CPU_gepb(GemmRun<float>* run, unsigned int p, unsigned int j, float* a_pack, float* b_pack, float* c_pack) {
    /*
      This should
      - pack B
      - iterate over A and C
    */
    opt3CPU_packB(run, p, j, b_pack);
    unsigned int i;
    for (i = 0; i < run->m; i += MR) {
        opt3CPU_aux_simd(a_pack + (i * KC), b_pack, c_pack);
        opt3CPU_unpackC(run, j, i, c_pack);
    }
}


inline void opt3CPU_gepp(GemmRun<float>* run, unsigned int p, float* a_pack, float* b_pack, float* c_pack) {
    /*
      This should
      - pack A: A is in row major format,
      - iterate over B
    */
    opt3CPU_packA(run, p, a_pack);
    for (unsigned int j = 0; j < run->n; j += NC) {
        opt3CPU_gepb(run, p, j, a_pack, b_pack, c_pack);
    }
}


template <typename T>
void opt3CPU_gemm_execute(GemmRun<T>* run) {
    /*
      This should call gepp iteration over panels. Panel A_p and B_p will
      make a contribution to all of C?
    */
    float* a_pack = (float *)aligned_alloc( 64, KC * run->m * sizeof(float) );
    float* b_pack = (float *)aligned_alloc( 64, KC * NC * sizeof(float) );
    float* c_pack = (float *)aligned_alloc( 64, MR * NC * sizeof(float) );
    #ifdef __AVX512F__
    __m512 zero_vec = _mm512_setzero_ps();
    for (unsigned int i = 0; i < MR * NC; i += 16) {
        _mm512_store_ps(c_pack + i, zero_vec);
    }
    #else
    #ifdef __AVX__
    __m256 zero_vec = _mm256_setzero_ps();
    for (unsigned int i = 0; i < MR * NC; i += 8) {
        _mm256_store_ps(c_pack + i, zero_vec);
    }
    #endif
    #endif

    for (unsigned int p = 0; p < run->k; p += KC) {
        opt3CPU_gepp(run, p, a_pack, b_pack, c_pack);
    }

    free(a_pack);
    free(b_pack);
    free(c_pack);
}

template void opt3CPU_gemm_execute<float>(GemmRun<float>*);
