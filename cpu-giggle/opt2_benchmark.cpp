#include <cstdlib>

#include <immintrin.h>

#include "data_manager.h"


#define NC 64
#define KC 1024
#define MR 64
#define NR 1


template <typename T>
inline void opt2CPU_packA(GemmRun<T>* run, unsigned int p, T* a_pack);
/*
  utility function for packing A
  - Packs the A matrix into a panel of size m by k_c
  - Within the packed array, the data should be in row major format
*/

template <>
inline void opt2CPU_packA(GemmRun<float>* run, unsigned int p, float* a_pack) {
    /*
    * floats
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

template <>
inline void opt2CPU_packA(GemmRun<double>* run, unsigned int p, double* a_pack) {
    /*
    * doubles
    */
    #ifdef __AVX__
    __m256d src;
    for (unsigned int i = 0; i < run->m; i++) {
        for (unsigned int j = 0; j < KC; j += 4) {
            src = _mm256_load_pd(run->a + (p + run->lda * i + j));
            _mm256_store_pd(a_pack + (i * KC + j), src);
        }
    }
    #endif
}



template <typename T>
inline void opt2CPU_packB(GemmRun<T>* run, unsigned int p, unsigned int j, T* b_pack);
/*
  utility function for packing B
  - Packs B into a block of size n_c and k_c
  - Within the packed array, the data should be in column major format
  maybe this should be some sort of block form, but we can start with pure
  column major format for now?
*/

template <>
inline void opt2CPU_packB(GemmRun<float>* run, unsigned int p, unsigned int j, float* b_pack) {
    /*
    * Floats
    */
    unsigned int offset = run->ldb * p + j;
    #ifdef __AVX512F__
    __m512 src;
    for (unsigned int pack_i = 0; pack_i < KC; pack_i++) {
        for (unsigned int pack_j = 0; pack_j < NC; pack_j += 16) {
            src = _mm512_load_ps(run->b + offset + pack_i * run->ldb + pack_j);
            _mm512_store_ps(b_pack + pack_i * NC + pack_j, src);
        }
    }
    #else
    #ifdef __AVX__
    __m256 src;
    for (unsigned int pack_i = 0; pack_i < KC; pack_i++) {
        for (unsigned int pack_j = 0; pack_j < NC; pack_j += 8) {
            src = _mm256_load_ps(run->b + offset + pack_i * run->ldb + pack_j);
            _mm256_store_ps(b_pack + pack_i * NC + pack_j, src);
        }
    }
    #endif
    #endif
}

template <>
inline void opt2CPU_packB(GemmRun<double>* run, unsigned int p, unsigned int j, double* b_pack) {
    /*
    * Doubles
    */
    unsigned int offset = run->ldb * p + j;
    #ifdef __AVX__
    __m256d src;
    for (unsigned int pack_i = 0; pack_i < KC; pack_i++) {
        for (unsigned int pack_j = 0; pack_j < NC; pack_j += 4) {
            src = _mm256_load_pd(run->b + offset + pack_i * run->ldb + pack_j);
            _mm256_store_pd(b_pack + pack_i * NC + pack_j, src);
        }
    }
    #endif
}



template <typename T>
inline void opt2CPU_unpackC(GemmRun<T>* run, unsigned int j, unsigned int i, T* c_pack);
/*
  utility function for unpacking C
  - Unpacks a m_r by n_c submatrix into C
*/

template <>
inline void opt2CPU_unpackC(GemmRun<float>* run, unsigned int j, unsigned int i, float* c_pack) {
    /*
    * floats
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
    #endif
    #endif
}

template <>
inline void opt2CPU_unpackC(GemmRun<double>* run, unsigned int j, unsigned int i, double* c_pack) {
    /*
    * double
    */
    unsigned int offset = i * run->ldc + j;
    #ifdef __AVX__
    __m256d m_cpack, m_c, m_sum, zero_vec;
    zero_vec = _mm256_setzero_pd();
    for (unsigned int pack_i = 0; pack_i < MR; pack_i++) {
        for (unsigned int pack_j = 0; pack_j < NC; pack_j += 4) {
            m_cpack = _mm256_load_pd(c_pack + pack_i * NC + pack_j);
            m_c = _mm256_load_pd(run->c + offset + pack_i * run->ldc + pack_j);

            m_sum = _mm256_add_pd(m_cpack, m_c);

            _mm256_store_pd(run->c + offset + pack_i * run->ldc + pack_j, m_sum);
            _mm256_store_pd(c_pack + pack_i * NC + pack_j, zero_vec);
        }
    }
    #endif
}



template <typename T>
inline void opt2CPU_aux_simd(T* a_pack, T* b_pack, T* c_pack);
/*
  - a_pack should be in row major format
  - b_pack should be in column major
  - c_pack will be in row major?!
*/

template <>
inline void opt2CPU_aux_simd(float* a_pack, float* b_pack, float* c_pack) {
    /*
    * float
    */
    #ifdef __AVX512F__
    __m512 a, b, c;
    for (unsigned int pack_z = 0; pack_z < KC; pack_z++) {
        for (unsigned int pack_i = 0; pack_i < MR; pack_i++) {
            // Load a and scatter
            a = _mm512_broadcast_f32x4(_mm_broadcast_ss(a_pack + pack_i * KC + pack_z));
            for (unsigned int pack_j = 0; pack_j < NC; pack_j += 16) {
                b = _mm512_load_ps(b_pack + NC * pack_z + pack_j);
                c = _mm512_load_ps(c_pack + pack_i * NC + pack_j);
                c = _mm512_fmadd_ps(a, b, c);
                _mm512_store_ps(c_pack + pack_i * NC + pack_j, c);
            }
        }
    }
    #else
    #ifdef __AVX__
    __m256 a, b, c;
    for (unsigned int pack_z = 0; pack_z < KC; pack_z++) {
        for (unsigned int pack_i = 0; pack_i < MR; pack_i++) {
            // Load a and scatter
            a = _mm256_broadcast_ss(a_pack + pack_i * KC + pack_z);
            for (unsigned int pack_j = 0; pack_j < NC; pack_j += 8) {
                b = _mm256_load_ps(b_pack + NC * pack_z + pack_j);
                c = _mm256_load_ps(c_pack + pack_i * NC + pack_j);
                c = _mm256_fmadd_ps(a, b, c);
                _mm256_store_ps(c_pack + pack_i * NC + pack_j, c);
            }
        }
    }
    #endif
    #endif
}

template <>
inline void opt2CPU_aux_simd(double* a_pack, double* b_pack, double* c_pack) {
    /*
    * double
    */
    #ifdef __AVX__
    __m256d a, b, c;
    for (unsigned int pack_z = 0; pack_z < KC; pack_z++) {
        for (unsigned int pack_i = 0; pack_i < MR; pack_i++) {
            // Load a and scatter
            a = _mm256_broadcast_sd(a_pack + pack_i * KC + pack_z);
            for (unsigned int pack_j = 0; pack_j < NC; pack_j += 4) {
                b = _mm256_load_pd(b_pack + NC * pack_z + pack_j);
                c = _mm256_load_pd(c_pack + pack_i * NC + pack_j);
                c = _mm256_fmadd_pd(a, b, c);
                _mm256_store_pd(c_pack + pack_i * NC + pack_j, c);
            }
        }
    }
    #endif
}



template <typename T>
inline void opt2CPU_gepb(GemmRun<T>* run, unsigned int p, unsigned int j, T* a_pack, T* b_pack, T* c_pack) {
    /*
      This should
      - pack B
      - iterate over A and C
    */
    opt2CPU_packB(run, p, j, b_pack);
    for (unsigned int i = 0; i < run->m; i += MR) {
        opt2CPU_aux_simd(a_pack + (i * KC), b_pack, c_pack);
        opt2CPU_unpackC(run, j, i, c_pack);
    }
}



template <typename T>
inline void opt2CPU_gepp(GemmRun<T>* run, unsigned int p, T* a_pack, T* b_pack, T* c_pack) {
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



template <typename T>
void opt2CPU_gemm_execute(GemmRun<T>* run);

template <>
void opt2CPU_gemm_execute(GemmRun<float>* run) {
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
        opt2CPU_gepp(run, p, a_pack, b_pack, c_pack);
    }

    free(a_pack);
    free(b_pack);
    free(c_pack);
}

template <>
void opt2CPU_gemm_execute(GemmRun<double>* run) {
    /*
      This should call gepp iteration over panels. Panel A_p and B_p will
      make a contribution to all of C?
    */
    double* a_pack = (double *)aligned_alloc( 64, KC * run->m * sizeof(double) );
    double* b_pack = (double *)aligned_alloc( 64, KC * NC * sizeof(double) );
    double* c_pack = (double *)aligned_alloc( 64, MR * NC * sizeof(double) );

    #ifdef __AVX__
    __m256d zero_vec = _mm256_setzero_pd();
    for (unsigned int i = 0; i < MR * NC; i += 4) {
        _mm256_store_pd(c_pack + i, zero_vec);
    }
    #endif

    for (unsigned int p = 0; p < run->k; p += KC) {
        opt2CPU_gepp(run, p, a_pack, b_pack, c_pack);
    }

    free(a_pack);
    free(b_pack);
    free(c_pack);
}
