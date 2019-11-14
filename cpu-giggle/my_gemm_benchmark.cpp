#include <stdio.h>
#include <stdlib.h>

#include "data_manager.h"

#include "cpu_benchmark.h"


#define NC 256
#define KC 256
#define MR 128
#define NR 1


void naiveCPU_gemm_execute(GemmRun* run) {
    float dot_prod = 0;
    for (unsigned int i = 0; i < run->m; i++) {
        for (unsigned int j = 0; j < run->n; j++) {
            for (unsigned int z = 0; z < run->k; z++) {
                dot_prod += run->a[i * run->lda + z] *
                            run->b[z * run->ldb + j];
            }
            run->c[i* run->ldc + j] *= run->beta;
            run->c[i* run->ldc + j] += run->alpha * dot_prod;
            dot_prod = 0;
        }
    }
}


inline void opt1CPU_packA(GemmRun* run, int p, float* a_pack) {
    /*
      utility function for packing A
      - Packs the A matrix into a panel of size m by k_c
      - Within the packed array, the data should be in row major format
    */

    for (int i = 0; i < run->m; i++) {
        for (int j = 0; j < KC; j++) {
            a_pack[i * KC + j] = run->a[p + run->lda * i + j];
        }
    }
}


inline void opt1CPU_packB(GemmRun* run, int p, int j, float* b_pack) {
    /*
      utility function for packing B
      - Packs B into a block of size n_c and k_c
      - Within the packed array, the data should be in column major format
      maybe this should be some sort of block form, but we can start with pure
      column major format for now?
    */
    unsigned int offset = run->ldb * p + j;
    for (int pack_i = 0; pack_i < KC; pack_i++) {
        for (int pack_j = 0; pack_j < NC; pack_j++) {
            b_pack[pack_j * KC + pack_i] = run->b[offset + pack_i * run->ldb + pack_j];
        }
    }
}


inline void opt1CPU_unpackC(GemmRun* run, int j, int i, float* c_pack) {
    /*
      utility function for unpacking C
      - Unpacks a m_r by n_c submatrix into C
    */
    unsigned int offset = i * run->ldc + j;
    for (int pack_i = 0; pack_i < MR; pack_i++) {
        for (int pack_j = 0; pack_j < NC; pack_j++) {
            run->c[offset + pack_i * run->ldc + pack_j] += c_pack[pack_i * NC + pack_j];
        }
    }
}


inline void opt1CPU_aux(int i, float* a_pack, float* b_pack, float* c_pack) {
    /*
      - a_pack should be in row major format
      - b_pack should be in column major
      - c_pack will be in row major?!
    */
    for (int pack_i = 0; pack_i < MR; pack_i++) {
        for (int pack_j = 0; pack_j < NC; pack_j++) {
            c_pack[pack_i * NC + pack_j] = 0;

            for (int pack_z = 0; pack_z < KC; pack_z++) {
                c_pack[pack_i * NC + pack_j] += (a_pack[pack_i * KC + pack_z] * b_pack[pack_j * KC + pack_z]);
            }
        }
    }
}


inline void opt1CPU_gepb(GemmRun* run, int p, int j, float* a_pack, float* b_pack, float* c_pack) {
    /*
      This should
      - pack B
      - iterate over A and C
    */
    opt1CPU_packB(run, p, j, b_pack);
    for (int i = 0; i < run->m; i += MR) {
        opt1CPU_aux(0, a_pack + (i * KC), b_pack, c_pack);
        opt1CPU_unpackC(run, j, i, c_pack);
    }
}


inline void opt1CPU_gepp(GemmRun* run, int p, float* a_pack, float* b_pack, float* c_pack) {
    /*
      This should
      - pack A: A is in row major format,
      - iterate over B
    */
    opt1CPU_packA(run, p, a_pack);
    for (int j = 0; j < run->n; j += NC) {
        opt1CPU_gepb(run, p, j, a_pack, b_pack, c_pack);
    }
}


void opt1CPU_gemm_execute(GemmRun* run) {
    /*
      This should call gepp iteration over panels. Panel A_p and B_p will
      make a contribution to all of C?
    */
    float* a_pack = (float *)calloc( KC * run->m, sizeof(float) );
    float* b_pack = (float *)calloc( KC * NC, sizeof(float) );
    float* c_pack = (float *)calloc( MR * NC, sizeof(float) );

    for (int p = 0; p < run->k; p += KC) {
        opt1CPU_gepp(run, p, a_pack, b_pack, c_pack);
    }

    free(a_pack);
    free(b_pack);
    free(c_pack);
}
