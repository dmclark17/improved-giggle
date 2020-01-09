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
#define NR 12
#define NC_CLEAN (NC / NR) * NR
#define NR_SMALL 4

inline void opt5CPU_packA(GemmRun<float>* run, unsigned int p, float* a_pack) {
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


inline void opt5CPU_packB(GemmRun<float>* run, unsigned int p, unsigned int j, float* b_pack) {
    /*
      utility function for packing B
      - Packs B into a block of size n_c and k_c
      - Within the packed array, the data should be in column major format
      maybe this should be some sort of block form, but we can start with pure
      column major format for now?
    */
    unsigned int offset = run->ldb * p + j;
    unsigned int block_index, global_index;
    unsigned int pack_j;

    for ( pack_j = 0; pack_j < NC_CLEAN; pack_j += NR) {
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

    // Clean up
    for ( ; pack_j < NC; pack_j += NR_SMALL) {
        for (unsigned int pack_i = 0; pack_i < KC; pack_i += 2) {

            for (unsigned int other_j = 0; other_j < NR_SMALL; other_j += 4) {

                // These 8 values will be in one register
                for (unsigned int inner_i = 0; inner_i < 2; inner_i++) {
                    for (unsigned int inner_j = 0; inner_j < 4; inner_j++) {
                        block_index = (pack_j * KC) + (pack_i * NR_SMALL) + (inner_i * 4) + other_j * 2 + inner_j;
                        global_index = offset + (pack_i + inner_i) * run->ldb + (pack_j + other_j + inner_j);
                        b_pack[block_index] = run->b[global_index];
                    }
                }
            }
        }
    }
}


inline void opt5CPU_unpackC(GemmRun<float>* run, unsigned int j, unsigned int i, float* c_pack) {
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

void opt5CPU_aux_asm(float* a_pack, float* b_pack, float* c_pack) {
    __asm__(
        "movq                %1, %%rdi     \n"
        "addq                $64, %%rdi    \n"
        "xor                %%r8, %%r8     \n"
        "1:                                \n"
        "    vxorps              %%xmm1, %%xmm1, %%xmm1    \n"
        "	 movq                %0, %%r9                  \n"
        "    movq                %%rdi, %%rax              \n"
        "    xor                 %%rcx, %%rcx              \n"
        "    vxorps              %%xmm2, %%xmm2, %%xmm2    \n"
        "    vxorps              %%xmm3, %%xmm3, %%xmm3    \n"
        "    vxorps              %%xmm4, %%xmm4, %%xmm4    \n"
        "    vxorps              %%xmm5, %%xmm5, %%xmm5    \n"
        "    vxorps              %%xmm6, %%xmm6, %%xmm6    \n"
        "    vxorps              %%xmm7, %%xmm7, %%xmm7    \n"
        "    vxorps              %%xmm8, %%xmm8, %%xmm8    \n"
        "    vxorps              %%xmm9, %%xmm9, %%xmm9    \n"
        "    vxorps              %%xmm10, %%xmm10, %%xmm10 \n"
        "    vxorps              %%xmm11, %%xmm11, %%xmm11 \n"
        "    vxorps              %%xmm13, %%xmm13, %%xmm13 \n"
        "    2:                                     \n"
        "        vmovaps             (%%r9),     %%ymm14  \n"
        "        vmovaps             -64(%%rax), %%ymm15  \n"
        "        vmovaps             -32(%%rax), %%ymm0   \n"
        "        vmovaps             (%%rax),    %%ymm12  \n"
        "        vfmadd231ps         %%ymm15, %%ymm14, %%ymm6  \n"
        "        vfmadd231ps         %%ymm0,  %%ymm14, %%ymm3  \n"
        "        vfmadd231ps         %%ymm12, %%ymm14, %%ymm1  \n"
        "        vpermilps           $57, %%ymm14, %%ymm14     \n"
        "        vfmadd231ps         %%ymm15, %%ymm14, %%ymm9  \n"
        "        vfmadd231ps         %%ymm0,  %%ymm14, %%ymm2  \n"
        "        vfmadd231ps         %%ymm12, %%ymm14, %%ymm13 \n"
        "        vpermilps           $57, %%ymm14, %%ymm14     \n"
        "        vfmadd231ps         %%ymm15, %%ymm14, %%ymm8  \n"
        "        vfmadd231ps         %%ymm0,  %%ymm14, %%ymm5  \n"
        "        vfmadd231ps         %%ymm12, %%ymm14, %%ymm11 \n"
        "        vpermilps           $57, %%ymm14, %%ymm14     \n"
        "        vfmadd231ps         %%ymm15, %%ymm14, %%ymm7  \n"
        "        vfmadd231ps         %%ymm0,  %%ymm14, %%ymm4  \n"
        "        vfmadd231ps         %%ymm12, %%ymm14, %%ymm10 \n"
        "        addq                $2, %%rcx                 \n"
        "        addq                $96, %%rax                \n"
        "        addq                $32, %%r9                 \n"
        "        cmpq                $512, %%rcx               \n"
        "        jb                  2b                        \n"
        "    vextractf128        $1, %%ymm6, %%xmm0       \n"
        "    vextractf128        $1, %%ymm7, %%xmm12      \n"
        "    vextractf128        $1, %%ymm8, %%xmm14      \n"
        "    vextractf128        $1, %%ymm9, %%xmm15      \n"
        "    vaddps              %%xmm6, %%xmm0,  %%xmm6   \n"
        "    vaddps              %%xmm7, %%xmm12, %%xmm7   \n"
        "    vaddps              %%xmm8, %%xmm14, %%xmm8   \n"
        "    vaddps              %%xmm9, %%xmm15, %%xmm9   \n"
        "    movq                %%r8, %%rax              \n"
        "    shlq                $4,  %%rax              \n"
        "    vmovaps             %%xmm6,    (%2,%%rax)  \n"
        "    vmovaps             %%xmm7,  16(%2,%%rax)  \n"
        "    vmovaps             %%xmm8,  32(%2,%%rax)  \n"
        "    vmovaps             %%xmm9,  48(%2,%%rax)  \n"
        "    vextractf128        $1, %%ymm3, %%xmm0       \n"
        "    vextractf128        $1, %%ymm4, %%xmm12      \n"
        "    vextractf128        $1, %%ymm5, %%xmm14      \n"
        "    vextractf128        $1, %%ymm2, %%xmm15      \n"
        "    vaddps              %%xmm3, %%xmm0,  %%xmm3   \n"
        "    vaddps              %%xmm4, %%xmm12, %%xmm4   \n"
        "    vaddps              %%xmm5, %%xmm14, %%xmm5   \n"
        "    vaddps              %%xmm2, %%xmm15, %%xmm2   \n"
        "    vmovaps             %%xmm3,  64(%2,%%rax)  \n"
        "    vmovaps             %%xmm4,  80(%2,%%rax)  \n"
        "    vmovaps             %%xmm5,  96(%2,%%rax)  \n"
        "    vmovaps             %%xmm2,  112(%2,%%rax) \n"
        "    vextractf128        $1, %%ymm1,  %%xmm0      \n"
        "    vextractf128        $1, %%ymm10, %%xmm12     \n"
        "    vextractf128        $1, %%ymm11, %%xmm14     \n"
        "    vextractf128        $1, %%ymm13, %%xmm15     \n"
        "    vaddps              %%xmm1,  %%xmm0,  %%xmm1  \n"
        "    vaddps              %%xmm10, %%xmm12, %%xmm10 \n"
        "    vaddps              %%xmm11, %%xmm14, %%xmm11 \n"
        "    vaddps              %%xmm13, %%xmm15, %%xmm13 \n"
        "    vmovaps             %%xmm1,  128(%2,%%rax) \n"
        "    vmovaps             %%xmm10, 144(%2,%%rax) \n"
        "    vmovaps             %%xmm11, 160(%2,%%rax) \n"
        "    vmovaps             %%xmm13, 176(%2,%%rax) \n"
        "    addq                $12,    %%r8            \n"
        "    addq                $24576, %%rdi              \n"
        "    cmpq                $60, %%r8               \n"
        "    jb                  1b                     \n"
        "vzeroupper                                 \n"
        :: "r" (a_pack), "r" (b_pack), "r" (c_pack)
        : "rdi", "r8", "r9", "rax", "rcx", "memory", "ymm0", "ymm1", "ymm2", "ymm3", "ymm4", "ymm5", "ymm6", "ymm7",
          "ymm8", "ymm9", "ymm10", "ymm11", "ymm12", "ymm13","ymm14", "ymm15"
    );
}


void opt5CPU_aux_asm_unroll(float* a_pack, float* b_pack, float* c_pack) {
    __asm__(
    "movq                %1, %%rdi     \n"
    "addq                $64, %%rdi    \n"
    "xor                %%r8, %%r8     \n"
    "1:                                \n"
    "    prefetcht0          -64(%%rax)                \n"
    "    prefetcht0          (%%r9)                    \n"
    "    vxorps              %%xmm6, %%xmm6, %%xmm6    \n"
    "	 movq                %0, %%r9                  \n"
    "    movq                %%rdi, %%rax              \n"
    "    xor                 %%rcx, %%rcx              \n"
    "    prefetcht0          -32(%%rax)                \n"
    "    vxorps              %%xmm2, %%xmm2, %%xmm2    \n"
    "    vxorps              %%xmm3, %%xmm3, %%xmm3    \n"
    "    vxorps              %%xmm4, %%xmm4, %%xmm4    \n"
    "    vxorps              %%xmm5, %%xmm5, %%xmm5    \n"
    "    vxorps              %%xmm1, %%xmm1, %%xmm1    \n"
    "    prefetcht0          (%%rax)                \n"
    "    vxorps              %%xmm7, %%xmm7, %%xmm7    \n"
    "    vxorps              %%xmm8, %%xmm8, %%xmm8    \n"
    "    vxorps              %%xmm9, %%xmm9, %%xmm9    \n"
    "    vxorps              %%xmm10, %%xmm10, %%xmm10 \n"
    "    vxorps              %%xmm11, %%xmm11, %%xmm11 \n"
    "    vxorps              %%xmm13, %%xmm13, %%xmm13 \n"
    "    2:                                     \n"
    "        vmovaps             (%%r9),     %%ymm14  \n"
    "        vmovaps             -64(%%rax), %%ymm15  \n"
    "        vfmadd231ps         %%ymm15, %%ymm14, %%ymm6  \n"
    "        vmovaps             -32(%%rax), %%ymm0   \n"
    "        vmovaps             (%%rax),    %%ymm12  \n"
    "        vfmadd231ps         %%ymm0,  %%ymm14, %%ymm3  \n"
    "        vfmadd231ps         %%ymm12, %%ymm14, %%ymm1  \n"
    "        vpermilps           $57, %%ymm14, %%ymm14     \n"
    "        vfmadd231ps         %%ymm15, %%ymm14, %%ymm9  \n"
    "        vfmadd231ps         %%ymm0,  %%ymm14, %%ymm2  \n"
    "        vfmadd231ps         %%ymm12, %%ymm14, %%ymm13 \n"
    "        vpermilps           $57, %%ymm14, %%ymm14     \n"
    "        vfmadd231ps         %%ymm15, %%ymm14, %%ymm8  \n"
    "        vfmadd231ps         %%ymm0,  %%ymm14, %%ymm5  \n"
    "        vfmadd231ps         %%ymm12, %%ymm14, %%ymm11 \n"
    "        vpermilps           $57, %%ymm14, %%ymm14     \n"
    "        vfmadd231ps         %%ymm15, %%ymm14, %%ymm7  \n"
    "        vmovaps             32(%%rax), %%ymm15  \n"
    "        vfmadd231ps         %%ymm0,  %%ymm14, %%ymm4  \n"
    "        vfmadd231ps         %%ymm12, %%ymm14, %%ymm10 \n"
    "        vmovaps             32(%%r9),     %%ymm14  \n"
    "        vmovaps             64(%%rax), %%ymm0   \n"
    "        vmovaps             96(%%rax),    %%ymm12  \n"
    "        vfmadd231ps         %%ymm15, %%ymm14, %%ymm6  \n"
    "        vfmadd231ps         %%ymm0,  %%ymm14, %%ymm3  \n"
    "        vfmadd231ps         %%ymm12, %%ymm14, %%ymm1  \n"
    "        vpermilps           $57, %%ymm14, %%ymm14     \n"
    "        vfmadd231ps         %%ymm15, %%ymm14, %%ymm9  \n"
    "        vfmadd231ps         %%ymm0,  %%ymm14, %%ymm2  \n"
    "        vfmadd231ps         %%ymm12, %%ymm14, %%ymm13 \n"
    "        vpermilps           $57, %%ymm14, %%ymm14     \n"
    "        vfmadd231ps         %%ymm15, %%ymm14, %%ymm8  \n"
    "        vfmadd231ps         %%ymm0,  %%ymm14, %%ymm5  \n"
    "        vfmadd231ps         %%ymm12, %%ymm14, %%ymm11 \n"
    "        vpermilps           $57, %%ymm14, %%ymm14     \n"
    "        vfmadd231ps         %%ymm15, %%ymm14, %%ymm7  \n"
    "        vfmadd231ps         %%ymm0,  %%ymm14, %%ymm4  \n"
    "        vfmadd231ps         %%ymm12, %%ymm14, %%ymm10 \n"
    "        vmovaps             64(%%r9),     %%ymm14  \n"
    "        vmovaps             128(%%rax), %%ymm15  \n"
    "        vfmadd231ps         %%ymm15, %%ymm14, %%ymm6  \n"
    "        vmovaps             160(%%rax), %%ymm0   \n"
    "        vmovaps             192(%%rax),    %%ymm12  \n"
    "        vfmadd231ps         %%ymm0,  %%ymm14, %%ymm3  \n"
    "        vfmadd231ps         %%ymm12, %%ymm14, %%ymm1  \n"
    "        vpermilps           $57, %%ymm14, %%ymm14     \n"
    "        vfmadd231ps         %%ymm15, %%ymm14, %%ymm9  \n"
    "        vfmadd231ps         %%ymm0,  %%ymm14, %%ymm2  \n"
    "        vfmadd231ps         %%ymm12, %%ymm14, %%ymm13 \n"
    "        vpermilps           $57, %%ymm14, %%ymm14     \n"
    "        vfmadd231ps         %%ymm15, %%ymm14, %%ymm8  \n"
    "        vfmadd231ps         %%ymm0,  %%ymm14, %%ymm5  \n"
    "        vfmadd231ps         %%ymm12, %%ymm14, %%ymm11 \n"
    "        vpermilps           $57, %%ymm14, %%ymm14     \n"
    "        vfmadd231ps         %%ymm15, %%ymm14, %%ymm7  \n"
    "        vmovaps             224(%%rax), %%ymm15  \n"
    "        vfmadd231ps         %%ymm0,  %%ymm14, %%ymm4  \n"
    "        vfmadd231ps         %%ymm12, %%ymm14, %%ymm10 \n"
    "        vmovaps             96(%%r9),     %%ymm14  \n"
    "        vmovaps             256(%%rax), %%ymm0   \n"
    "        vmovaps             288(%%rax),    %%ymm12  \n"
    "        vfmadd231ps         %%ymm15, %%ymm14, %%ymm6  \n"
    "        vfmadd231ps         %%ymm0,  %%ymm14, %%ymm3  \n"
    "        vfmadd231ps         %%ymm12, %%ymm14, %%ymm1  \n"
    "        vpermilps           $57, %%ymm14, %%ymm14     \n"
    "        vfmadd231ps         %%ymm15, %%ymm14, %%ymm9  \n"
    "        vfmadd231ps         %%ymm0,  %%ymm14, %%ymm2  \n"
    "        vfmadd231ps         %%ymm12, %%ymm14, %%ymm13 \n"
    "        vpermilps           $57, %%ymm14, %%ymm14     \n"
    "        vfmadd231ps         %%ymm15, %%ymm14, %%ymm8  \n"
    "        vfmadd231ps         %%ymm0,  %%ymm14, %%ymm5  \n"
    "        vfmadd231ps         %%ymm12, %%ymm14, %%ymm11 \n"
    "        vpermilps           $57, %%ymm14, %%ymm14     \n"
    "        vfmadd231ps         %%ymm15, %%ymm14, %%ymm7  \n"
    "        vfmadd231ps         %%ymm0,  %%ymm14, %%ymm4  \n"
    "        vfmadd231ps         %%ymm12, %%ymm14, %%ymm10 \n"
    "        addq                $8, %%rcx                 \n"
    "        addq                $384, %%rax                \n"
    "        addq                $128, %%r9                 \n"
    "        cmpq                $512, %%rcx               \n"
    "        jb                  2b                        \n"
    "    vextractf128        $1, %%ymm6, %%xmm0       \n"
    "    vextractf128        $1, %%ymm7, %%xmm12      \n"
    "    vextractf128        $1, %%ymm8, %%xmm14      \n"
    "    vextractf128        $1, %%ymm9, %%xmm15      \n"
    "    vaddps              %%xmm6, %%xmm0,  %%xmm6   \n"
    "    vaddps              %%xmm7, %%xmm12, %%xmm7   \n"
    "    vaddps              %%xmm8, %%xmm14, %%xmm8   \n"
    "    vaddps              %%xmm9, %%xmm15, %%xmm9   \n"
    "    movq                %%r8, %%rax              \n"
    "    shlq                $4,  %%rax              \n"
    "    vmovaps             %%xmm6,    (%2,%%rax)  \n"
    "    vmovaps             %%xmm7,  16(%2,%%rax)  \n"
    "    vmovaps             %%xmm8,  32(%2,%%rax)  \n"
    "    vmovaps             %%xmm9,  48(%2,%%rax)  \n"
    "    vextractf128        $1, %%ymm3, %%xmm0       \n"
    "    vextractf128        $1, %%ymm4, %%xmm12      \n"
    "    vextractf128        $1, %%ymm5, %%xmm14      \n"
    "    vextractf128        $1, %%ymm2, %%xmm15      \n"
    "    vaddps              %%xmm3, %%xmm0,  %%xmm3   \n"
    "    vaddps              %%xmm4, %%xmm12, %%xmm4   \n"
    "    vaddps              %%xmm5, %%xmm14, %%xmm5   \n"
    "    vaddps              %%xmm2, %%xmm15, %%xmm2   \n"
    "    vmovaps             %%xmm3,  64(%2,%%rax)  \n"
    "    vmovaps             %%xmm4,  80(%2,%%rax)  \n"
    "    vmovaps             %%xmm5,  96(%2,%%rax)  \n"
    "    vmovaps             %%xmm2,  112(%2,%%rax) \n"
    "    vextractf128        $1, %%ymm1,  %%xmm0      \n"
    "    vextractf128        $1, %%ymm10, %%xmm12     \n"
    "    vextractf128        $1, %%ymm11, %%xmm14     \n"
    "    vextractf128        $1, %%ymm13, %%xmm15     \n"
    "    vaddps              %%xmm1,  %%xmm0,  %%xmm1  \n"
    "    vaddps              %%xmm10, %%xmm12, %%xmm10 \n"
    "    vaddps              %%xmm11, %%xmm14, %%xmm11 \n"
    "    vaddps              %%xmm13, %%xmm15, %%xmm13 \n"
    "    vmovaps             %%xmm1,  128(%2,%%rax) \n"
    "    vmovaps             %%xmm10, 144(%2,%%rax) \n"
    "    vmovaps             %%xmm11, 160(%2,%%rax) \n"
    "    vmovaps             %%xmm13, 176(%2,%%rax) \n"
    "    addq                $12,    %%r8            \n"
    "    addq                $24576, %%rdi              \n"
    "    cmpq                $60, %%r8               \n"
    "    jb                  1b                     \n"
    "vzeroupper                                 \n"
    :: "r" (a_pack), "r" (b_pack), "r" (c_pack)
    : "rdi", "r8", "r9", "rax", "rcx", "memory", "ymm0", "ymm1", "ymm2", "ymm3", "ymm4", "ymm5", "ymm6", "ymm7",
    "ymm8", "ymm9", "ymm10", "ymm11", "ymm12", "ymm13","ymm14", "ymm15"
    );
}


__attribute__((noinline))
void opt5CPU_aux_simd(float* a_pack, float* b_pack, float* c_pack) {
    /*
      - a_pack should be in row major format
      - b_pack should be in column major
      - c_pack will be in row major?!
    */
    __m256 a0;
    __m256 b0, b1, b2;
    __m256 c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11;
    __m128 sum0, sum1, sum2, sum3, sum4, sum5, sum6, sum7;


    const __m256i perm1 = _mm256_setr_epi32(3, 0, 1, 2, 3, 0, 1, 2);

    unsigned int pack_n, pack_z;

    for (pack_n = 0; pack_n < NC_CLEAN; pack_n += NR) {

        c0 = _mm256_setzero_ps();
        c1 = _mm256_setzero_ps();
        c2 = _mm256_setzero_ps();
        c3 = _mm256_setzero_ps();

        c4 = _mm256_setzero_ps();
        c5 = _mm256_setzero_ps();
        c6 = _mm256_setzero_ps();
        c7 = _mm256_setzero_ps();

        c8 = _mm256_setzero_ps();
        c9 = _mm256_setzero_ps();
        c10 = _mm256_setzero_ps();
        c11 = _mm256_setzero_ps();

        #pragma clang loop unroll(disable)
        for (pack_z = 0; pack_z < KC; pack_z += 2) {
            a0 = _mm256_load_ps(a_pack + (pack_z) * 4 + (0));
            b0 = _mm256_load_ps(b_pack + (pack_n * KC) + (pack_z) * NR + (0));
            b1 = _mm256_load_ps(b_pack + (pack_n * KC) + (pack_z) * NR + (8));
            b2 = _mm256_load_ps(b_pack + (pack_n * KC) + (pack_z) * NR + (16));

            c0 = _mm256_fmadd_ps(a0, b0, c0);
            c4 = _mm256_fmadd_ps(a0, b1, c4);
            c8 = _mm256_fmadd_ps(a0, b2, c8);

            a0 = _mm256_permutevar_ps(a0, perm1);
            c1 = _mm256_fmadd_ps(a0, b0, c1);
            c5 = _mm256_fmadd_ps(a0, b1, c5);
            c9 = _mm256_fmadd_ps(a0, b2, c9);

            a0 = _mm256_permutevar_ps(a0, perm1);
            c2 = _mm256_fmadd_ps(a0, b0, c2);
            c6 = _mm256_fmadd_ps(a0, b1, c6);
            c10 = _mm256_fmadd_ps(a0, b2, c10);

            a0 = _mm256_permutevar_ps(a0, perm1);
            c3 = _mm256_fmadd_ps(a0, b0, c3);
            c7 = _mm256_fmadd_ps(a0, b1, c7);
            c11 = _mm256_fmadd_ps(a0, b2, c11);
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

        // Round three
        sum0 = _mm256_extractf128_ps(c8, 1);
        sum1 = _mm256_extractf128_ps(c9, 1);
        sum2 = _mm256_extractf128_ps(c10, 1);
        sum3 = _mm256_extractf128_ps(c11, 1);

        sum4 = _mm256_extractf128_ps(c8, 0);
        sum5 = _mm256_extractf128_ps(c9, 0);
        sum6 = _mm256_extractf128_ps(c10, 0);
        sum7 = _mm256_extractf128_ps(c11, 0);

        sum0 = _mm_add_ps(sum0, sum4);
        sum1 = _mm_add_ps(sum1, sum5);
        sum2 = _mm_add_ps(sum2, sum6);
        sum3 = _mm_add_ps(sum3, sum7);

        _mm_store_ps(c_pack + pack_n * MR + (8) * 4, sum0);
        _mm_store_ps(c_pack + pack_n * MR + (9) * 4, sum1);
        _mm_store_ps(c_pack + pack_n * MR + (10) * 4, sum2);
        _mm_store_ps(c_pack + pack_n * MR + (11) * 4, sum3);
    }


}

inline void opt5CPU_aux_simd_cleanup(float* a_pack, float* b_pack, float* c_pack) {
    __m256 a0;
    __m256 b0;
    __m256 c0, c1, c2, c3;
    __m128 sum0, sum1, sum2, sum3, sum4, sum5, sum6, sum7;


    const __m256i perm1 = _mm256_setr_epi32(3, 0, 1, 2, 3, 0, 1, 2);

    unsigned int pack_n, pack_z;

    for (pack_n = NC_CLEAN ; pack_n < NC; pack_n += NR_SMALL) {

        c0 =  _mm256_setzero_ps();
        c1 =  _mm256_setzero_ps();
        c2 =  _mm256_setzero_ps();
        c3 =  _mm256_setzero_ps();

        for (pack_z = 0; pack_z < KC; pack_z += 2) {
            a0 = _mm256_load_ps(a_pack + (pack_z) * 4 + (0));
            b0 = _mm256_load_ps(b_pack + (pack_n * KC) + (pack_z) * NR_SMALL + (0));

            c0 = _mm256_fmadd_ps(a0, b0, c0);

            a0 = _mm256_permutevar_ps(a0, perm1);
            c1 = _mm256_fmadd_ps(a0, b0, c1);

            a0 = _mm256_permutevar_ps(a0, perm1);
            c2 = _mm256_fmadd_ps(a0, b0, c2);

            a0 = _mm256_permutevar_ps(a0, perm1);
            c3 = _mm256_fmadd_ps(a0, b0, c3);
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
    }
}


inline void opt5CPU_gepb(GemmRun<float>* run, unsigned int p, unsigned int j, float* a_pack, float* b_pack, float* c_pack) {
    /*
      This should
      - pack B
      - iterate over A and C
    */
    opt5CPU_packB(run, p, j, b_pack);
    unsigned int i;
    for (i = 0; i < run->m; i += MR) {
        opt5CPU_aux_asm_unroll(a_pack + (i * KC), b_pack, c_pack);
        opt5CPU_aux_simd_cleanup(a_pack + (i * KC), b_pack, c_pack);
        opt5CPU_unpackC(run, j, i, c_pack);
    }
}


inline void opt5CPU_gepp(GemmRun<float>* run, unsigned int p, float* a_pack, float* b_pack, float* c_pack) {
    /*
      This should
      - pack A: A is in row major format,
      - iterate over B
    */
    opt5CPU_packA(run, p, a_pack);
    for (unsigned int j = 0; j < run->n; j += NC) {
        opt5CPU_gepb(run, p, j, a_pack, b_pack, c_pack);
    }
}


template <typename T>
void opt5CPU_gemm_execute(GemmRun<T>* run) {
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
        opt5CPU_gepp(run, p, a_pack, b_pack, c_pack);
    }

    free(a_pack);
    free(b_pack);
    free(c_pack);
}

template void opt5CPU_gemm_execute<float>(GemmRun<float>*);
