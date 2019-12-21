#ifndef _IGIGGLE_MKL_H_
#define _IGIGGLE_MKL_H_

#include "data_manager.h"


void mkl_gemm_execute(GemmRun* run);

void accelerate_gemm_execute(GemmRun* run);

void naiveCPU_gemm_execute(GemmRun* run);

inline void opt1CPU_packA(GemmRun* run, unsigned int p, float* a_pack);
inline void opt1CPU_packB(GemmRun* run, unsigned int p, int j, float* b_pack);
inline void opt1CPU_unpackC(GemmRun* run, unsigned int j, unsigned int i, float* c_pack);
inline void opt1CPU_aux(float* a_pack, float* b_pack, float* c_pack);
inline void opt1CPU_gepb(GemmRun* run, unsigned int p, unsigned int j, float* a_pack, float* b_pack, float* c_pack);
inline void opt1CPU_gepp(GemmRun* run, unsigned int p, float* a_pack, float* b_pack, float* c_pack);
void opt1CPU_gemm_execute(GemmRun* run);


inline void opt2CPU_packA(GemmRun* run, unsigned int p, float* a_pack);
inline void opt2CPU_packB(GemmRun* run, unsigned int p, int j, float* b_pack);
inline void opt2CPU_unpackC(GemmRun* run, unsigned int j, unsigned int i, float* c_pack);
inline void opt2CPU_aux(float* a_pack, float* b_pack, float* c_pack);
inline void opt2CPU_aux_simd(float* a_pack, float* b_pack, float* c_pack);
inline void opt2CPU_gepb(GemmRun* run, unsigned int p, unsigned int j, float* a_pack, float* b_pack, float* c_pack);
inline void opt2CPU_gepp(GemmRun* run, unsigned int p, float* a_pack, float* b_pack, float* c_pack);
void opt2CPU_gemm_execute(GemmRun* run);


inline void opt3CPU_packA(GemmRun* run, unsigned int p, float* a_pack);
inline void opt3CPU_packB(GemmRun* run, unsigned int p, int j, float* b_pack);
inline void opt3CPU_unpackC(GemmRun* run, unsigned int j, unsigned int i, float* c_pack);
inline void opt3CPU_aux(float* a_pack, float* b_pack, float* c_pack);
inline void opt3CPU_aux_simd(float* a_pack, float* b_pack, float* c_pack);
inline void opt3CPU_gepb(GemmRun* run, unsigned int p, unsigned int j, float* a_pack, float* b_pack, float* c_pack);
inline void opt3CPU_gepp(GemmRun* run, unsigned int p, float* a_pack, float* b_pack, float* c_pack);
void opt3CPU_gemm_execute(GemmRun* run);


void naiveOMP_CPU_gemm_execute(GemmRun* run);
void opt1OMP_CPU_gemm_execute(GemmRun* run);

#endif /* end of include guard: _IGIGGLE_MKL_H_ */
