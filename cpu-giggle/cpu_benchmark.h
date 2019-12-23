#ifndef _IGIGGLE_MKL_H_
#define _IGIGGLE_MKL_H_

#include "data_manager.h"

template <typename T>
void mkl_gemm_execute(GemmRun<T>* run);

template <typename T>
void accelerate_gemm_execute(GemmRun<T>* run);

template <typename T>
void naiveCPU_gemm_execute(GemmRun<T>* run);

template <typename T>
void opt1CPU_gemm_execute(GemmRun<T>* run);

template <typename T>
void opt2CPU_gemm_execute(GemmRun<T>* run);

template <typename T>
void opt3CPU_gemm_execute(GemmRun<T>* run);

template <typename T>
void naiveOMP_CPU_gemm_execute(GemmRun<T>* run);

template <typename T>
void opt1OMP_CPU_gemm_execute(GemmRun<T>* run);

#endif /* end of include guard: _IGIGGLE_MKL_H_ */
