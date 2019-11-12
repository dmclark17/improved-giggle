#ifndef _IGIGGLE_MKL_H_
#define _IGIGGLE_MKL_H_

#include "data_manager.h"


void mkl_gemm_execute(GemmRun* run);

void naiveCPU_gemm_execute(GemmRun* run);

void opt1CPU_gemm_execute(GemmRun* run);

#endif /* end of include guard: _IGIGGLE_MKL_H_ */
