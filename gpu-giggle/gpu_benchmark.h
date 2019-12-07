#ifndef _IGIGGLE_CUBLAS_H_
#define _IGIGGLE_CUBLAS_H_

#include "data_manager.h"


void cublass_gemm_execute(GemmRun* run);

void naiveGPU_gemm_execute(GemmRun* run);

void opt1GPU_gemm_execute(GemmRun* run);

#endif /* end of include guard: _IGIGGLE_CUBLAS_H_ */
