#ifndef _IGIGGLE_CUBLAS_H_
#define _IGIGGLE_CUBLAS_H_

#include "data_manager.h"


void cublass_gemm_execute(GemmRun<float>* run);

void naiveGPU_gemm_execute(GemmRun<float>* run);

void opt1GPU_gemm_execute(GemmRun<float>* run);

void opt2GPU_gemm_execute(GemmRun<float>* run);

void opt3GPU_gemm_execute(GemmRun<float>* run);

#endif /* end of include guard: _IGIGGLE_CUBLAS_H_ */
