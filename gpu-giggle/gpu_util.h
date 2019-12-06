#ifndef _IGIGGLE_GPU_UTIL_H_
#define _IGIGGLE_GPU_UTIL_H_

#include "data_manager.h"


void init_cuda_matrices(GemmRun* run,
                        size_t* pitch_A, size_t* pitch_B, size_t* pitch_C,
                        float** cuda_A, float** cuda_B, float** cuda_C);


void deinit_cuda_matrices(GemmRun* run, size_t pitch_C,
                          float* cuda_A, float* cuda_B, float* cuda_C);

#endif /* end of include guard: _IGIGGLE_GPU_UTIL_H_ */
