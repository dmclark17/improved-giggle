add_library(gpu-benchmark-lib STATIC
    cublas_benchmark.cu
    naive_benchmark.cu
    opt1_benchmark.cu
    opt2_benchmark.cu
    opt3_benchmark.cu
    gpu_benchmark.h
    gpu_util.cu
    gpu_util.h
)

set(GPU_BENCHMARK_INCLUDE_DIR "${IGIGGLE_SOURCE_DIR}/gpu-giggle" PARENT_SCOPE)

target_link_libraries(gpu-benchmark-lib -lcublas)

include_directories(gpu-benchmark-lib ${IGIGGLE_HELPER_INCLUDE_DIR})


set_target_properties(gpu-benchmark-lib PROPERTIES CXX_STANDARD 17)
set_target_properties(gpu-benchmark-lib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
