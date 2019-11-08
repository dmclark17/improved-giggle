add_library(gpu-benchmark-lib STATIC
    cublas_benchmark.cu
    gpu_benchmark.h
)

set(GPU_BENCHMARK_INCLUDE_DIR "${IGIGGLE_SOURCE_DIR}/gpu-giggle" PARENT_SCOPE)

target_link_libraries(gpu-benchmark-lib -lcublas)

include_directories(gpu-benchmark-lib ${IGIGGLE_HELPER_INCLUDE_DIR})


set_target_properties(gpu-benchmark-lib PROPERTIES CXX_STANDARD 11)
set_target_properties(gpu-benchmark-lib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
