add_library(cublas-benchmark-lib STATIC
    cublas_benchmark.cu
    gpu_benchmark.h
)

set(CUBLAS_BENCHMARK_INCLUDE_DIR "${IGIGGLE_SOURCE_DIR}/gpu-giggle" PARENT_SCOPE)

target_link_libraries(cublas-benchmark-lib -lcublas)

include_directories(cublas-benchmark-lib ${IGIGGLE_HELPER_INCLUDE_DIR})


set_target_properties(cublas-benchmark-lib PROPERTIES CXX_STANDARD 11)
set_target_properties(cublas-benchmark-lib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
