add_executable(benchmark benchmark.cpp)

target_link_libraries(benchmark igiggle-helper-lib)
target_include_directories(benchmark PUBLIC ${IGIGGLE_HELPER_INCLUDE_DIR})

target_link_libraries(benchmark cpu-benchmark-lib)
target_include_directories(benchmark PUBLIC ${CPU_BENCHMARK_INCLUDE_DIR})

if(CMAKE_CUDA_COMPILER)
    target_link_libraries(benchmark gpu-benchmark-lib)
    target_include_directories(benchmark PUBLIC ${GPU_BENCHMARK_INCLUDE_DIR})
endif()


find_package(gflags REQUIRED)
target_link_libraries(benchmark gflags)


set_target_properties(benchmark PROPERTIES CXX_STANDARD 11)
