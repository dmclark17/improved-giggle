add_executable(generate generate.cpp)

target_link_libraries(generate igiggle-helper-lib)
target_include_directories(generate PUBLIC ${IGIGGLE_HELPER_INCLUDE_DIR})

target_link_libraries(generate mkl-benchmark-lib)
target_include_directories(generate PUBLIC ${MKL_BENCHMARK_INCLUDE_DIR})

find_package(gflags REQUIRED)
target_link_libraries(generate gflags)


set_target_properties(generate PROPERTIES CXX_STANDARD 11)
