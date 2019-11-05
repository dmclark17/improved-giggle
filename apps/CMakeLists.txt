add_executable(generate generate.cpp)


target_link_libraries(generate mkl-benchmark-lib)
target_include_directories(generate PUBLIC ${MKL_BENCHMARK_INCLUDE_DIR})


set_target_properties(generate PROPERTIES CXX_STANDARD 11)
