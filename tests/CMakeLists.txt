add_subdirectory("${IGIGGLE_SOURCE_DIR}/extern/googletest" "extern/googletest")

macro(package_add_cpu_test TESTNAME)
    add_executable(${TESTNAME} ${ARGN})
    target_link_libraries(${TESTNAME} gtest gmock gtest_main)
    add_test(NAME ${TESTNAME} COMMAND ${TESTNAME})
    set_target_properties(${TESTNAME} PROPERTIES FOLDER tests)
    set_target_properties(${TESTNAME} PROPERTIES CXX_STANDARD 11)

    # Custom stuff
    target_link_libraries(${TESTNAME} igiggle-helper-lib)
    target_include_directories(${TESTNAME} PUBLIC ${IGIGGLE_HELPER_INCLUDE_DIR})

    target_link_libraries(${TESTNAME} cpu-benchmark-lib)
    target_include_directories(${TESTNAME} PUBLIC ${CPU_BENCHMARK_INCLUDE_DIR})
endmacro()

macro(package_add_gpu_test TESTNAME)
    add_executable(${TESTNAME} ${ARGN})
    target_link_libraries(${TESTNAME} gtest gmock gtest_main)
    add_test(NAME ${TESTNAME} COMMAND ${TESTNAME})
    set_target_properties(${TESTNAME} PROPERTIES FOLDER tests)
    set_target_properties(${TESTNAME} PROPERTIES CXX_STANDARD 11)

    # Custom stuff
    target_link_libraries(${TESTNAME} igiggle-helper-lib)
    target_include_directories(${TESTNAME} PUBLIC ${IGIGGLE_HELPER_INCLUDE_DIR})

    target_link_libraries(${TESTNAME} gpu-benchmark-lib)
    target_include_directories(${TESTNAME} PUBLIC ${GPU_BENCHMARK_INCLUDE_DIR})
endmacro()

package_add_cpu_test(mkl_test mkl_test.cpp)
package_add_cpu_test(naive_test naive_test.cpp)
package_add_cpu_test(opt1_test opt1_test.cpp)
package_add_cpu_test(opt2_test opt2_test.cpp)
package_add_cpu_test(opt3_test opt3_test.cpp)

if(CMAKE_CUDA_COMPILER)
    package_add_gpu_test(cublas_test cublas_test.cpp)
endif()
