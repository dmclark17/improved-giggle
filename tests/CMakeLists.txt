add_subdirectory("${IGIGGLE_SOURCE_DIR}/extern/googletest" "extern/googletest")

add_library(igiggle-test-helper-lib STATIC
    test_helper.cpp
    test_helper.h
)

target_link_libraries(igiggle-test-helper-lib igiggle-helper-lib)
target_include_directories(igiggle-test-helper-lib PUBLIC ${IGIGGLE_HELPER_INCLUDE_DIR})

target_link_libraries(igiggle-test-helper-lib gtest gmock gtest_main)

set_target_properties(igiggle-test-helper-lib PROPERTIES CXX_STANDARD 17)

set(IGIGGLE_TEST_HELPER_INCLUDE_DIR "${IGIGGLE_SOURCE_DIR}/tests")


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

    target_link_libraries(${TESTNAME} igiggle-test-helper-lib)
    target_include_directories(${TESTNAME} PUBLIC ${IGIGGLE_TEST_HELPER_INCLUDE_DIR})
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

    target_link_libraries(${TESTNAME} cpu-benchmark-lib)
    target_include_directories(${TESTNAME} PUBLIC ${CPU_BENCHMARK_INCLUDE_DIR})
endmacro()

#package_add_cpu_test(cpu_mkl_test cpu_mkl_test.cpp)
package_add_cpu_test(cpu_naive_test cpu_naive_test.cpp)
package_add_cpu_test(cpu_opt1_test cpu_opt1_test.cpp)
#package_add_cpu_test(cpu_opt2_test cpu_opt2_test.cpp)
#package_add_cpu_test(cpu_opt3_test cpu_opt3_test.cpp)
#package_add_cpu_test(cpu_omp_naive_test cpu_omp_naive_test.cpp)
#package_add_cpu_test(cpu_omp_opt1_test cpu_omp_opt1_test.cpp)


if(CMAKE_CUDA_COMPILER)
    package_add_gpu_test(gpu_cublas_test gpu_cublas_test.cpp)
    package_add_gpu_test(gpu_naive_test gpu_naive_test.cpp)
    package_add_gpu_test(gpu_opt1_test gpu_opt1_test.cpp)
    package_add_gpu_test(gpu_opt2_test gpu_opt2_test.cpp)
    package_add_gpu_test(gpu_opt3_test gpu_opt3_test.cpp)
endif()
