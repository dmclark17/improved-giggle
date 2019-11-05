add_subdirectory("${IGIGGLE_SOURCE_DIR}/extern/googletest" "extern/googletest")

macro(package_add_test TESTNAME)
    add_executable(${TESTNAME} ${ARGN})
    target_link_libraries(${TESTNAME} gtest gmock gtest_main)
    add_test(NAME ${TESTNAME} COMMAND ${TESTNAME})
    set_target_properties(${TESTNAME} PROPERTIES FOLDER tests)
    target_link_libraries(${TESTNAME} mkl-benchmark-lib)
    target_include_directories(${TESTNAME} PUBLIC ${MKL_BENCHMARK_INCLUDE_DIR})
endmacro()

package_add_test(mkl_test mkl_test.cpp)
