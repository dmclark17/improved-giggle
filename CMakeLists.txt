cmake_minimum_required(VERSION 3.10.1 FATAL_ERROR)
project(improved-giggle CXX)

include(CheckLanguage)

set(IGIGGLE_SOURCE_DIR $ENV{IMPROVED_GIGGLE_ROOT})
set(CMAKE_MODULE_PATH ${IGIGGLE_SOURCE_DIR}/cmake)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -mavx -mfma -Wall -Wextra")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -mavx -mfma -Wall -Wextra")

add_subdirectory(src)
add_subdirectory(cpu-giggle)


check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    add_subdirectory(gpu-giggle)
    add_definitions(-D_IG_HASCUDA)
endif()


add_subdirectory(apps)


option(PACKAGE_TESTS "Build the tests" ON)
if(PACKAGE_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
