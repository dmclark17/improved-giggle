cmake_minimum_required(VERSION 3.10.1 FATAL_ERROR)
project(improved-giggle CXX)

set(CMAKE_MODULE_PATH /home/david/improved-giggle/cmake)
set(IGIGGLE_SOURCE_DIR /home/david/improved-giggle)

add_subdirectory(cpu-giggle)

add_subdirectory(apps)


option(PACKAGE_TESTS "Build the tests" ON)
if(PACKAGE_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
