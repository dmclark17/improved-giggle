cmake_minimum_required(VERSION 3.10.1 FATAL_ERROR)
project(improved-giggle CXX)

set(IGIGGLE_SOURCE_DIR /home/david/improved-giggle)
set(CMAKE_MODULE_PATH ${IGIGGLE_SOURCE_DIR}/cmake)

add_subdirectory(src)
add_subdirectory(cpu-giggle)
add_subdirectory(apps)


option(PACKAGE_TESTS "Build the tests" ON)
if(PACKAGE_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
