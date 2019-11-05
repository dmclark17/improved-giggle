cmake_minimum_required(VERSION 3.10.1 FATAL_ERROR)
project(improved-giggle CXX)

set(CMAKE_MODULE_PATH /home/david/improved-giggle/cmake)

add_subdirectory(cpu-giggle)

add_subdirectory(apps)
