cmake_minimum_required(VERSION 3.10.1 FATAL_ERROR)
project(improved-giggle CXX)

add_executable(generate generate.cpp)

set_target_properties(generate PROPERTIES CXX_STANDARD 11)
