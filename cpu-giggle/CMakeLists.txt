add_library(cpu-benchmark-lib STATIC
    mkl_benchmark.cpp
    opt1_benchmark.cpp
    naive_benchmark.cpp
    cpu_benchmark.h
)

set(CPU_BENCHMARK_INCLUDE_DIR "${IGIGGLE_SOURCE_DIR}/cpu-giggle" PARENT_SCOPE)

find_package(BLAS REQUIRED)
find_package(MKL)
if(MKL_FOUND)
  target_link_libraries(cpu-benchmark-lib ${MKL_LIBRARIES} ${CMAKE_DL_LIBS} ${CMAKE_THREAD_LIBS_INIT})
  include_directories(${MKL_INCLUDE_DIRS})
  set(IG_MKL_FOUND "true" PARENT_SCOPE)
endif()
target_link_libraries(cpu-benchmark-lib ${BLAS_LIBRARIES} ${CMAKE_DL_LIBS} ${CMAKE_THREAD_LIBS_INIT})


include_directories(cpu-benchmark-lib ${IGIGGLE_HELPER_INCLUDE_DIR})


set_target_properties(cpu-benchmark-lib PROPERTIES CXX_STANDARD 11)
