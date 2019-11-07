add_library(mkl-benchmark-lib STATIC
    mkl_benchmark.cpp
    cpu_benchmark.h
)

set(MKL_BENCHMARK_INCLUDE_DIR "/home/david/improved-giggle/cpu-giggle" PARENT_SCOPE)

find_package(BLAS REQUIRED)
find_package(MKL)
if(MKL_FOUND)
  target_link_libraries(mkl-benchmark-lib ${MKL_LIBRARIES} ${CMAKE_DL_LIBS} ${CMAKE_THREAD_LIBS_INIT})
  include_directories(${MKL_INCLUDE_DIRS})
endif()
target_link_libraries(mkl-benchmark-lib ${BLAS_LIBRARIES} ${CMAKE_DL_LIBS} ${CMAKE_THREAD_LIBS_INIT})


include_directories(mkl-benchmark-lib ${IGIGGLE_HELPER_INCLUDE_DIR})


set_target_properties(mkl-benchmark-lib PROPERTIES CXX_STANDARD 11)
