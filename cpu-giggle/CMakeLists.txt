add_library(cpu-benchmark-lib STATIC
    mkl_benchmark.cpp
    accelerate_benchmark.cpp
    opt5_benchmark.cpp
    opt4_benchmark.cpp
    opt3_benchmark.cpp
    opt2_benchmark.cpp
    opt1_benchmark.cpp
    naive_benchmark.cpp
    naive_omp_benchmark.cpp
    opt1_omp_benchmark.cpp
    cpu_benchmark.h
)

set(CPU_BENCHMARK_INCLUDE_DIR "${IGIGGLE_SOURCE_DIR}/cpu-giggle" PARENT_SCOPE)

if(MKL_FOUND)
    target_link_libraries(cpu-benchmark-lib ${MKL_LIBRARIES} ${CMAKE_DL_LIBS} ${CMAKE_THREAD_LIBS_INIT})
    include_directories(${MKL_INCLUDE_DIRS})
    set(IG_MKL_FOUND "true" PARENT_SCOPE)
endif()
target_link_libraries(cpu-benchmark-lib ${BLAS_LIBRARIES} ${CMAKE_DL_LIBS} ${CMAKE_THREAD_LIBS_INIT})


include_directories(cpu-benchmark-lib ${IGIGGLE_HELPER_INCLUDE_DIR})

set_target_properties(cpu-benchmark-lib PROPERTIES CXX_STANDARD 17)

# Outputs the assembly
#set_target_properties(cpu-benchmark-lib PROPERTIES COMPILE_FLAGS "-save-temps")
#set_target_properties(cpu-benchmark-lib PROPERTIES COMPILE_FLAGS "-Wa,-adhln=opt3_benchmark.s -g -fverbose-asm")
