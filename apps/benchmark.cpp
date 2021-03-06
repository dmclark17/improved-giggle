#include <cstdio>
#include <string>
#include <iostream>
#include <chrono>

#include <gflags/gflags.h>

#include "data_manager.h"
#include "cpu_benchmark.h"
#ifdef _IG_HASCUDA
#include "gpu_benchmark.h"
#endif

DEFINE_string(benchmark, "mkl", "benchmark to run");
DEFINE_uint64(number, 1, "Amount of times to run each size");
DEFINE_string(sizes, "3,4", "Comma separated sizes of matrices to run");

int main(int argc, char *argv[]) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    std::string delimiter = ",";
    std::string token;
    std::string remain = FLAGS_sizes;
    std::size_t found;

    std::vector<int> matrix_sizes;

    while((found = remain.find(delimiter)) != std::string::npos) {
        token = remain.substr(0, found);
        remain = remain.substr(found+1, std::string::npos);
        matrix_sizes.push_back(std::stoi(token));
    }
    matrix_sizes.push_back(std::stoi(remain));

    void (*gemm_execute)(GemmRun<float>*);

    if (FLAGS_benchmark == "mkl") {
        #ifdef _IG_HASMKL
        std::cout << "Running MKL Benchmark for " << matrix_sizes.size() << " sizes" << std::endl;
        gemm_execute = mkl_gemm_execute<float>;
        #else
        std::cout << "MKL not found: mkl not supported" << std::endl;
        return 1;
        #endif

    } else if (FLAGS_benchmark == "accelerate") {
        #ifdef __APPLE__
        std::cout << "Running naiveCPU Benchmark for " << matrix_sizes.size() << " sizes" << std::endl;
        gemm_execute = accelerate_gemm_execute<float>;
        #else
        std::cout << "Accelerate not found: apple not supported" << std::endl;
        return 1;
        #endif

    } else if (FLAGS_benchmark == "cublas") {
        #ifdef _IG_HASCUDA
        std::cout << "Running cuBLAS Benchmark for " << matrix_sizes.size() << " sizes" << std::endl;
        gemm_execute = cublass_gemm_execute;
        #else
        std::cout << "CUDA compiler not found: cublas not supported" << std::endl;
        return 1;
        #endif

    } else if (FLAGS_benchmark == "naiveGPU") {
        #ifdef _IG_HASCUDA
        std::cout << "Running naiveGPU Benchmark for " << matrix_sizes.size() << " sizes" << std::endl;
        gemm_execute = naiveGPU_gemm_execute;
        #else
        std::cout << "CUDA compiler not found: gpu not supported" << std::endl;
        return 1;
        #endif

    } else if (FLAGS_benchmark == "opt1GPU") {
        #ifdef _IG_HASCUDA
        std::cout << "Running opt1GPU Benchmark for " << matrix_sizes.size() << " sizes" << std::endl;
        gemm_execute = opt1GPU_gemm_execute;
        #else
        std::cout << "CUDA compiler not found: gpu not supported" << std::endl;
        return 1;
        #endif

    } else if (FLAGS_benchmark == "opt2GPU") {
        #ifdef _IG_HASCUDA
        std::cout << "Running opt2GPU Benchmark for " << matrix_sizes.size() << " sizes" << std::endl;
        gemm_execute = opt2GPU_gemm_execute;
        #else
        std::cout << "CUDA compiler not found: gpu not supported" << std::endl;
        return 1;
        #endif

    } else if (FLAGS_benchmark == "opt3GPU") {
        #ifdef _IG_HASCUDA
        std::cout << "Running opt3GPU Benchmark for " << matrix_sizes.size() << " sizes" << std::endl;
        gemm_execute = opt3GPU_gemm_execute;
        #else
        std::cout << "CUDA compiler not found: gpu not supported" << std::endl;
        return 1;
        #endif

    } else if (FLAGS_benchmark == "naiveCPU") {
        std::cout << "Running naiveCPU Benchmark for " << matrix_sizes.size() << " sizes" << std::endl;
        gemm_execute = naiveCPU_gemm_execute<float>;

    } else if (FLAGS_benchmark == "opt1CPU") {
        std::cout << "Running opt1CPU Benchmark for " << matrix_sizes.size() << " sizes" << std::endl;
        gemm_execute = opt1CPU_gemm_execute<float>;

    } else if (FLAGS_benchmark == "opt2CPU") {
        std::cout << "Running opt2CPU Benchmark for " << matrix_sizes.size() << " sizes" << std::endl;
        #ifdef __AVX512F__
        std::cout << "Using AVX-512F" << std::endl;
        #else
        std::cout << "Using AVX" << std::endl;
        #endif
        gemm_execute = opt2CPU_gemm_execute<float>;

    } else if (FLAGS_benchmark == "opt3CPU") {
        std::cout << "Running opt3CPU Benchmark for " << matrix_sizes.size() << " sizes" << std::endl;
        #ifdef __AVX512F__
        std::cout << "Using AVX-512F" << std::endl;
        #else
        std::cout << "Using AVX" << std::endl;
        #endif
        gemm_execute = opt3CPU_gemm_execute<float>;

    } else if (FLAGS_benchmark == "opt4CPU") {
        std::cout << "Running opt4CPU Benchmark for " << matrix_sizes.size() << " sizes" << std::endl;
        #ifdef __AVX512F__
        std::cout << "Not using AVX-512F" << std::endl;
        #else
        std::cout << "Using AVX" << std::endl;
        #endif
        gemm_execute = opt4CPU_gemm_execute<float>;

    } else if (FLAGS_benchmark == "opt5CPU") {
        std::cout << "Running opt5CPU Benchmark for " << matrix_sizes.size() << " sizes" << std::endl;
        #ifdef __AVX512F__
        std::cout << "Not using AVX-512F" << std::endl;
        #else
        std::cout << "Using AVX" << std::endl;
        #endif
        gemm_execute = opt5CPU_gemm_execute<float>;

    } else if (FLAGS_benchmark == "naiveOmpCPU") {
        std::cout << "Running naiveOmpCPU Benchmark for " << matrix_sizes.size() << " sizes" << std::endl;
        gemm_execute = naiveOMP_CPU_gemm_execute<float>;

    } else if (FLAGS_benchmark == "opt1OmpCPU") {
        std::cout << "Running opt1OmpCPU Benchmark for " << matrix_sizes.size() << " sizes" << std::endl;
        #ifdef __AVX512F__
        std::cout << "Using AVX-512F" << std::endl;
        #else
        std::cout << "Using AVX" << std::endl;
        #endif
        gemm_execute = opt1OMP_CPU_gemm_execute<float>;

    } else {
        std::cout << "Benchmark " << FLAGS_benchmark << " not supported" << std::endl;
        return 1;
    }


    for (int size : matrix_sizes) {
        double duration = 0;
        for (unsigned int i = 0; i < FLAGS_number; i++) {
            GemmRun<float>* run;
            allocate_run<float>(&run, size);
            generate_matrix_random<float>(run->a, run->lda, run->m);
            generate_matrix_random<float>(run->b, run->ldb, run->k);

            std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
            gemm_execute(run);
            std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
            duration += (double) std::chrono::duration_cast<std::chrono::milliseconds>( t2 - t1 ).count();

            deallocate_run(run);
        }
        std::cout << "size: " << size << " time: " << duration / FLAGS_number << " ms" << std::endl;
    }

    return 0;
}
