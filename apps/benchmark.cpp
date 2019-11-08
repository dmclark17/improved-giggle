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
DEFINE_string(sizes, "3,4", "Comma seperated sizes of matrices to run");

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

    if (FLAGS_benchmark.compare("mkl") == 0) {
        std::cout << "Running MKL Benchmark for " << matrix_sizes.size() << " sizes" << std::endl;
        for (int size : matrix_sizes) {
            double duration;
            for (int i = 0; i < FLAGS_number; i++) {
                GemmRun* run;
                allocate_run(&run, size);
                generate_matrix_random(run->a, run->lda, run->m);
                generate_matrix_random(run->b, run->ldb, run->k);

                std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
                mkl_gemm_execute(run);
                std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
                duration += (double) std::chrono::duration_cast<std::chrono::milliseconds>( t2 - t1 ).count();

                deallocate_run(run);
            }
            std::cout << "size: " << size << " time: " << duration / FLAGS_number << " ms" << std::endl;
        }
    } else if (FLAGS_benchmark.compare("cublas") == 0) {
        #ifdef _IG_HASCUDA
        std::cout << "Running cuBLAS Benchmark for " << matrix_sizes.size() << " sizes" << std::endl;
        for (int size : matrix_sizes) {
            double duration;
            for (int i = 0; i < FLAGS_number; i++) {
                GemmRun* run;
                allocate_run(&run, size);
                generate_matrix_random(run->a, run->lda, run->m);
                generate_matrix_random(run->b, run->ldb, run->k);

                std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
                cublass_gemm_execute(run);
                std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
                duration += (double) std::chrono::duration_cast<std::chrono::milliseconds>( t2 - t1 ).count();

                deallocate_run(run);
            }
            std::cout << "size: " << size << " time: " << duration / FLAGS_number << " ms" << std::endl;
        }
        #else
        std::cout << "CUDA not supported" << std::endl;
        #endif
    } else if (FLAGS_benchmark.compare("naiveCPU") == 0) {
        std::cout << "Running naiveCPU Benchmark for " << matrix_sizes.size() << " sizes" << std::endl;
        for (int size : matrix_sizes) {
            double duration;
            for (int i = 0; i < FLAGS_number; i++) {
                GemmRun* run;
                allocate_run(&run, size);
                generate_matrix_random(run->a, run->lda, run->m);
                generate_matrix_random(run->b, run->ldb, run->k);

                std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
                naive_gemm_execute(run);
                std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
                duration += (double) std::chrono::duration_cast<std::chrono::milliseconds>( t2 - t1 ).count();

                deallocate_run(run);
            }
            std::cout << "size: " << size << " time: " << duration / FLAGS_number << " ms" << std::endl;
        }
    } else {
        std::cout << "Benchmark " << FLAGS_benchmark << " not supported" << std::endl;
    }

}
