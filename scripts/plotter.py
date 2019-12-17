import matplotlib.pyplot as plt


def parse_output(filename, results={}):
    with open(filename, 'r') as f:
        result_string = f.readline()

        result_list = []
        results[result_string.split(' ')[1]] = result_list

        for line in f:
            line_list = line.strip().split(" ")
            result_list.append((float(line_list[1]), float(line_list[3])))

    return results


def plot_output(results):
    plt.figure()
    for key in results:
        plt.plot([x[0] for x in results[key]], [x[1] for x in results[key]],
                 label=key)

    plt.xlabel("Matrix Size")
    plt.ylabel("Time (ms)")
    plt.title("GPU Matrix-matrix multiplication benchmark")
    plt.legend()
    plt.savefig("gpu.pdf")


def plot_double_output(results_one, results_two):
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    for key in results_one:
        plt.plot([x[0] for x in results_one[key]], [x[1] for x in results_one[key]],
                 label=key)

    plt.xlabel("Matrix Size")
    plt.ylabel("Time (ms)")
    plt.legend()

    plt.subplot(1, 2, 2)
    for key in results_two:
        plt.plot([x[0] for x in results_two[key]], [x[1] for x in results_two[key]],
                 label=key)

    plt.xlabel("Matrix Size")
    plt.ylabel("Time (ms)")
    plt.legend()

    plt.savefig("cpu_cluster.pdf")


if __name__ == "__main__":
    """
    ./apps/benchmark --number 4 --sizes 512,512,1024,1536,2048 --benchmark naiveCPU > naiveCPU_results.txt
    ./apps/benchmark --number 4 --sizes 512,512,1024,1536,2048 --benchmark opt1CPU > opt1CPU_results.txt
    ./apps/benchmark --number 4 --sizes 512,512,1024,1536,2048 --benchmark opt2CPU > opt2CPU_results_short.txt
    ./apps/benchmark --number 4 --sizes 512,512,1024,1536,2048,2560,3072,3584 --benchmark mkl > mkl_one_results.txt
    ./apps/benchmark --number 4 --sizes 512,512,1024,1536,2048,2560,3072,3584 --benchmark opt2CPU > opt2CPU_results.txt
    ./apps/benchmark --number 4 --sizes 512,512,1024,1536,2048,2560,3072,3584 --benchmark opt3CPU > opt3CPU_results.txt


    ./apps/benchmark --number 4 --sizes 512,512,1024,1536,2048,2560,3072,3584 --benchmark naiveGPU > naiveGPU_results.txt
    ./apps/benchmark --number 4 --sizes 512,512,1024,1536,2048,2560,3072,3584 --benchmark opt1GPU > opt1GPU_results.txt
    ./apps/benchmark --number 4 --sizes 512,512,1024,1536,2048,2560,3072,3584 --benchmark opt2GPU > opt2GPU_results.txt
    ./apps/benchmark --number 4 --sizes 512,512,1024,1536,2048,2560,3072,3584 --benchmark cublas > cublas_results.txt
    ./apps/benchmark --number 4 --sizes 512,512,1024,1536,2048,2560,3072,3584 --benchmark mkl > mkl_four_results.txt
    """

    # results = parse_output("../build/mkl_four_results.txt", results={})
    # results = parse_output("../build/naiveGPU_results.txt", results=results)
    # results = parse_output("../build/opt1GPU_results.txt", results=results)
    # results = parse_output("../build/opt2GPU_results.txt", results=results)
    # results = parse_output("../build/cublas_results.txt", results=results)
    #
    # plot_output(results)

    results = parse_output("small_6130.txt", results={})
    results = parse_output("small_8268.txt", results=results)
    results = parse_output("small_k80.txt", results=results)
    results = parse_output("small_v100.txt", results=results)


    results_tiny = parse_output("tiny_6130.txt", results={})
    results_tiny = parse_output("tiny_8268.txt", results=results_tiny)
    results_tiny = parse_output("tiny_k80.txt", results=results_tiny)
    results_tiny = parse_output("tiny_v100.txt", results=results_tiny)


    print(results)
    print(results_tiny)

    plot_double_output(results_tiny, results)
