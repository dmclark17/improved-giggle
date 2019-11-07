import matplotlib.pyplot as plt


def parse_output(filename, results={}):
    with open(filename, 'r') as f:
        result_string = f.readline()
        result_list = []
        if ("MKL" in result_string):
            results["MKL"] = result_list
        elif ("cuBLAS" in result_string):
            results["cuBLAS"] = result_list

        for line in f:
            line_list = line.strip().split(" ")
            result_list.append((int(line_list[1]), int(line_list[3])))

    return results


def plot_output(results):
    plt.figure()
    for key in results:
        plt.plot([x[0] for x in results[key]], [x[1] for x in results[key]],
                 label=key)

    plt.xlabel("Matrix Size")
    plt.ylabel("Time (ms)")
    plt.title("Matrix-matrix multiplication benchmark")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    results = parse_output("../build/mkl_results.txt")
    results = parse_output("../build/cublas_results.txt", results=results)
    plot_output(results)
