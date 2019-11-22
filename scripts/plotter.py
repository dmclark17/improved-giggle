import matplotlib.pyplot as plt


def parse_output(filename, results={}):
    with open(filename, 'r') as f:
        result_string = f.readline()

        result_list = []
        results[result_string.split(' ')[1]] = result_list

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
    plt.title("Matrix-matrix multiplication benchmark (O0)")
    plt.legend()
    plt.savefig("CPUO0.png")


if __name__ == "__main__":
    results = parse_output("../build/naiveCPU.txt")
    results = parse_output("../build/opt1CPU.txt", results=results)
    results = parse_output("../build/opt2CPU.txt", results=results)
    print(results)
    plot_output(results)
