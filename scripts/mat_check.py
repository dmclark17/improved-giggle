import numpy as np


def generate_matrix_diff(length):
    mat = np.zeros((length, length))
    for i in range(length):
        for j in range(length):
            mat[i, j] = float(i - j)
    return mat


def generate_matrix_prod(length):
    mat = np.zeros((length, length))
    for i in range(length):
        for j in range(length):
            mat[i, j] = float(i * j)
    return mat


if __name__ == "__main__":
    A = generate_matrix_prod(3)
    B = generate_matrix_diff(3)

    C = np.matmul(A, B)

    print(C)
