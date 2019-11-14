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
    A = generate_matrix_prod(4)
    B = generate_matrix_diff(4)

    C = np.matmul(A, B)

    C_test = np.zeros((4, 4))

    i = 0
    print(B)


    print(B[:2, :2])
    print(A[:, [i, i+1]])
    print(np.matmul(A[:, [i, i+1]], B[:2, :2]))

    # for i in range(0, 4, 2):
    i = 0
    C_test += np.matmul(A[:, [i, i+1]], B[[i, i+1], [i, i+1]])

    # print(C)

    print("C_test")
    print(C_test)
