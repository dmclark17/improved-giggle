import numpy as np


def generate_matrix_diff(length):
    mat = np.zeros((length, length))
    for i in range(length):
        for j in range(length):
            mat[i, j] = float(i - j) ** 2 + 1
    return mat


def generate_matrix_prod(length):
    mat = np.zeros((length, length))
    for i in range(length):
        for j in range(length):
            mat[i, j] = float(i * j) + (i * length) + j
            mat[i, j] *= mat[i, j]
            mat[i, j] += 1
    return mat


if __name__ == "__main__":
    np.set_printoptions(suppress=True)

    A = generate_matrix_prod(16)
    B = generate_matrix_diff(16)

    C = np.matmul(A, B)

    C_test = np.zeros((16, 16))

    i = 0
    # print(B)


    # print(B[:2, :2])
    # print(A[:, [i, i+1]])
    # print(np.matmul(A[:, [i, i+1]], B[:2, :2]))

    # for i in range(0, 16, 2):
    # # i = 0
    #     # C_test += np.matmul(A[:, [i, i+1]], B[[i, i+1], [i, i+1]])
    #     print("A", A[:, [i, i+1]])
    #     print("B", np.shape(B[[i, i+1], :]))
    #     C_test += np.matmul(A[:, [i, i+1]], B[[i, i+1], :])

    i = 0
    C_test += np.matmul(A[:, [i, i+1]], B[[i, i+1], :])

    # print(C)

    # print("C_test")
    # print(C_test)

    diff = np.sum(np.power(C - C_test, 2))

    # print(diff)

    np.set_printoptions(suppress=True)

    print(C[7, 1])
    print(C[7, 0])
