import numpy as np


def open_matrix(file_path):
    """Open the channel files"""
    with open(file_path, "r") as file:
        lines = file.readlines()

    matrix = []

    for line in lines:
        # We devide the line in elements using space as a separator :
        elements = line.strip().split()
        row = [float(element) for element in elements]
        matrix.append(row)
    return np.array(matrix)


def blahut_arimoto(f_y_x):
    m = f_y_x.shape[0] # number of row

    # Initialization :
    fx = np.ones((1, m)) / m
    # Compute the r(x) that maximizes the capacity
    for iteration in range(10000):
        fxt = fx.reshape(-1, 1) #transpose
        r_x_y = fxt * f_y_x / np.sum(fxt * f_y_x, axis=0)

        f_x_plus_1 = np.prod(np.power(r_x_y, f_y_x), axis=1)
        f_x_plus_1 = f_x_plus_1 / np.sum(f_x_plus_1)

        difference = np.linalg.norm(f_x_plus_1 - fx)
        fx = f_x_plus_1
        if difference < 1e-12:
            # stopping condition
            break

    # Calculate the capacity
    c = 0
    for i in range(m):
        if fx[i] > 0:
            c += np.sum(fx[i] * f_y_x[i, :] *
                        np.log2(r_x_y[i, :] / fx[i] + 1e-16))
    return c

f_y_x = open_matrix("channel_A.txt")
C = blahut_arimoto(f_y_x)

print('Capacity: ', C)
