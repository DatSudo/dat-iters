import numpy as np
import re

from dat_iters import *


def main() -> None:
    matrix_shape = int(input("Matrix shape: "))

    matrix = np.zeros(shape=(matrix_shape, matrix_shape))
    constants = np.zeros(shape=(matrix_shape, 1))

    for r in range(matrix_shape):
        print("-"*30)
        while True:
            new_row = list(map(float, input(f"Row {r+1}: ").split(" ")))
            if abs(new_row[r]) >= sum([abs(n) for n in (new_row[:r] + new_row[r+1:])]) and \
                    len(new_row) == matrix_shape:
                matrix[r] = new_row
                break
            else:
                print("--- Seems like your matrix is not diagonally \
                      dominant or is not a square matrix ---")
        while True:
            new_const = list(map(float, input(f"Row {r+1} constant: ").split(" ")))
            if len(new_const) == 1:
                constants[r] = new_const
                break
            else:
                print("--- Wrong input ---")

    iter = DatIters(5, matrix, constants)

    ans1 = iter.jacobi()
    ans2 = iter.gauss_seidel()

    print(f"JACOBI:\n")
    for i in range(matrix_shape):
        print(f"x_{i+1} = ", end="")
        print(re.sub('[\[\]]', '', np.array_str(ans1[i][0])))
    print("-"*30)
    print(f"GAUSS-SEIDEL:\n")
    for i in range(matrix_shape):
        print(f"x_{i+1} = ", end="")
        print(re.sub('[\[\]]', '', np.array_str(ans2[i][0])))

if __name__ == "__main__":
    main()
