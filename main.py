from dat_iters import *

import re
import sys

# Try importing numpy
try:
    import numpy as np
except ImportError:
    print(
    """
    --- Install first the required package: numpy
    --- To install it run: python -m pip install numpy
    --- If that doesn't work, try replacing 'python' with 'python3' or 'pip' with 'pip3'
    """
    )

    # Exit program when numpy isn't installed
    sys.exit(0)


def main() -> None:
    print(INSTRUCTION)

    # Get matrix shape
    while True:
        matrix_shape = input("Matrix shape: ")
        if matrix_shape.isnumeric() and int(matrix_shape) >= 1:
            matrix_shape = int(matrix_shape)
            break
        else:
            print("--- Input is not an integer ---")

    # Initialize matrix
    matrix = np.zeros(shape=(matrix_shape, matrix_shape))
    # Initialize matrix for constants
    constants = np.zeros(shape=(matrix_shape, 1))

    # Get coefficients for each equation/row
    for r in range(matrix_shape):
        print("-"*63)
        while True:
            new_row = list(map(float, input(f"Row {r+1}: ").split(" ")))

            # Check if diagonally dominant
            if abs(new_row[r]) >= sum([abs(n) for n in (new_row[:r] + new_row[r+1:])]) and \
                    len(new_row) == matrix_shape: # Check if input is compatible with matrix shape
                matrix[r] = new_row
                break
            else:
                print("--- Seems like your matrix is not diagonally dominant or is not a square matrix ---")
        while True:
            new_const = list(map(float, input(f"Row {r+1} constant: ").split(" ")))
            if len(new_const) == 1:
                constants[r] = new_const
                break
            else:
                print("--- Wrong input ---")

    print("-"*63)
    while True:
        iter_opt = input("Iterative method to use: [J]acobi or [G]auss-Seidel\n>").lower()
        if iter_opt not in {"j", "g"}:
            print("--- Type j/J for Jacobi or g/G for Gauss-Seidel ---")
        else:
            break

    iter = DatIters(matrix_shape, matrix, constants)

    if iter_opt == "j":
        result = iter.jacobi()
    else:
        result = iter.gauss_seidel()

    # Print results
    print("="*63)
    print(f"RESULT:\n")
    for i in range(matrix_shape):
        print(f"x_{i+1} = ", end="")
        print(re.sub('[\[\]]', '', np.array_str(result[i][0])))
    print(f"\nNumber of iterations = {iter.num_iters}")

    print("="*63)


if __name__ == "__main__":
    main()
