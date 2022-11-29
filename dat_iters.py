import numpy as np


ITERATION_LIMIT = 10000

INSTRUCTION = """
===============================================================
* Make sure that your equations is already diagonally dominant.
* Inputs in each row (your equation's coefficients) should
be separated by spaces.
* Input for constant is the only one constant in your equation.

Examples:
If you have a system of equations

5x - y + 2z = 1
3x + 8y -2z = -3
x + y + 4z = 2

Inputs should be

---------------------------------------------------------------
Matrix shape: 3 (you have 3 equations, 3 unknown variables)
---------------------------------------------------------------
Row 1: 5 -1 2
Row 1 constant: 1
---------------------------------------------------------------
Row 2: 3 8 -2
Row 2 constant: -3
---------------------------------------------------------------
Row 3: 1 1 4
Row 3 constant: 2
---------------------------------------------------------------
===============================================================
"""


class DatIters:

    def __init__(self, matrix_shape: int, 
                       matrix: np.matrix,
                       constants: np.matrix) -> None:

        self.matrix: np.matrix = matrix
        self.constants: np.matrix = constants
        self.diag = np.diag(np.diag(matrix))
        self.diag_inverse: np.matrix = np.linalg.inv(self.diag)
        self.init_values: np.matrix = np.matrix(
                [[2] for _ in range(matrix_shape)]
        )
        # self.tolerance: float = 1e-15
        # self.error: float = 1.0

    def jacobi(self) -> np.matrix:
        for _ in range(ITERATION_LIMIT):
            dx = self.diag_inverse * (
                    self.constants - self.matrix * self.init_values)
            next_values = self.init_values + dx
            
            self.init_values = next_values

        return next_values

    def gauss_seidel(self) -> np.matrix:
        for _ in range(ITERATION_LIMIT):
            tri_lower_inverse = np.linalg.inv(np.tril(self.matrix))
            tri_upper = np.triu(self.matrix - self.diag)

            next_values = tri_lower_inverse * (
                    self.constants - tri_upper * self.init_values)

            self.init_values = next_values

        return next_values


