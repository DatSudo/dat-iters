import numpy as np


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

        self.matrix = matrix
        self.constants = constants
        self.diag = np.diag(np.diag(matrix))
        self.diag_inverse = np.linalg.inv(self.diag)
        self.init_values = np.matrix([[2] for _ in range(matrix_shape)])
        self.tolerance = 1e-15
        self.error = 1.0
        self.num_iters = 0

    def jacobi(self) -> np.matrix:
        while self.error > self.tolerance:
            dx = self.diag_inverse * (
                    self.constants - self.matrix * self.init_values)
            next_values = self.init_values + dx

            e = abs(dx / self.init_values)
            self.error = max(map(max, e))
            
            self.init_values = next_values
            self.num_iters += 1

        return next_values

    def gauss_seidel(self) -> np.matrix:
        while self.error > self.tolerance:
            tri_lower_inverse = np.linalg.inv(np.tril(self.matrix))
            tri_upper = np.triu(self.matrix - self.diag)

            next_values = tri_lower_inverse * (
                    self.constants - tri_upper * self.init_values)

            e = abs((next_values - self.init_values) / next_values)
            self.error = max(map(max, e))

            self.init_values = next_values
            self.num_iters += 1

        return next_values

