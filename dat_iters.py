import numpy as np


class DatIters:

    def __init__(self, matrix_shape: int, 
                       matrix: np.matrix,
                       constants: np.matrix) -> None:

        self.matrix: np.matrix = matrix
        self.constants: np.matrix = constants
        self.diag_inverse: np.matrix = np.linalg.inv(np.diag(np.diag(matrix)))
        self.init_values: np.matrix = np.matrix([[2] for _ in range(matrix_shape)])
        self.tolerance: np.float128 = 1e-16
        self.error: np.float128 = 1

    def jacobi(self) -> np.matrix:
        while self.error > self.tolerance:
            dx = self.diag_inverse * (
                    self.constants - self.matrix * self.init_values)
            next_values = self.init_values + dx
            
            self.error = max(map(max, abs(dx / self.init_values)))

            self.init_values = next_values

        return next_values

    def gauss_seidel(self) -> np.matrix:
        while self.error > self.tolerance:
            pass

    def is_diag_dominant(self) -> bool:
        pass


