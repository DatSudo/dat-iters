import numpy as np


class DatIters:

    def __init__(self, matrix: np.matrix, init_values: np.matrix) -> None:
        self.matrix: np.matrix = matrix
        self.init_values: np.matrix = init_values
        self.tolerance = 1e-10
        self.error = 1

    def jacobi(self):
        pass
    

    def gauss_seidel(self):
        pass


