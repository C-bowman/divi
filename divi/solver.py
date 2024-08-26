from numpy import ndarray, zeros
from scipy.linalg import solve_banded
from divi.operators import build_difference_opreator
from divi.utils import diagonal_form


class DiffusionSolver:
    def __init__(self, radius: ndarray):
        self.radius = radius
        self.grad = build_difference_opreator(axis=self.radius, order=1, n_points=3)
        self.laplace = build_difference_opreator(axis=self.radius, order=2, n_points=3)

    def solve(
        self,
        diffusivity: ndarray,
        diffusivity_gradient: ndarray,
        core_value: float,
        sol_value: float,
    ) -> ndarray:
        # total matrix operator
        A = diffusivity_gradient[:, None] * self.grad + diffusivity[:, None] * self.laplace

        # modify the operator to impose boundary conditions
        A[0, :] = 0.0
        A[0, 0] = 1.0
        A[-1, :] = 0.0
        A[-1, -1] = 1.0

        # target vector
        b = zeros(self.radius.size)
        b[0] = core_value
        b[-1] = sol_value

        sol = solve_banded((1, 1), diagonal_form(A, lower=1, upper=1), b)
        return sol
