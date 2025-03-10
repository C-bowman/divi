from numpy import ndarray, searchsorted, diff, zeros
from divi.models import smooth_transport_profile
from divi.solver import DiffusionSolver


def linear_interpolation_coeffs(
    axis: ndarray, points: ndarray
) -> tuple[ndarray, ndarray]:
    inds = zeros([points.size, 2], dtype=int)
    weights = zeros([points.size, 2])
    inds[:, 0] = searchsorted(axis, points) - 1
    inds[:, 1] = inds[:, 0] + 1

    a = axis[inds[:, 0]]
    b = axis[inds[:, 1]]
    weights[:, 1] = (points - a) / (b - a)
    weights[:, 0] = 1 - weights[:, 1]
    return inds, weights


class ProfilePrediction:
    def __init__(self, prediction_radius: ndarray, solver_radius: ndarray):

        assert (diff(solver_radius) > 0.0).all()
        assert solver_radius.min() <= prediction_radius.min()
        assert solver_radius.max() >= prediction_radius.max()

        self.radius = prediction_radius
        self.solver = DiffusionSolver(radius=solver_radius)
        self.delta_R = 1e-5
        self.interp_inds, self.interp_weights = linear_interpolation_coeffs(
            solver_radius, prediction_radius
        )

    def __call__(self, theta: ndarray) -> ndarray:
        transp_pars = theta[:8]
        core_val, sol_val = theta[-2:]

        # could pre-compute a 2D radius array here and do one function call
        D = smooth_transport_profile(self.solver.radius, transp_pars)
        f1 = smooth_transport_profile(self.solver.radius - self.delta_R, transp_pars)
        f2 = smooth_transport_profile(self.solver.radius + self.delta_R, transp_pars)
        grad_D = (f2 - f1) * (0.5 / self.delta_R)

        profile = self.solver.solve(
            diffusivity=D,
            diffusivity_gradient=grad_D,
            core_value=core_val,
            sol_value=sol_val,
        )

        return (profile[self.interp_inds] * self.interp_weights).sum(axis=1)
