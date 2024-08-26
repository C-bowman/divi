from collections.abc import Sequence
from numpy import ndarray, zeros


def triangle_cdf(x: ndarray, start: float, end: float) -> ndarray:
    """
    Returns the cumulative distribution function for the (symmetric) triangle
    distribution which spans the interval [start, end].
    """
    mid = 0.5 * (start + end)
    y = zeros(x.size)

    left = (x > start) & (x <= mid)
    right = (x > mid) & (x < end)

    y[left] = (x[left] - start) ** 2 / ((end - start) * (mid - start))
    y[right] = 1 - (x[right] - end) ** 2 / ((end - start) * (end - mid))
    y[x >= end] = 1.0
    return y


def triangle_ramp(x: ndarray, start: float, end: float) -> ndarray:
    """
    Returns the cumulative distribution function for the (symmetric) triangle
    distribution which spans the interval [start, end].
    """
    mid = 0.5 * (start + end)
    dx = end - start
    y = zeros(x.size)

    left = (x > start) & (x <= mid)
    right = (x > mid) & (x < end)
    after = x >= end

    y[left] = (x[left] - start) ** 3 / (dx**2 * 1.5)
    y[right] = x[right] - (x[right] - end) ** 3 / (dx**2 * 1.5) - mid
    y[after] = x[after] - mid
    return y


def smooth_ramp(
    x: ndarray, start: float, end: float, right_side=True
) -> ndarray:
    y = zeros(x.size)
    dx = end - start
    inside = (x > start) & (x <= end)
    if right_side:
        after = x > end
        y[inside] = 0.5 * (x[inside] - start)**2 / dx
        y[after] = (x[after] + (0.5 * dx - end))
    else:
        after = x < start
        y[inside] = -0.5 * (end - x[inside])**2 / dx
        y[after] = (x[after] + (0.5 * dx - end))
    return y


def smooth_barrier_edge(
    x: ndarray, start: float, end: float, gradient: float, right_side=True
) -> ndarray:
    ramp = smooth_ramp(x, start, end, right_side=right_side)
    cdf = triangle_cdf(x, start, end)
    sigmoid = cdf if right_side else 1 - cdf
    return gradient * ramp + sigmoid


def smooth_transport_profile(x: ndarray, params: Sequence[float]) -> ndarray:
    """
    A smooth transport profile model with a continuous first-derivative where
    the flat sections of the core, transport barrier and SOL are connected using
    the CDF of a triangle distribution.
    """
    y_core, y_sol, x_tb, w_tb, core_rise, sol_rise, core_grad, sol_grad = params
    y_tb = 1.0
    left_end = x_tb - 0.5 * w_tb
    left_start = left_end - core_rise
    left_barrier = smooth_barrier_edge(
        x, left_start, left_end, core_grad, right_side=False
    )

    right_start = x_tb + 0.5 * w_tb
    right_end = right_start + sol_rise
    right_barrier = smooth_barrier_edge(
        x, right_start, right_end, sol_grad, right_side=True
    )
    return left_barrier * (y_core - y_tb) + right_barrier * (y_sol - y_tb) + y_tb
