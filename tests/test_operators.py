from numpy import exp, sin, cos, linspace
from divi.operators import build_difference_opreator


def test_difference_operators():
    # build an axis with variable spacing
    n = 128
    dx = 0.1 * (1 + 0.5 * sin(linspace(0, 12, n)))
    x = dx.cumsum()

    # generate a test function and its analytic derivatives
    k = 0.2
    f = cos(x) * exp(-k * x)
    df = -sin(x) * exp(-k * x) - k * f
    d2f = k * sin(x) * exp(-k * x) - k * df - f

    # get the finite-difference matrices
    D1 = build_difference_opreator(axis=x, order=1, n_points=7)
    D2 = build_difference_opreator(axis=x, order=2, n_points=7)

    # find the maximum errors
    D1_error = abs(D1 @ f - df).max()
    D2_error = abs(D2 @ f - d2f).max()

    assert D1_error < 1e-4
    assert D2_error < 1e-4
