from numpy import array, arange, ndarray, zeros, ones, maximum, minimum, diff
from scipy.special import factorial
from scipy.linalg import solve


def get_fd_coeffs(points: ndarray, order: int = 1) -> ndarray:
    # check validity of inputs
    if type(points) is not ndarray:
        points = array(points)
    n = len(points)
    if n <= order:
        raise ValueError(
            "The order of the derivative must be less than the number of points"
        )
    # build the linear system
    b = zeros(n)
    b[order] = factorial(order)
    A = ones([n, n])
    for i in range(1, n):
        A[i, :] = points**i
    # return the solution
    return solve(A, b)


def build_difference_opreator(axis: ndarray, order: int, n_points: int) -> ndarray:
    """
    Constructs a finite-difference matrix operator which estimates derivatives of
    the field.

    :param axis: \
        The axis positions at which the derivative is estimated.

    :param order: \
        The order of the derivative to estimate.

    :param n_points: \
        The number of points used to estimate the derivative at each position along
        the axis.

    :return: \
        The matrix operator as a ``numpy.ndarray``/
    """
    assert (diff(axis) > 0.).all()
    assert n_points > order
    assert n_points % 2 == 1 and n_points > 2
    L = axis.size
    d = n_points // 2
    A = zeros([L, L])
    inds = arange(L)
    starts = minimum(maximum(inds - d, 0), L - n_points)
    stops = minimum(maximum(inds + d + 1, n_points), L)
    for i in range(L):
        slc = slice(starts[i], stops[i])
        positions = axis[slc] - axis[i]
        A[i, slc] = get_fd_coeffs(points=positions, order=order)
    return A

