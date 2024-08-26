from numpy import ndarray, zeros, diag


def diagonal_form(A: ndarray, lower: int = 1, upper: int = 1) -> ndarray:
    """
    Converts a square matrix to diagonal-ordered form for use with
    banded-matrix solvers, e.g. ``scipy.linalg.solve_banded``.

    :param A: \
        A square matrix to be converted to diagonal-ordered form.

    :param lower:
        The number of lower diagonals to include.

    :param upper: \
        The number of upper diagonals to include.

    :return: \
        A copy of the ``A`` matrix in diagonal-ordered form as a ``numpy.ndarray``.
    """
    assert A.ndim == 2
    assert A.shape[0] == A.shape[1]
    assert A.shape[0] > lower >= 0
    assert A.shape[0] > upper >= 0

    A_df = zeros([1 + upper + lower, A.shape[0]])
    A_df[upper, :] = diag(A, k=0)

    for u in range(upper):
        row = upper - u - 1
        A_df[row, u + 1:] = diag(A, k=u + 1)

    for l in range(lower):
        row = l + upper + 1
        A_df[row, :-l - 1] = diag(A, k=-l - 1)

    return A_df
