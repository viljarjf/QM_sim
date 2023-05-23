"""

Use the scipy backend eigen solver to skip some overhead

"""
import numpy as np
from scipy import sparse as sp
from scipy._lib._threadsafety import ReentrancyLock
from scipy.sparse.linalg import eigsh
from scipy.sparse.linalg._eigen.arpack.arpack import _SymmetricArpackParams


def scipy_get_eigen(
    mat: sp.dia_matrix, n: int, shape: tuple[int], **kwargs
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate :code:`n` eigenvalues of :code:`mat`. Reshape output to :code:`shape`

    :param mat: Matrix to calculate eigenvectors and -values for
    :type mat: sp.dia_matrix
    :param n: Amount of eigenvectors and -values to calculate
    :type n: int
    :param shape: Output shape for eigenvectors
    :type shape: tuple[int]
    :return: eigenvalues, eigenvectors
    :rtype: tuple[np.ndarray(shape = (:code:`n`)), np.ndarray(shape = (:code:`n`, :code:`shape`)]
    """
    if kwargs.get("sigma") is None:
        v, w = _eigsh(mat._mul_vector, mat.shape[0], mat.dtype, k=n, **kwargs)
    else:
        # Fallback to default solver
        v, w = eigsh(mat, k=n, which="SA", **kwargs)

    # Reshape into system shape.
    # Arrays returned from eigsh are fortran ordered
    w = np.array([w[:, i].reshape(shape, order="F") for i in range(n)])
    return v, w


def _eigsh(
    A_OP,
    n,
    dtype,
    k=6,
    sigma=None,
    which="SA",
    v0=None,
    ncv=None,
    maxiter=None,
    tol=0,
    return_eigenvectors=True,
):
    """Copied from the scipy sourcecode, removing some overhead.
    See https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.eigsh.html
    """
    mode = 1
    M_matvec = None
    Minv_matvec = None
    params = _SymmetricArpackParams(
        n,
        k,
        dtype.char,
        A_OP,
        mode,
        M_matvec,
        Minv_matvec,
        sigma,
        ncv,
        v0,
        maxiter,
        which,
        tol,
    )

    with ReentrancyLock(
        "Nested calls to eigs/eighs not allowed: ARPACK is not re-entrant"
    ):
        while not params.converged:
            params.iterate()

        return params.extract(return_eigenvectors)
