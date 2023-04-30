"""

Use the scipy backend eigen solver to skip some overhead

"""
from scipy._lib._threadsafety import ReentrancyLock
from scipy.sparse.linalg._eigen.arpack.arpack import _SymmetricArpackParams
from scipy import sparse as sp
import numpy as np

def get_eigen(mat: sp.dia_matrix, n: int, shape: tuple[int], **kwargs):
    """Calculate `n` eigenvalues of `mat`. Reshape output to `shape`

    Args:
        mat (sp.dia_matrix): Matrix to calculate eigenvectors and -values
        n (int): Amount of eigenvectors and -values to find
        shape (tuple[int]): shape of eivenvectors

    Returns:
        np.ndarray: eigenvalues, shape (n,)
        np.ndarray: eigenvectors, shape (n, *`shape`)
    """
    if kwargs.get("sigma") is None:
        l, v = eigsh(mat._mul_vector, mat.shape[0], 
            mat.dtype, k=n, **kwargs)
    else:
        l, v = eigsh(mat, k=n, **kwargs)

    # Reshape into system shape.
    # Arrays returned from eigsh are fortran ordered
    v = np.array([v[:, i].reshape(shape, order="F") for i in range(n)])
    return l, v


def eigsh(A_OP, n, dtype, k=6, sigma=None, which='SA', v0=None,
          ncv=None, maxiter=None, tol=0, return_eigenvectors=True,
          ):
    mode = 1
    M_matvec = None
    Minv_matvec = None
    params = _SymmetricArpackParams(n, k, dtype.char, A_OP, mode,
                                    M_matvec, Minv_matvec, sigma,
                                    ncv, v0, maxiter, which, tol)

    with ReentrancyLock("Nested calls to eigs/eighs not allowed: "
        "ARPACK is not re-entrant"):
        while not params.converged:
            params.iterate()

        return params.extract(return_eigenvectors)    
