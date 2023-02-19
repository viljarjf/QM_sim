"""

Use the scipy backend eigen solver to skip some overhead

"""
from scipy._lib._threadsafety import ReentrancyLock
from scipy.sparse.linalg._eigen.arpack.arpack import _SymmetricArpackParams


def eigsh(A_OP, n, dtype, k=6, sigma=None, which='LA', v0=None,
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
