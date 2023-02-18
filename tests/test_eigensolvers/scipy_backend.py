import numba
import numpy as np

from typing import Callable

from scipy._lib._threadsafety import ReentrancyLock
from scipy.sparse.linalg._eigen.arpack.arpack import _SymmetricArpackParams, _UnsymmetricArpackParams


def eigsh(A_OP, n, dtype, k=6, sigma=None, which='LM', v0=None,
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

def eigs(A_OP, n, dtype, k=6, sigma=None, which='LM', v0=None,
          ncv=None, maxiter=None, tol=0, return_eigenvectors=True,
          ):
    mode = 1
    M_matvec = None
    Minv_matvec = None
    params = _UnsymmetricArpackParams(n, k, dtype.char, A_OP, mode,
                                    M_matvec, Minv_matvec, sigma,
                                    ncv, v0, maxiter, which, tol)

    with ReentrancyLock("Nested calls to eigs/eighs not allowed: "
        "ARPACK is not re-entrant"):
        while not params.converged:
            params.iterate()

        return params.extract(return_eigenvectors)

class TestHam:
    def __init__(self, N, L, stencil, m = 1.0):
        self.N = np.array(N)
        self.L = np.array(L)
        self.stencil = np.array(stencil)
        self.m = m
        self.ndim = len(N)
        self.V = np.zeros(self.n)
        self.h0 = np.array([-0.0380998212 / (m * (l/n)**2) for l, n in zip(L, N)])
    
    def set_potential(self, V: list | np.ndarray):
        if isinstance(V, list):
            self.V = np.array(V)
        else:
            self.V = V.copy()

    @property
    def n(self) -> int:
        res = 1
        for n in self.N:
            res *= n
        return res
    
    def matop(self) -> Callable[[np.ndarray], np.ndarray]:
        V = self.V
        ndim = self.ndim
        h0 = self.h0
        n = self.n
        stencil = self.stencil
        N = self.N

        @numba.njit(parallel=True)
        def scipy_numba_matop(x_in: np.ndarray) -> np.ndarray:

            y_out = V.copy()

            offset = 1
            
            for dim in range(ndim):

                #  1 / dx^2
                h0_dim = h0[dim]

                #  y_i = x_i (*) stencil
                #  stencil usually has more terms, so it can be e.g.
                #  y_i = stencil[1]*x_i-1 + stencil[0]*x_i + stencil[1]*x_i+1
                for i in numba.prange(n):
                    stencil_sum = x_in[i] * stencil[0]
                    for j, el in enumerate(stencil):
                        # enumerate(stencil, start=1) fails with jit
                        if j == 0:
                            continue
                        #  account for current dimension
                        j *= offset
                        #  range-check the x-vector
                        if ((i + j) < n):
                            stencil_sum += x_in[i + j] * el
                        
                        if ((i - j) >= 0):
                            stencil_sum += x_in[i - j] * el
                        
                    y_out[i] += h0_dim * stencil_sum

                offset *= N[dim]

            return y_out
        
        # compile
        scipy_numba_matop(np.zeros(self.N).flatten())
        
        return scipy_numba_matop

    
