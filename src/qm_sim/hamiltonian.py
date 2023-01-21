
import numpy as np
from scipy import sparse as sp

from qm_sim import nature_constants as const


class Hamiltonian(sp.dia_matrix):

    def __init__(self, N: tuple, L: tuple, m: float | np.ndarray, fd_scheme: str = "two-point"):
        """Discrete hamiltonian of a system

        Args:
            N (tuple): 
                Discretization count along each axis
            L (tuple): 
                System size along each axis
            m: (float | np.ndarray):
                Mass of the particle in the system. 
                Can be constant (float) or vary in the simulation area (array).
                If an array is used, `m.shape == N` must hold
            fd_scheme (str, optional): 
                Finite difference scheme. 
                Options are: 
                    - "two-point. 
                Defaults to "two-point".
        """
        if len(N) != len(L):
            raise ValueError("`N` and `L` must have same length")
        
        self.N = N
        self.L = L
        self._dim = len(N)
        
        # index of the 0-offset.
        # Set in _get_fd_matrix
        self._centerline_index = None

        if isinstance(m, np.ndarray):
            if m.shape != self.N:
                raise ValueError(f"Inconsistent shape of `m`: {m.shape}, should be {self.N}")
            m = m.flatten()

        # Prefactor in hamiltonian.
        # Is either float or array, depending on `m`
        h0 = -const.h_bar**2 / (2 * m)
        
        self.mat = self._get_fd_matrix(fd_scheme)

        # Multiplying the diagonal data directly is easier
        # if we have non-constant mass
        self.mat.data *= h0


    def _get_fd_matrix(self, fd_scheme) -> sp.dia_matrix:

        if fd_scheme == "two-point":
            dz = self.L[0] / self.N[0]
            mat = 1/dz ** 2 * sp.diags(
                [1, -2, 1], # values
                [-1, 0, 1], # offsets
                shape=(self.N[0], self.N[0]), 
                dtype=np.float64, 
                format="dia"
                )

            self._centerline_index = 1

        if self._dim == 2:
            # need to iterate the scheme once again
            # TODO
            pass
        return mat


    def set_static_potential(self, V0: np.ndarray):
        self.mat.data[self._centerline_index, :] += V0.flatten()
        self._default_data = self.mat.data.copy()
    

    def add(self, potential: np.ndarray) -> sp.dia_matrix:
            self.mat.data = self._default_data.copy()
            self.mat.data[self._centerline_index, :] += potential.flatten()
            return self.mat
    
    def asarray(self) -> np.array:
        return self.mat.toarray()
