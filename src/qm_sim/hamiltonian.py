
import numpy as np
from scipy import sparse as sp

from qm_sim import nature_constants as const
from qm_sim import finite_difference


class Hamiltonian:

    def __init__(self, N: tuple, L: tuple, m: float | np.ndarray, fd_scheme: str = "three-point"):
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
                    - "three-point"
                    - "five-point"
                    - "seven-point"
                    - "nine-point"
                Defaults to "three-point".
        """
        if len(N) != len(L):
            raise ValueError("`N` and `L` must have same length")
        
        self.N = N
        self.L = L
        self._dim = len(N)
        self.delta = [Li / Ni for Li, Ni in zip(L, N)]
        
        # index of the 0-offset.
        # Set in _get_fd_matrix
        self._centerline_index = None

        if isinstance(m, np.ndarray):
            if m.shape != self.N:
                raise ValueError(f"Inconsistent shape of `m`: {m.shape}, should be {self.N}")
            m = m.flatten()
        
        self.m = max(m)

        scheme_order = {
            "three-point": 2,
            "five-point": 4,
            "seven-point": 6,
            "nine-point": 8,
        }
        order = scheme_order.get(fd_scheme)
        if order is None:
            raise ValueError("Requested finite difference is invalid")
        self.mat = finite_difference.nabla_squared(N, L, order)
        self._centerline_index = list(self.mat.offsets).index(0)

        # Prefactor in hamiltonian.
        # Is either float or array, depending on `m`
        h0 = -const.h_bar**2 / (2 * m)
        
        # Multiplying the diagonal data directly is easier
        # if we have non-constant mass
        self.mat.data *= h0

        # No potential by default
        self.V0 = 0


    def set_static_potential(self, V0: np.ndarray):
        self.V0 = V0
        self.mat.data[self._centerline_index, :] += V0.flatten()
        self._default_data = self.mat.data.copy()


    def __add__(self, other: np.ndarray) -> sp.dia_matrix:
            self.mat.data = self._default_data.copy()
            self.mat.data[self._centerline_index, :] += other.flatten()
            return self.mat
    

    def __matmul__(self, other):
        return self.mat @ other
    
    
    def asarray(self) -> np.array:
        return self.mat.toarray()

    def eigen(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        """Calculate the n smallest eigenenergies and the corresponding eigenstates of the hamiltonian

        Args:
            n (int): 
                Amount of eigenenergies/states to output

        Returns:
            np.ndarray:
                Eigenenergies, shape (n,)
            np.ndarray:
                Normalised eigenstates, shape (n, *N) for a system with shape N
        """
        E, psi = sp.linalg.eigsh(self.mat, k=n, which="SA")
        psi = np.array([psi[:, i].reshape(self.N[::-1]).T for i in range(n)])

        # calculate normalisation factor
        nf = [psi[i, :]**2 for i in range(n)]
        for i, (L, N) in enumerate(zip(self.L, self.N)):
            dx = L / N
            for j in range(n):
                nf[j] = np.trapz(nf[j], dx=dx)
        # normalise
        for i in range(n):
            psi[i] /= nf[i]**0.5
        
        return E, psi
