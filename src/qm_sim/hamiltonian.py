
import numpy as np
from scipy import sparse as sp

from qm_sim import nature_constants as const


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

        # lookup the first half of the stencil, to be mirrored later
        if fd_scheme == "three-point":
            stencil = [1, -2]
        elif fd_scheme == "five-point":
            stencil = [-1/12, 4/3, -5/2]
        elif fd_scheme == "seven-point":
            stencil = [-1/12, 4/3, -5/2]
        elif fd_scheme == "nine-point":
            stencil = [1/90, -3/20, 3/2, -49/18]
        else:
            raise ValueError("Finite difference scheme not found")
        
        # set the indices
        # e.g. [-2, -1, 0, 1, 2]
        indices = list(-i for i in reversed(range(len(stencil))))
        indices += [-i for i in indices[-2::-1]]
        indices = np.array(indices)

        # mirror the stencil
        # i.e. [a, b, c] -> [a, b, c, b, a]
        stencil += stencil[-2::-1]


        mat = 1
        prev_N = 1
        for L, N in zip(self.L, self.N):
            dz = L / N
            next_mat = 1/dz**2 * sp.diags(
                stencil,
                indices * prev_N,
                shape=(N * prev_N, N * prev_N),
                dtype=np.float64,
                format="dia"
                )
            mat = next_mat + sp.kron(sp.eye(N), mat, format="dia")
            prev_N *= N
        
        self._centerline_index = list(mat.offsets).index(0)

        return mat


    def set_static_potential(self, V0: np.ndarray):
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
                Eigenstates, shape (n, *N) for a system with shape N
        """
        E, psi = sp.linalg.eigsh(self.mat, k=n, which="SA")
        psi = np.array([psi[:, i] for i in range(n)])
        return E, psi
