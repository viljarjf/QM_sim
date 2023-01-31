
import numpy as np
from scipy.sparse import dia_matrix
from scipy.sparse.linalg import eigsh
from typing import Callable
from tqdm import tqdm

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
        else:
            self.m = m
        
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
        self.V = np.zeros(shape = N)
        self.__V_timedep = False


    def set_potential(self, V: np.ndarray | Callable[[float], np.ndarray]):
        """Set a (potentially time dependent) potential for the QM-system's Hamiltonian
        
        Args: 
            V (np.ndarray | Callable[[float], np.ndarray]):
                The energetic value of the potential at a given point associated with given array indices,
                if callable, the call variable will represent a changable parameter (usually time) with a 
                return type identical to the static case where V is an np.ndarray
        """
        # Todo: Set up tests to check if V is valid
        if callable(V):
            self.__V_timedep = True
        else:
            self.__V_timedep = False
        self.V = V


    def get_V(self, t: float = 0) -> np.ndarray:
        if self.__V_timedep:
            return self.V(t)
        return self.V


    def adiabatic_evolution(self, E_n: float, t0: float, dt : float, steps: int) -> tuple[np.ndarray, np.ndarray]:
        """Adiabatically evolve an eigenstate with a slowly varying time-dependent potential with
        energy (close to) E_n using the Adiabatic approximation. 
        https://en.wikipedia.org/wiki/Adiabatic_theorem

        Note: This is only valid given that the adiabatic theorem holds, ie. no degeneracy and a gap between 
        eigenvalues. Current implementation assumes this holds and does not check if it does (yet?). 
        There is no mathematical guarantee (yet?) that the iterative solver will "hug" the correct eigenvector
        at every step, but it should be good if V varies smoothly enough and dt is small enough. 
        
        Args:
            E_n (float): 
                The Eigenvalue for you want to adiabatically evolve 
            t_0 (float): 
                Starting time for time-evolution
            dt (float): 
                The (small) time parameter increment used to update the eigenstate temporally
            steps (int): 
                Number of increments.
        
        returns:
            tuple[np.ndarray(shape = steps), np.ndarray(shape = (N, steps))]:
                (E(t), Psi(t)), Time evolution of the eigenstate.
        """
        if not self.__V_timedep:
            raise RuntimeError("Hamiltonian needs to be time-dependent for method to be meaingful.")
        
        Psi_t = np.empty(shape = (self.N[0],self.N[1],steps+1))
        En_t = np.empty(steps+1)
        
        for i in tqdm(range(steps+1)):
            H = self.mat.copy()
            H.data[self._centerline_index, :] += self.V(t0+i*dt).flatten()
            
            En_t[i], psi = eigsh(
                A=H,
                k=1,
                # smartly condition eigsolver to "hug" the single eigenvalue solution; eigenvector and eigenvalue should be
                # far closer to the previous one than any other if the adiabatic theorem is fulfilled
                sigma=En_t[i-1] if i != 0 else E_n,
                v0=Psi_t[:,:,i-1] if i != 0 else np.ones(shape = self.N).flatten()
            )
            Psi_t[:,:,i] = psi[:].reshape(self.N[::-1]).T
        return En_t, Psi_t


    def __add__(self, other: np.ndarray) -> dia_matrix:
            self.mat.data = self._default_data.copy()
            self.mat.data[self._centerline_index, :] += other.flatten()
            return self.mat
    

    def __matmul__(self, other):
        return self.mat @ other
    
    
    def asarray(self) -> np.array:
        return self.mat.toarray()


    def eigen(self, n: int, t: float = 0) -> tuple[np.ndarray, np.ndarray]:
        """Calculate the n smallest eigenenergies and the corresponding eigenstates of the hamiltonian

        Args:
            n (int): 
                Amount of eigenenergies/states to output
            t (float):
                (Optional) If the potential is time-dependent, solves the Time-independent Schrödinger eq. as if it was frozen at time t.
                Does nothing if potential is time-independent
        Returns:
            np.ndarray:
                Eigenenergies, shape (n,)
            np.ndarray:
                Normalised eigenstates, shape (n, *N) for a system with shape N
        """
        # set initial conditions
        if self.__V_timedep:
            Vt = self.V(t)
        else:
            Vt = self.V

        H = self.mat.copy()
        H.data[self._centerline_index, :] += Vt.flatten()
        self._default_data = H.copy()

        E, psi = eigsh(H, k=n, which="SA")
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
