from typing import Callable

import numpy as np
from scipy.sparse import dia_matrix
from scipy.sparse.linalg import eigsh as scipy_eigsh
from tqdm import tqdm

from .. import nature_constants as const
from .. import plot
from ..spatial_derivative import get_scheme_order
from ..spatial_derivative.cartesian import nabla, laplacian
from ..temporal_solver import TemporalSolver, get_temporal_solver
from .eigsh import eigsh


class Hamiltonian:

    def __init__(self, N: tuple, L: tuple, m: float | np.ndarray, 
        spatial_scheme: str = "three-point", temporal_scheme: str = "leapfrog",
        verbose: bool = True):
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
            spatial_scheme (str, optional): 
                Finite difference scheme for spatial derivative. 
                Options are: 
                    - three-point
                    - five-point
                    - seven-point
                    - nine-point
                Defaults to "three-point".
            temporal_scheme (str, optional):
                Finite difference scheme for temporal derivative.
                Options are:
                    - crank-nicolson
                    - leapfrog
                Defaults to "leapfrog"
            verbose (bool):
                Option to display calculation and iteration info during runtime
                True by default.
        """
        if len(N) != len(L):
            raise ValueError("`N` and `L` must have same length")
        
        self.N = N
        self.L = L
        self._dim = len(N)
        self.delta = [Li / Ni for Li, Ni in zip(L, N)]

        order = get_scheme_order(spatial_scheme)
        if order is None:
            raise ValueError("Requested finite difference is invalid")

        # Handle non-isotropic effective mass
        if isinstance(m, np.ndarray):
            if m.shape != self.N:
                raise ValueError(f"Inconsistent shape of `m`: {m.shape}, should be {self.N}")
            m_inv = 1 / m.flatten()

            _n = nabla(N, L, order)
            _n2 = laplacian(N, L, order)

            # nabla m_inv nabla + m_inv nabla^2
            _n.data *= _n @ m_inv   # First term
            _n2.data *= m_inv       # Second term
            self.mat = _n + _n2
        else:
            print("Const")
            self.mat = laplacian(N, L, order)
            self.mat *= 1/m
        
        self._centerline_index = list(self.mat.offsets).index(0)
        self._default_data = None

        # Prefactor in hamiltonian.
        # Is either float or array, depending on `m`
        h0 = -const.h_bar**2 / 2
        
        # Multiplying the diagonal data directly is easier
        # if we have non-constant mass
        self.mat.data *= h0

        # static zero potential by default
        self._V = lambda t: np.zeros(shape = N)

        self.verbose = verbose

        self._temporal_solver = get_temporal_solver(temporal_scheme)
    
    @property
    def V(self) -> Callable[[float], np.ndarray]:
        """Potential as a function of time. 
        Output has same shape as system.

        Returns:
            Callable[[float], np.ndarray]: 
                input: time
                output: potential
        """
        return self._V

    @V.setter
    def V(self, V: np.ndarray | Callable[[float], np.ndarray]):
        """Set a (potentially time dependent) potential for the QM-system's Hamiltonian
        
        Args: 
            V (np.ndarray | Callable[[float], np.ndarray]):
                The energetic value of the potential at a given point associated with given array indices,
                if callable, the call variable will represent a changable parameter (usually time) with a 
                return type identical to the static case where V is an np.ndarray
        """
        if not callable(V):
            # Avoid the lambda func referring to itself
            array_V = V
            V = lambda t: array_V
        
        if V(0).shape != self.shape:
            raise ValueError(f"Inconsistent shape. Shape must be {self.shape}")

        self._V = V

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
        E, psi = self._get_eigen(n, t)

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
        
        Psi_t = np.empty(shape = (*self.N,steps+1))
        En_t = np.empty(steps+1)
        t = t0

        # initialize
        _, Psi_t[:, :, 0] = self._get_eigen(1, t, sigma=E_n)
        En_t[0] = E_n

        for i in tqdm(range(1, steps+1), desc="Adiabatic evolution", disable=not self.verbose):
            t += dt
            En_t[i], Psi_t[:,:,i] = self._get_eigen( 1, t,
                # smartly condition eigsolver to "hug" the single eigenvalue solution; eigenvector and eigenvalue should be
                # far closer to the previous one than any other if the adiabatic theorem is fulfilled
                sigma=En_t[i-1],
                v0=-Psi_t[:,:,i-1]
            )
        return En_t, Psi_t

    def temporal_evolution(self, t0: float, t_final: float, dt_storage: float = None, 
        psi_0: np.ndarray = None) -> tuple[np.ndarray, np.ndarray]:

        # Default: superposition of 1st and 2nd eigenstate
        if psi_0 is None:
            _, psi = self.eigen(2)
            psi_0 = 2**-0.5 * (psi[0] + psi[1])
        
        # Calculate a good dt from von Neumann analysis of the leapfrog scheme
        # NOTE: this assumes the temporal part is at most 4x the static part,
        #       and that the potential takes its maximum somwhere at t=t0
        V_max = np.max(self.V(t0) * 4)
        V_min = np.min(self.V(t0) * 4)

        E_max = max(
            abs(V_min),
            abs(V_max + 4 * np.max(self.mat.data)),
            )
        dt = 0.25 * const.h_bar / E_max

        # solve
        f = lambda t: 1/(1j*const.h_bar) * self.__call__(t)
        solver = self._temporal_solver(f, self.shape)
        t, psi = solver.iterate(psi_0.astype(np.complex128), t0, 
            t_final, dt, dt_storage, self.verbose)

        return t, psi
    temporal_evolution.__doc__ = TemporalSolver.iterate.__doc__
    
    def _get_eigen(self, n: int, t: float, **kwargs) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculate eigenvalues and eigenstates at time `t`.
        """
        if kwargs.get("sigma") is None:
            E, psi = eigsh(self._fast_matmul_op(t), self.N_total, 
                self.mat.dtype, which="SA", k=n, **kwargs)
        else:
            E, psi = scipy_eigsh(self(t), k=n, **kwargs)

        # Reshape into system shape.
        # Arrays returned from eigsh are fortran ordered
        psi = np.array([psi[:, i].reshape(self.N, order="F") for i in range(n)])
        return E, psi
    
    def plot_eigen(self, n: int, t: float = 0):
        """Calculate and plot n eigenstates at time t

        Args:
            n (int): Amount of eigenstates to find
            t (float, optional): Time at which to find eigenstates. Defaults to 0.
        """
        E, psi = self.eigen(n, t)
        plot.eigen(E, psi)

    def plot_temporal(self, t_final: float, dt: float, psi_0: np.ndarray = None, t0: float = 0):
        """Plot the temporal evolution of the eigenstates

        Args:
            t_final (float): Simulation end time
            dt (float): Simulation time between each frame
            psi_0 (np.ndarray, optional): Initial state. Defaults to None.
        """
        t, psi = self.temporal_evolution(t0, t_final, dt, psi_0)
        plot.temporal(t, psi, self.V)

    def plot_potential(self, t: float = 0):
        """Plot the potential at time t

        Args:
            t (float, optional): Time. Defaults to 0.
        """
        plot.potential(self.V(t))

    def __add__(self, other: np.ndarray) -> dia_matrix:
        if self._default_data is None:
            self._default_data = self.mat.data.copy()
    
        self.mat.data = self._default_data.copy()
        self.mat.data[self._centerline_index, :] += other.flatten()
        return self.mat
    
    def __call__(self, t: float) -> dia_matrix:
        return self + self.V(t)
    
    @property
    def shape(self):
        return self.N

    @property
    def N_total(self) -> int:
        i = 1
        for j in self.N:
            i *= j
        return i
    
    @property
    def ndim(self):
        return len(self.N)

    def __matmul__(self, other):
        return self.mat @ other
    
    def _fast_matmul_op(self, t: float = 0):
        mat = self(t)
        return mat._mul_vector
    
    def asarray(self) -> np.array:
        return self.mat.toarray()
    
   