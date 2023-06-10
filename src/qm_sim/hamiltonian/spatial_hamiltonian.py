"""

Real-space discretized Hamiltonian class, with solving and plotting functionality

"""

from typing import Any, Callable

import numpy as np
from scipy.sparse import dia_matrix
from scipy.sparse.linalg import eigsh as adiabatic_eigsh
from tqdm import tqdm

from .. import plot
from ..eigensolvers import get_eigensolver
from ..nature_constants import h_bar
from ..spatial_derivative import get_scheme_order
from ..spatial_derivative.cartesian import (
    CartesianDiscretization,
    laplacian,
    nabla,
)
from ..temporal_solver import TemporalSolver, get_temporal_solver


class SpatialHamiltonian:
    def __init__(
        self,
        N: tuple[int] | int,
        L: tuple[float] | float,
        m: float | np.ndarray,
        spatial_scheme: str = "three-point",
        temporal_scheme: str = "leapfrog",
        eigensolver: str = "scipy",
        verbose: bool = True,
        boundary_condition: str = "zero",
    ):
        """Non-stationary Hamiltonian in real space.

        :param N: Discretization count along each axis
        :type N: tuple[int] | int
        :param L: System size along each axis
        :type L: tuple[float] | float
        :param m: Mass of the particle in the system.
            Can be constant (float) or vary in the simulation area (array).
            If an array is used, :code:`m.shape == N` must hold
        :type m: float | np.ndarray
        :param spatial_scheme: Finite difference scheme for spatial derivative.
            Options are:

            - three-point
            - five-point
            - seven-point
            - nine-point

            Defaults to "three-point"
        :type spatial_scheme: str, optional
        :param temporal_scheme: Finite difference scheme for temporal derivative.
            Options are:

            - crank-nicolson
            - leapfrog
            - scipy-Runge-Kutta 5(4)
            - scipy-Runge-Kutta 3(2)
            - scipy-DOP853
            - scipy-backwards-differentiation

            Defaults to "leapfrog"
        :type temporal_scheme: str, optional
        :param eigensolver: Choose which eigensolver backend to use.
            Options are:

            - scipy
            - torch (optional dependency, must be installed)

            Defaults to "scipy"
        :type eigensolver: str, optional
        :param verbose: Option to display calculation and iteration info during runtime.
            Defaults to True
        :type verbose: bool, optional
        :param boundary_condition: Which boundary condition to apply.
            Options are:

            - zero
            - periodic

            Defaults to "zero"
        :type boundary_condition: str, optional
        """
        # Creating this object performs the necessary type checking
        self.discretization = CartesianDiscretization(L, N)
        self.N = self.discretization.N
        self.L = self.discretization.L
        self.deltas = self.discretization.dx

        self.eigensolver = get_eigensolver(eigensolver)
        if self.eigensolver is None:
            raise ValueError(f"Eigensolver {eigensolver} not found")

        order = get_scheme_order(spatial_scheme)
        if order is None:
            raise ValueError("Requested finite difference is invalid")

        # Handle non-isotropic effective mass
        if isinstance(m, np.ndarray) and np.all(m == m.flat[0]):
            m = m.flatten()[0]
        if isinstance(m, np.ndarray):
            print("Warning: Continuity is NOT satisfied (yet) with non-isotropic mass")
            if m.shape != self.N:
                raise ValueError(
                    f"Inconsistent shape of `m`: {m.shape}, should be {self.N}"
                )
            m_inv = 1 / m.flatten()

            _n = nabla(
                self.discretization, order=order, boundary_condition=boundary_condition
            )
            _n2 = laplacian(
                self.discretization, order=order, boundary_condition=boundary_condition
            )

            # nabla m_inv nabla + m_inv nabla^2
            _n.data *= _n @ m_inv  # First term
            _n2.data *= m_inv  # Second term
            self.mat = _n + _n2
        else:
            self.mat = laplacian(
                self.discretization, order=order, boundary_condition=boundary_condition
            )
            self.mat *= 1 / m

        self._centerline_index = list(self.mat.offsets).index(0)
        self._default_data = None

        # Multiply the laplacian with the remaining factors
        self.mat.data *= -(h_bar**2) / 2

        # static zero potential by default
        self._V = lambda t: np.zeros(shape=N)

        self.verbose = verbose

        self._temporal_solver = get_temporal_solver(temporal_scheme)

    @property
    def V(self) -> Callable[[float], np.ndarray]:
        """Potential as a function of time

        :return: input time, output potential. Output must have same shape as systme
        :rtype: Callable[[float], np.ndarray]
        """
        return self._V

    @V.setter
    def V(self, V: np.ndarray | Callable[[float], np.ndarray]):
        """Set a (potentially time dependent) potential for the QM-system's Hamiltonian

        :param V: The energetic value of the potential at a given point associated with
            given array indices,
            if callable, the call variable will represent a changable parameter
            (usually time) with a return type identical to the static case where V is an np.ndarray
        :type V: np.ndarray | Callable[[float], np.ndarray]
        """
        if not callable(V):
            # Avoid the lambda func referring to itself
            array_V = V
            V = lambda t: array_V

        if V(0).shape != self.shape:
            raise ValueError(f"Inconsistent shape. Shape must be {self.shape}")

        self._V = V

    def eigen(self, n: int, t: float = 0, **kwargs) -> tuple[np.ndarray, np.ndarray]:
        """Calculate the n smallest eigenenergies and the corresponding eigenstates of
        the hamiltonian

        :param n: Amount of eigenenergies/states to output
        :type n: int
        :param t: If the potential is time-dependent, solves the Time-independent
            Schrödinger eq. as if it was frozen at time t.
            Does nothing if potential is time-independent.
            Defaults to 0
        :type t: float, optional
        :return:
            - Eigenenergies
            - Normalised eigenstates

        :rtype: tuple[np.ndarray(shape = (n)), np.ndarray(shape = (n, N))]
        """

        # The adiabatic solver uses some features of the eigensolver
        # not exposed by the `self.eigensolver` function
        if kwargs.pop("is_adiabatic", False):
            E, psi = adiabatic_eigsh(self(t), k=n, **kwargs)
            psi = np.array([psi[:, i].reshape(self.N, order="F") for i in range(n)])
        else:
            E, psi = self.eigensolver(self(t), n, self.N)

        # calculate normalisation factor
        normalisation_factor = [psi[i, :] ** 2 for i in range(n)]
        for i, (L, N) in enumerate(zip(self.L, self.N)):
            dx = L / N
            for j in range(n):
                normalisation_factor[j] = np.trapz(normalisation_factor[j], dx=dx)
        # normalise
        for i in range(n):
            psi[i] /= normalisation_factor[i] ** 0.5
        return E, psi

    def adiabatic_evolution(
        self, E_n: float, t0: float, dt: float, steps: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Adiabatically evolve an eigenstate with a slowly varying time-dependent potential with
        energy (close to) :code:`E_n` using the Adiabatic approximation.
        https://en.wikipedia.org/wiki/Adiabatic_theorem

        Note: This is only valid given that the adiabatic theorem holds, ie. no degeneracy and a
        gap betweeneigenvalues. Current implementation assumes this holds and does not check if
        it does (yet?).
        There is no mathematical guarantee (yet?) that the iterative solver will "hug" the
        correct eigenvector at every step, but it should be good if V varies smoothly enough and
        :code:`dt` is small enough.


        :param E_n: The Eigenvalue for you want to adiabatically evolve
        :type E_n: float
        :param t0: Starting time for time-evolution
        :type t0: float
        :param dt: The (small) time parameter increment used to update the eigenstate temporally
        :type dt: float
        :param steps: Number of increments.
        :type steps: int
        :return: (E(t), Psi(t)), Time evolution of the eigenstate.
        :rtype: tuple[np.ndarray(shape = steps), np.ndarray(shape = (N, steps))]
        """
        Psi_t = np.empty(shape=(*self.N, steps + 1))
        En_t = np.empty(steps + 1)
        t = t0

        # initialize
        _, Psi_t[:, :, 0] = self.eigen(1, t, sigma=E_n, is_adiabatic=True)
        En_t[0] = E_n

        for i in tqdm(
            range(1, steps + 1), desc="Adiabatic evolution", disable=not self.verbose
        ):
            t += dt
            En_t[i], Psi_t[:, :, i] = self.eigen(
                1,
                t,
                # smartly condition eigsolver to "hug" the single eigenvalue solution;
                # eigenvector and eigenvalue should be far closer to the previous one
                # than any other if the adiabatic theorem is fulfilled
                sigma=En_t[i - 1],
                v0=-Psi_t[:, :, i - 1],
                is_adiabatic=True,
            )
        return En_t, Psi_t

    def temporal_evolution(
        self,
        t0: float,
        t_final: float,
        dt_storage: float = None,
        psi_0: np.ndarray = None,
    ) -> tuple[np.ndarray, np.ndarray]:
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
        dt = 0.25 * h_bar / E_max

        # solve
        func = lambda t: 1 / (1j * h_bar) * self.__call__(t)
        solver = self._temporal_solver(func, self.shape)
        t, psi = solver.iterate(
            psi_0.astype(np.complex128), t0, t_final, dt, dt_storage, self.verbose
        )

        return t, psi

    temporal_evolution.__doc__ = TemporalSolver.iterate.__doc__

    def plot_eigen(self, n: int, t: float = 0):
        """Calculate and plot :code:`n` eigenstates at time :code:`t`

        :param n: Amount of eigenstates to plot
        :type n: int
        :param t: Time at which to find eigenstates, defaults to 0
        :type t: float, optional
        """
        E, psi = self.eigen(n, t)
        plot.eigen(E, psi)

    def plot_temporal(
        self, t_final: float, dt: float, psi_0: np.ndarray = None, t0: float = 0
    ):
        """Plot the temporal evolution of the eigenstates

        :param t_final: Simulation end time
        :type t_final: float
        :param dt: Simulation time between each timestep
        :type dt: float
        :param psi_0: Initial state, defaults to None
        :type psi_0: np.ndarray, optional
        :param t0: Simulation start time, defaults to 0
        :type t0: float, optional
        """
        t, psi = self.temporal_evolution(t0, t_final, dt, psi_0)
        plot.temporal(t, psi, self.V)

    def plot_potential(self, t: float = 0):
        """Plot the potential at a given time

        :param t: Time at which to plot, defaults to 0
        :type t: float, optional
        """
        plot.potential(self.V(t))

    def __add__(self, other: np.ndarray) -> dia_matrix:
        """Add a vector to the main diagonal, and return the matrix

        :param other: Array to add to the main diagonal
        :type other: np.ndarray
        :return: Matrix data, i.e. NOT a Hamiltonian object
        :rtype: dia_matrix
        """
        if self._default_data is None:
            self._default_data = self.mat.data.copy()

        self.mat.data = self._default_data.copy()
        self.mat.data[self._centerline_index, :] += other.flatten()
        return self.mat

    def __call__(self, t: float) -> dia_matrix:
        """Get the matrix at a given time

        :param t: Time at which to get the matrix at
        :type t: float
        :return: Matrix data, i.e. NOT a Hamiltonian object
        :rtype: dia_matrix
        """
        return self + self.V(t)

    @property
    def shape(self) -> tuple[int]:
        """System shape

        :return: Tuple with discretization count along each axis
        :rtype: tuple[int]
        """
        return self.N

    @property
    def N_total(self) -> int:
        """Total amount of discretization points

        :return: Discretization point count
        :rtype: int
        """
        i = 1
        for j in self.N:
            i *= j
        return i

    @property
    def ndim(self) -> int:
        """System dimensionallity

        :return: Amount of dimensions in the system
        :rtype: int
        """
        return len(self.N)

    def __matmul__(self, other: Any) -> Any:
        """Matrix multiplication with the underlying data

        :param other: matrix to multiply with
        :type other: Any
        :return: Matrix product
        :rtype: Any
        """
        return self.mat @ other

    def _fast_matmul_op(self, t: float = 0) -> Callable[[Any], np.ndarray]:
        """Evaluate the Hamiltonian at a given time,
        and return a matrix-vector multiplication function

        :param t: Time at which to evaluate the Hamiltonian, defaults to 0
        :type t: float, optional
        :return: Fast matrix-vector multiplication function
        :rtype: Callable[[Any], np.ndarray]
        """
        mat = self(t)
        return mat._mul_vector

    def asarray(self) -> np.ndarray:
        """Return the data matrix as a dense array

        :return: Dense array of the discretized Hamiltonian
        :rtype: np.ndarray
        """
        return self.mat.toarray()

    def get_coordinate_arrays(self) -> tuple[np.ndarray]:
        return self.discretization.get_coordinate_arrays()

    get_coordinate_arrays.__doc__ = (
        CartesianDiscretization.get_coordinate_arrays.__doc__
    )
