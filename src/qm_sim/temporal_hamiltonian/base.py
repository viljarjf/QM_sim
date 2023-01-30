from typing import Callable
from abc import ABC, abstractmethod

import numpy as np
from scipy import sparse as sp

from qm_sim.hamiltonian import Hamiltonian

from ..nature_constants import h_bar


class BaseTemporalHamiltonian(ABC):

    name: str
    order: int
    explicit: bool
    stable: bool

    def __init__(self, H0: Hamiltonian, Vt: Callable[[float], np.ndarray], 
        psi_0: np.ndarray = None, dt: float = None):
        """
        Create a temporal hamiltonian

        Args:
            H0 (Hamiltonian): 
                Stationary hamiltonian, initialised with the non-temporal potential
            Vt (Callable[[float], np.ndarray]): 
                Temporal potential. Should return a ndarray of shape  `H0.N`
            psi_0 (np.ndarray, optional): 
                Initial condition.
                If None, it is set from the stationary solutions of `H0`.
                Defaults to None
            dt (float, optional): 
                Time interval in the discretisation.
                If None, it is estimated based on von Neumann analysis.
                Defaults to None
        """

        self.H0 = H0
        self.Vt = Vt
        
        if dt is None:
            self.dt = self._get_dt()
        else:
            self.dt = dt
        
        if psi_0 is None:
            self.psi_0 = self._get_psi_0()
        else:
            self.psi_0 = psi_0

        # Store time, state, and potential for each iteration
        self.t = [0]
        self.psi = [self.psi_0]
        self.V = [self.Vt(0)]

    def _get_psi_0(self) -> np.ndarray:
        """Calculate a initial state from eigenstates of `H0`

        Returns:
            np.ndarray: initial wave function
        """

        # Default initial condition:
        # Equal amounts of the two lowest eigenstates
        _, _psi = self.H0.eigen(2)

        return (_psi[0, :] + _psi[1, :]) * 2**-0.5

        
    def _get_dt(self) -> float:
        """Calculate a decent `dt` for the given scheme,
        using von Neumann analysis.

        Returns:
            float: time delta
        """
        # Just use the leapfrog analysis to begin with
        # TODO perform the vN analysis for each scheme
        # (i.e. make this func abstract)
        V_max = max(self.Vt)
        V_min = min(self.Vt)

        E_max = max(
            abs(V_min),
            abs(V_max + 4 * h_bar**2 / (4*self.H0.m * sum(d**2 for d in self.H0.delta))),
            )
        return 0.25 * h_bar / E_max

    @abstractmethod
    def iterate(self, t: float, dt_storage: float = None):
        """
        Iterate the time propagation scheme until time=t.
        Store data each `dt` 
        (the calculations are performed with the previously initialised `dt`)

        Args:
            t (float): 
                End time for calculations
            dt_storage (float, optional): 
                Data storage period. 
                If None, store each calculation `dt`
                Defaults to None.
        """
        pass

    def get_psi(self) -> np.ndarray:
        return np.array(self.psi)
    
    def get_t(self) -> np.ndarray:
        return np.array(self.t)
    
    def get_Vt(self) -> np.ndarray:
        Vt = np.array(self.Vt)
        return self.H0.V0 + Vt
