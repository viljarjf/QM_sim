from typing import Callable

import numpy as np
from scipy import sparse as sp

from qm_sim.hamiltonian import Hamiltonian

class TemporalHamiltonian:

    def __init__(self, H0: Hamiltonian, Vt: Callable[[float], np.ndarray], 
        psi_0: np.ndarray = None, dt: float = None, scheme: str = "cranck-nicholson"):
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
            scheme (str, optional):
                Time differentiation scheme.
                Options are:
                    1. cranck-nicholson
                    2. leapfrog
                    3. forward-euler
                Defaults to cranc-nicholson
        """

        self.H0 = H0
        self.Vt = Vt
        self.scheme = scheme
        
        if psi_0 is None:
            self.psi_0 = self._get_psi_0(self.scheme)
        else:
            self.psi_0 = psi_0
        
        if dt is None:
            self.dt = self._get_dt(self.scheme)
        else:
            self.dt = dt

        # Store time, state, and potential for each iteration
        self.t = [0]
        self.psi = [self.psi_0]
        self.V = [self.Vt(0)]
        
    def _get_psi_0(self, scheme: str) -> np.ndarray:
        """Calculate a initial state from eigenstates of `H0`

        Args:
            scheme (str): Differentiation scheme

        Returns:
            np.ndarray: initial wave function
        """
        pass

    def _get_dt(self, scheme: str) -> float:
        """Calculate a decent `dt` for the given scheme,
        using von Neumann analysis

        Args:
            scheme (str): Differentiation scheme

        Returns:
            float: time delta
        """
        pass

    def iterate(t: float, dt: float = None):
        """
        Iterate the time propagation scheme until time=t.
        Store data each `dt` 
        (the calculations are performed with the previously initialised `dt`)

        Args:
            t (float): 
                End time for calculations
            dt (float, optional): 
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
        return np.array(self.Vt)
