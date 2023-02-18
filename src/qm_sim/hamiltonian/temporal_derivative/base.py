from abc import ABC, abstractmethod

import numpy as np

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..hamiltonian import Hamiltonian
from ...nature_constants import h_bar

class BaseTemporalDerivative(ABC):

    name: str
    order: int
    explicit: bool
    stable: bool

    def __init__(self, H: "Hamiltonian", v_0: np.ndarray = None, dt: float = None):
        """
        Initialise a temporal derivative

        Args:
            H (Callable[[float], LinearOperator]): 
                Function of time, returning a linear operator 
                representing the state function at that time.
            v_0 (np.ndarray, optional): 
                Initial condition.
                If None, it is set from the stationary solutions of `H` at time 0.
                Defaults to None
            dt (float, optional): 
                Time interval in the discretisation.
                If None, it is estimated based on von Neumann analysis.
                Defaults to None
        """

        self.H = H
        
        if dt is None:
            self.dt = self._get_dt()
        else:
            self.dt = dt
        
        if v_0 is None:
            v_0 = self._get_initial_condition()
        self.v_0 = v_0


    def _get_initial_condition(self) -> np.ndarray:
        """Calculate a initial state from eigenstates of `H`

        Returns:
            np.ndarray: initial wave function
        """
        # Default initial condition:
        # Equal superposition of the two lowest eigenstates
        _, _psi = self.H.eigen(2)
        return (_psi[0, :] + _psi[1, :]) * 2**-0.5

        
    def _get_dt(self) -> float:
        """Calculate a decent `dt` for the given scheme,
        using von Neumann analysis.

        Returns:
            float: time delta
        """
        # Just use the leapfrog analysis to begin with
        # TODO: perform the vN analysis for each scheme
        # (i.e. make this func abstract)
        # NOTE: this assumes the temporal part is at most 4x the static part
        V_max = np.max(self.H.get_V() * 4)
        V_min = np.min(self.H.get_V() * 4)

        E_max = max(
            abs(V_min),
            abs(V_max + 4 * h_bar**2 / (4*self.H.m * sum(d**2 for d in self.H.delta))),
            )
        return 0.25 * h_bar / E_max

    @abstractmethod
    def iterate(self, t_final: float, dt_storage: float = None) -> tuple[np.ndarray, np.ndarray]:
        """
        Iterate the time propagation scheme.
        Store the current state every `dt_storage`

        Args:
            t_final (float): 
                End time for calculations
            dt_storage (float, optional): 
                Data storage period. 
                If None, store each calculation `dt`
                Defaults to None.

        Returns:
            np.ndarray:
                Time values, shape (n,) for n storage times
            np.ndarray:
                State at times stored in the other output. shape (n, H.shape)
        """
        pass
