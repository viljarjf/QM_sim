from typing import Callable

import numpy as np

from ..hamiltonian import Hamiltonian
from .base import BaseTemporalHamiltonian as TH
from .crank_nicolson import CrankNicolson
from .leapfrog import Leapfrog

_SCHEMES = [
    CrankNicolson,
    Leapfrog,
]
_SCHEME_NAMES = [s.name for s in _SCHEMES]

def temporal_hamiltonian(H0: Hamiltonian, Vt: Callable[[float], np.ndarray], 
        psi_0: np.ndarray = None, dt: float = None, scheme: str = "crank-nicolson") -> TH:
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
                    - cranck-nicholson
                    - leapfrog
                    - forward-euler
                    - backward-euler
                Defaults to cranc-nicholson
        """

        # raises ValueError if scheme is not found
        scheme_no = _SCHEME_NAMES.index(scheme)

        return _SCHEMES[scheme_no](H0, Vt, psi_0, dt)
