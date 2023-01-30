from .base import BaseTemporalHamiltonian

import numpy as np

class Leapfrog(BaseTemporalHamiltonian):
    order = 2
    explicit = True
    stable = True # conditionally stable, dt is chosen accordingly
    name = "leapfrog"


    def _get_psi_0(self) -> np.ndarray:
        """Calculate a initial state from eigenstates of `H0`.
        This needs to be overridden to preserve the 2nd order accuracy

        Returns:
            np.ndarray: initial wave function
        """
        # psi_0 = psi_half - c*H_half*psi_half
        # psi_1 = psi_half + c*H_half*psi_half
        # H_half = H_0 + Vt_half
        # c = dt / (2i*hbar)

        psi_half = super()._get_psi_0()
        Vt_half = self.Vt(self.dt/2)
        
    
           
    def _get_dt(self) -> float:
        pass

    def iterate(self, t: float, dt_storage: float = None):
        pass
