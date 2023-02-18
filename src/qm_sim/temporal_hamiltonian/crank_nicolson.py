import numpy as np
from scipy.sparse.linalg import spsolve
from tqdm import tqdm

from ..nature_constants import h_bar
from .base import BaseTemporalHamiltonian


class CrankNicolson(BaseTemporalHamiltonian):
    order = 2
    explicit = False
    stable = True
    name = "crank-nicolson"

    def iterate(self, t: float, dt_storage: float = None):

        dt = self.dt
        H0 = self.H0
        prefactor = dt/(2j * h_bar)
        H = prefactor * H0
        Vt = self.Vt
        tn = 0

        # Unity "matrix" (in a nice format for our )
        I = np.ones(self.H0.N).flatten()

        psi_n = self.psi_0
        pbar = tqdm(desc="Crank-Nicholson solver", total=t, disable=not self.H0.verbose)
        while tn < t:

            # psi^n+1 = psi^n-1 + dt/2*(F^n + F^n-1)
            # F^n = 1/ihbar * H^n @ psi^n
            # H^n = H0 + V^n

            Hn = prefactor * (H0 + Vt(tn))

            psi_n = spsolve(Hn - I, (H + I) @ psi_n)

            tn += dt
            H = Hn

            # store data every `dt_storage` seconds
            if tn // dt_storage > len(self.psi):
                self.psi.append(psi_n)
                self.t.append(tn)
                self.V.append(Vt(tn))
            pbar.update(tn)
