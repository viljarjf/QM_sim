import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.sparse import dia_matrix
from tqdm import tqdm

from ...nature_constants import h_bar
from .base import BaseTemporalDerivative


def I_plus_cH(a: complex, H: dia_matrix) -> dia_matrix:
    out = H.copy()
    out.data *= a
    zero_ind = list(out.offsets).index(0)
    out.data[zero_ind] += 1
    return out

class CrankNicolson(BaseTemporalDerivative):
    order = 2
    explicit = False
    stable = True
    name = "crank-nicolson"

    def iterate(self, t: float, dt_storage: float = None):

        dt = self.dt
        H = self.H
        prefactor = dt/(2j * h_bar)
        Vt = self.Vt
        tn = 0

        I = np.ones(H.N).flatten()

        Hn = prefactor*H(tn)
        psi_n = self.psi_0

        pbar = tqdm(desc="Crank-Nicholson solver", total=t, disable=not self.H0.verbose)
        while tn < t:

            # psi^n+1 = psi^n + dt/2*(F^n+1 + F^n)
            # F^n = 1/ihbar * H^n @ psi^n
            # => psi^n+1 = psi^n + dt/2ihbar*(H^n+1 @ psi^n+1 + H^n @ psi^n)
            # (I - dt/2ihbar*(H^n+1) @ psi^n+1 = (I + dt/2ihbar*H^n) @ psi^n


            psi_n = spsolve(
                I_plus_cH(-prefactor, H(t + dt)), 
                I_plus_cH(prefactor, H(t)) @ psi_n,
                )

            tn += dt
            H = Hn

            # store data every `dt_storage` seconds
            if tn // dt_storage > len(self.psi):
                self.psi.append(psi_n)
                self.t.append(tn)
                self.V.append(Vt(tn))
            pbar.update(tn)
