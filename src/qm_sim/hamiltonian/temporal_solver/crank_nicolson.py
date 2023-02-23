import numpy as np
from scipy.linalg import solve_banded
from scipy.sparse import dia_matrix
from tqdm import tqdm

from ...nature_constants import h_bar
from .base import BaseTemporalSolver


def I_plus_aH(a: complex, H: dia_matrix) -> dia_matrix:
    """Return I + aH, where `I` is the unit matrix and a is `a` constant"""
    out = H.copy().astype(np.complex128)
    out.data *= a
    zero_ind = list(out.offsets).index(0)
    out.data[zero_ind] += 1
    return out

class CrankNicolson(BaseTemporalSolver):
    order = 2
    explicit = False
    stable = True
    name = "crank-nicolson"

    def __init__(self, H: "Hamiltonian", v_0: np.ndarray = None, dt: float = None):
        if H.ndim != 1:
            raise ValueError("Crank-Nicolson solver only supports 1D systems")
        super().__init__(H, v_0, dt)

    def iterate(self, t_final: float, dt_storage: float = None):

        dt = self.dt
        H = self.H
        prefactor = dt/(2j * h_bar)

        tn = 0
        psi_n = self.v_0

        t = [tn]
        psi = [psi_n]

        with self.tqdm(t_final) as pbar:
            while tn < t_final:

                # psi^n+1 = psi^n + dt/2*(F^n+1 + F^n)
                # F^n = 1/ihbar * H^n @ psi^n
                # => psi^n+1 = psi^n + dt/2ihbar*(H^n+1 @ psi^n+1 + H^n @ psi^n)
                # (I - dt/2ihbar*(H^n+1) @ psi^n+1 = (I + dt/2ihbar*H^n) @ psi^n

                lhs = I_plus_aH(-prefactor, H(tn + dt))
                rhs = I_plus_aH(prefactor, H(tn)) @ psi_n
                
                # Reformulate the solve_banded function in terms of the dia_matrix class
                psi_n = solve_banded(
                    (-lhs.offsets[0], lhs.offsets[-1]),
                    lhs.data[::-1, :],
                    rhs,
                    )

                pbar.update(dt)
                tn += dt

                # store data every `dt_storage` seconds
                if tn // dt_storage > len(psi):
                    psi.append(psi_n)
                    t.append(tn)
        return np.array(t), np.array(psi)
