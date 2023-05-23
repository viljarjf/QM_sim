from typing import Callable

import numpy as np
from scipy.linalg import solve_banded
from scipy.sparse import dia_matrix

from .base import TemporalSolver


def I_plus_aH(a: complex, H: dia_matrix) -> dia_matrix:
    """Return I + aH, where :code:`I`is the unit matrix and :code:`a` is constant"""
    out = H.copy().astype(np.complex128)
    out.data *= a
    zero_ind = list(out.offsets).index(0)
    out.data[zero_ind] += 1
    return out


class CrankNicolson(TemporalSolver):
    order = 2
    explicit = False
    stable = True
    name = "crank-nicolson"

    def __init__(
        self, H: Callable[[float], np.ndarray], output_shape: tuple[int] = None
    ):
        TemporalSolver.__init__(self, H, output_shape)
        if len(self.output_shape) != 1:
            raise ValueError("Crank-Nicolson solver only supports 1D systems")

    def iterate(
        self,
        v_0: np.ndarray,
        t0: float,
        t_final: float,
        dt: float,
        dt_storage: float = None,
        verbose: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        if dt_storage is None:
            dt_storage = dt

        H = self.H
        prefactor = dt / 2

        tn = t0
        psi_n = v_0

        t = [tn]
        psi = [psi_n]

        with self.tqdm(t0, t_final, verbose) as pbar:
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

                # store data every :code:`dt_storage`seconds
                if tn // dt_storage > len(psi):
                    psi.append(psi_n)
                    t.append(tn)
        return np.array(t), np.array(psi)
