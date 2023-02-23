from tqdm import tqdm
import numpy as np

from .base import BaseTemporalSolver
from ...nature_constants import h_bar


class Leapfrog(BaseTemporalSolver):
    order = 2
    explicit = True
    stable = True # conditionally stable, dt is chosen accordingly
    name = "leapfrog"

    def iterate(self, t_final: float, dt_storage: float = None) -> tuple[np.ndarray, np.ndarray]:

        # Override initial condition to preserve 2nd order accuracy
        # psi_0 = psi_half - c*H_half*psi_half
        # psi_1 = psi_half + c*H_half*psi_half
        # c = dt / (2i*hbar)

        dt = self.dt
        H = self.H

        psi_half = self.v_0.flatten()
        psi_0 = psi_half - dt / (2j*h_bar) * (H(dt/2) @ psi_half)
        psi_1 = psi_half + dt / (2j*h_bar) * (H(dt/2) @ psi_half)

        steps = 0
        tn = dt/2

        psi = [psi_1.reshape(self.H.shape)]
        t = [tn]
        with self.tqdm(t_final) as pbar:
            while tn < t_final:
                steps += 1

                # psi^n+1 = psi^n-1 + 2*dt*F^n
                # F^n = 1/ihbar * H^n @ psi^n
                # H^n = H0 + V^n

                psi_2 = 2*dt / (1j*h_bar) * (H(tn) @ psi_1) + psi_0
                psi_0, psi_1 = psi_1, psi_2

                pbar.update(dt)
                tn += dt

                # store data every `dt_storage` seconds
                if tn // dt_storage > len(psi):
                    psi.append(psi_0.reshape(self.H.shape))
                    t.append(tn)
        return np.array(t), np.array(psi)
