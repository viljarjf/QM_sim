from tqdm import tqdm
import numpy as np

from .base import BaseTemporalDerivative
from ...nature_constants import h_bar


class Leapfrog(BaseTemporalDerivative):
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

        psi_half = self.v_0
        psi_0 = psi_half - dt / (2j*h_bar) * (H(dt/2) @ psi_half)
        psi_1 = psi_half + dt / (2j*h_bar) * (H(dt/2) @ psi_half)

        steps = 0
        tn = 0

        psi = [psi_0]
        t = [tn]
        with tqdm(desc="Leapfrog solver", total=t_final, disable=not self.H.verbose) as pbar:
            pbar.bar_format = "{l_bar}{bar}| {n:#.02g}/{total:#.02g}"
            while tn < t_final:
                steps += 1

                # psi^n+1 = psi^n-1 + 2*dt*F^n
                # F^n = 1/ihbar * H^n @ psi^n
                # H^n = H0 + V^n

                psi_2 = H(tn) @ (2*dt / (1j*h_bar) * psi_1) + psi_0

                psi_0, psi_1 = psi_1, psi_2

                tn += dt
                pbar.update(dt)

                # store data every `dt_storage` seconds
                if tn // dt_storage > len(psi):
                    psi.append(psi_1)
                    t.append(tn)

        return np.array(t), np.array(psi)
