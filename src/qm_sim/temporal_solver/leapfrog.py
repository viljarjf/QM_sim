import numpy as np

from .base import TemporalSolver


class Leapfrog(TemporalSolver):
    order = 2
    explicit = True
    stable = True  # conditionally stable, dt is chosen accordingly
    name = "leapfrog"

    def iterate(
        self,
        v_0: np.ndarray,
        t0: float,
        t_final: float,
        dt: float,
        dt_storage: float = None,
        verbose: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        # Override initial condition to preserve 2nd order accuracy
        # psi_0 = psi_half - c*H_half*psi_half
        # psi_1 = psi_half + c*H_half*psi_half
        # c = dt / (2i*hbar)

        if dt_storage is None:
            dt_storage = dt

        H = self.H

        psi_half = v_0.flatten()
        psi_0 = psi_half - dt / 2 * (H(t0 + dt / 2) @ psi_half)
        psi_1 = psi_half + dt / 2 * (H(t0 + dt / 2) @ psi_half)

        tn = t0 + dt / 2

        psi = [psi_1.reshape(self.output_shape)]
        t = [tn]
        with self.tqdm(t0, t_final, verbose) as pbar:
            while tn < t_final:
                # psi^n+1 = psi^n-1 + 2*dt*F^n
                # F^n = 1/ihbar * H^n @ psi^n
                # H^n = H0 + V^n

                psi_2 = 2 * dt * (H(tn) @ psi_1) + psi_0
                psi_0, psi_1 = psi_1, psi_2

                pbar.update(dt)
                tn += dt

                # store data every :code:`dt_storage`seconds
                if tn // dt_storage > len(psi):
                    psi.append(psi_0.reshape(self.output_shape))
                    t.append(tn)
        return np.array(t), np.array(psi)
