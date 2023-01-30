from .base import BaseTemporalHamiltonian
from ..nature_constants import h_bar


class Leapfrog(BaseTemporalHamiltonian):
    order = 2
    explicit = True
    stable = True # conditionally stable, dt is chosen accordingly
    name = "leapfrog"

    def iterate(self, t: float, dt_storage: float = None):

        # Override initial condition to preserve 2nd order accuracy
        # psi_0 = psi_half - c*H_half*psi_half
        # psi_1 = psi_half + c*H_half*psi_half
        # H_half = H_0 + Vt_half
        # c = dt / (2i*hbar)

        dt = self.dt
        H0 = self.H0
        Vt = self.Vt

        psi_half = self.psi_0
        Vt_half = self.Vt(dt/2)
        psi_0 = psi_half - dt / (2j*h_bar) * (self.H0 + Vt_half) @ psi_half
        psi_1 = psi_half + dt / (2j*h_bar) * (self.H0 + Vt_half) @ psi_half

        steps = 0
        tn = 0
        while tn < t:
            steps += 1

            # psi^n+1 = psi^n-1 + 2*dt*F^n
            # F^n = 1/ihbar * H^n @ psi^n
            # H^n = H0 + V^n

            psi_2 = (H0 + Vt(tn)) @ (2*dt / (1j*h_bar) * psi_1) + psi_0

            psi_0, psi_1 = psi_1, psi_2

            tn += dt

            # store data every `dt_storage` seconds
            if tn // dt_storage > len(self.psi):
                self.psi.append(psi_1)
                self.t.append(tn)
                self.V.append(Vt(tn))
