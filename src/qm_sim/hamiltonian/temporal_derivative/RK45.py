import numpy as np
from scipy.integrate import solve_ivp
from tqdm import tqdm

from ...nature_constants import h_bar
from .base import BaseTemporalDerivative


class RungeKutta45(BaseTemporalDerivative):
    order = 4
    explicit = True
    stable = True 
    name = "RK45"

    def iterate(self, t_final: float, dt_storage: float = None) -> tuple[np.ndarray, np.ndarray]:

        sol = solve_ivp(
            lambda t, y: (self.H(t) @ y) / (1j*h_bar), 
            [0, t_final], 
            self.v_0.astype(np.complex128), 
            t_eval=np.arange(0, t_final, dt_storage),
            first_step=self.dt,
            )
        print(sol.nfev)
        t = sol.t
        psi = [sol.y[:, i].reshape(*self.H.shape).real for i in range(len(t))]

        return t, np.array(psi)

