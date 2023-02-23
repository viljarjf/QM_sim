import numpy as np
from scipy.integrate import solve_ivp

from ...nature_constants import h_bar
from .base import BaseTemporalDerivative

class ScipySolver(BaseTemporalDerivative):
    method: str

    _skip_registration = True

    def iterate(self, t_final: float, dt_storage: float = None) -> tuple[np.ndarray, np.ndarray]:
        if self.H.verbose:
            print("Warning: verbose option not available for scipy temporal solvers")
        v_0 = self.v_0.astype(np.complex128).flatten()
        sol = solve_ivp(
            lambda t, y: (self.H(t) @ y) / (1j*h_bar), 
            [0, t_final], 
            v_0, 
            t_eval=np.arange(0, t_final, dt_storage),
            first_step=self.dt,
            # dense_output=True,
            method=self.method,
            )
        t = sol.t
        psi = [sol.y[:, i].reshape(*self.H.shape) for i in range(len(t))]

        return t, np.array(psi)
    
    def __init_subclass__(cls):
        cls.name = "scipy-" + cls.name
        cls._skip_registration = False
        return super().__init_subclass__()

class RungeKutta45(ScipySolver):
    order = 5
    explicit = True
    stable = True
    name = "Runge-Kutta 5(4)"
    method = "RK45"

class RungeKutta23(ScipySolver):
    order = 3
    explicit = True
    stable = True
    name = "Runge-Kutta 3(2)"
    method = "RK23"

class DOP853(ScipySolver):
    order = 8
    explicit = True
    stable = True
    name = "DOP853"
    method = "DOP853"

# Solver does not support complex numbers
# class Radau(ScipySolver):
#     order = 5
#     explicit = False
#     stable = True
#     name = "radau"
#     method = "Radau"

class BDF(ScipySolver):
    order = None
    explicit = False
    stable = True
    name = "backwards-differentiation"
    method = "BDF"

# Solver does not support complex numbers
# class LSODA(ScipySolver):
#     order = None
#     explicit = None
#     stable = None
#     name = "fortran LSODA"
#     method = "LSODA"
