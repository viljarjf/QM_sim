import numpy as np
from scipy.integrate import solve_ivp

from .base import TemporalSolver


class ScipySolver(TemporalSolver):
    """Base class for scipy's :code:`solve_ivp`-based solvers"""

    #: Name of the :code:`solve_ivp` method
    method: str

    _skip_registration = True

    def iterate(
        self,
        v_0: np.ndarray,
        t0: float,
        t_final: float,
        dt: float,
        dt_storage: float = None,
        verbose: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        pbar = self.tqdm(t0, t_final, verbose)

        # Precalculate the coefficient for negligible speedup

        def ode_fun(t, y):
            pbar.progress(t)
            return self.H(t) @ y

        sol = solve_ivp(
            ode_fun,
            [t0, t_final],
            v_0.flatten(),
            t_eval=np.arange(t0, t_final, dt_storage),
            first_step=dt,
            method=self.method,
        )
        t = sol.t
        psi = [sol.y[:, i].reshape(*self.output_shape) for i in range(len(t))]

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
