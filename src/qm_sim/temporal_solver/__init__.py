"""

Classes to solve :math:`y' = f(y)`

"""


from .base import TemporalSolver, get_temporal_solver
# Tell python where to find subclasses
from .crank_nicolson import CrankNicolson
from .leapfrog import Leapfrog
from .scipy_solvers import BDF, DOP853, RungeKutta23, RungeKutta45
