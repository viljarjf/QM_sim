from .base import BaseTemporalDerivative as TemporalDerivative
from .crank_nicolson import CrankNicolson
from .leapfrog import Leapfrog
from .RK45 import RungeKutta45

_SCHEMES = [
    CrankNicolson,
    Leapfrog,
    RungeKutta45
]
_SCHEME_NAMES = [s.name for s in _SCHEMES]

def get_temporal_solver(scheme: str) -> TemporalDerivative:
    if scheme in _SCHEME_NAMES:
        return _SCHEMES[_SCHEME_NAMES.index(scheme)]
    raise ValueError(f"Scheme {scheme} not found." 
        "Options are: \n{'\n'.join(_SCHEME_NAMES)}")
