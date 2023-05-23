"""
Module for different solvers for eigenproblems

Current options: 

- scipy 
- pytorch (optional dependency, must be installed)

"""

from typing import Callable

import numpy as np
from scipy import sparse as sp

# Dict to lookup the solvers
__SOLVERS = {}

from .scipy_eigen import scipy_get_eigen

__SOLVERS["scipy"] = scipy_get_eigen

# Import guard the pytorch backend since it is optional
try:
    from .pytorch_eigen import torch_get_eigen

    __SOLVERS["pytorch"] = torch_get_eigen
    __SOLVERS["torch"] = torch_get_eigen
except (ImportError, RuntimeError):
    # Maybe display a warning?
    pass

# Function signature of the solver
Eigensolver = Callable[[sp.spmatrix, int, tuple[int]], tuple[np.ndarray, np.ndarray]]


def get_eigensolver(solver: str) -> Eigensolver | None:
    return __SOLVERS.get(solver)
