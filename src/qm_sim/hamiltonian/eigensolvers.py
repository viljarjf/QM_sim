from typing import Callable
import numpy as np
from scipy import sparse as sp

# Dict to lookup the solvers
__SOLVERS = {}

import scipy_eigen
__SOLVERS["scipy"] = scipy_eigen.get_eigen

# Import guard the pytorch backend since it is optional
try:
    import pytorch_eigen
    __SOLVERS["pytorch"] = pytorch_eigen.get_eigen
    __SOLVERS["torch"] = pytorch_eigen.get_eigen
except ImportError:
    # Maybe display a warning?
    pass

# Function signature of the solver
Eigensolver = Callable[[sp.spmatrix, int, tuple[int]], tuple[np.ndarray, np.ndarray]]

def get_eigensolver(solver: str) -> Eigensolver | None:
    return __SOLVERS.get(solver)
