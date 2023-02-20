from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from . import Hamiltonian

from matplotlib import pyplot as plt
import numpy as np

from ..nature_constants import e_0

def _shape_from_int(n: int) -> tuple[int, int]:
    """Get plot shape from n (plot count)"""
    if n < 1:
        raise ValueError("Plot count must be positive")
    shapes = {
        1:  (1,1),
        2:  (1,2),
        3:  (1,3),
        4:  (2,2),
        5:  (2,3),
        6:  (2,3),
        7:  (2,4),
        8:  (2,4),
        9:  (3,3),
        10: (3,4),
        11: (3,4),
        12: (3,4),
    }
    return shapes.get(n, default=(1,n))

def plot_eigen(self: "Hamiltonian", n: int, t: float):
    """Plot eigenvectors of hamiltonian. 
    If not previously calculated, then calling this will calculate them

    Args:
        n (int): Amount of eigenstates to plot
    """
    shape = _shape_from_int(n)
    E, psi = self.eigen(n, t)
    from matplotlib import pyplot as plt
    plt.figure()
    plt.suptitle("$|\Psi|^2$")
    for i in range(n):
        plt.subplot(*shape, i+1)
        plt.title(f"E{i} = {E[i] / e_0 :.3f} eV")
        plt.yticks([])
        plt.plot(abs(psi[i])**2)
    plt.tight_layout()
    plt.show()