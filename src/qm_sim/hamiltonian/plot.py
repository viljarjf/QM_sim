from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from . import Hamiltonian

from matplotlib import pyplot as plt
from matplotlib.animation import ArtistAnimation
import numpy as np

from ..nature_constants import e_0

def _get_plot_fun(ndim: int, ax: plt.Axes = None):
    if ax is None:
        ax = plt

    if ndim == 1:
        return lambda *args, **kwargs: ax.plot(*args, c="b", **kwargs)
    elif ndim == 2:
        # To preserve the return type of `plot` above
        return lambda *args, **kwargs: [ax.imshow(*args, **kwargs)]
    elif ndim == 3:
        raise NotImplementedError("3D plots not yet supported")
    else:
        # It would be impressive if this is ever executed
        raise ValueError(f"Invalid system dimensionality: {self.ndim}D")

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
    return shapes.get(n, (1,n))

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
    plot = _get_plot_fun(self.ndim)
    for i in range(n):
        plt.subplot(*shape, i+1)
        plt.title(f"E{i} = {E[i] / e_0 :.3f} eV")
        plot(abs(psi[i])**2)
    plt.tight_layout()
    plt.show()

def plot_temporal(self: "Hamiltonian", t_final: float, dt: float, psi_0: np.ndarray = None):
    """Animate the temporal evolution

    Args:
        t_final (float): 
            Final simulation time
        dt (float): 
            Time between each simulation frame.
            Note that the solver uses its own timestep.
    """
    # Calculate the temporal evolution
    t, psi = self.temporal_evolution(t_final, dt, psi_0)

    # Get potential at each timestep
    V = np.array([self.get_V(tn) for tn in t])

    # We want to plot the probability distribution
    psi = np.abs(psi)**2

    # Plot the results
    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1_plot = _get_plot_fun(self.ndim, ax1)
    ax2_plot = _get_plot_fun(self.ndim, ax2)

    psi_plot, = ax1_plot(psi[0, :])
    V_plot, = ax2_plot(V[0, :] / e_0)

    ax1.set_title(f"$|\Psi|^2$")
    ax2.set_title("Potential [eV]")
    if self.ndim == 1:
        ax1.set_ylim(0, np.max(psi) * 1.1)
        ax2.set_ylim(np.min(V / e_0), np.max(V / e_0))
    elif self.ndim == 2:
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax2.set_xticks([])
        ax2.set_yticks([])
    fig.tight_layout()

    ims = [(psi_plot, V_plot)]
    for n in range(psi.shape[0]):
        psi_plot, = ax1_plot(psi[n, ...], animated=True)
        V_plot, = ax2_plot(V[n, ...] / e_0, animated=True)
        ims.append((psi_plot, V_plot,))

    ani = ArtistAnimation(fig, ims, blit=True, interval=50)
    plt.show()

def plot_potential(self: "Hamiltonian", t: float = 0):
    """Plot the potential at a given time

    Args:
        t (float, optional): time at which to plot. Defaults to 0.
    """
    plt.figure()
    _get_plot_fun(self.ndim)(self.get_V(t) / e_0)
    plt.title("Potential [eV]")
    plt.show()
