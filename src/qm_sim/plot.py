"""

Plots for the solutions to the Schrödinger equation

"""
from typing import Callable

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import ArtistAnimation

from .nature_constants import e_0


def _get_plot_fun(ndim: int, ax: plt.Axes = None):
    if ax is None:
        ax = plt

    if ndim == 1:
        return lambda *args, **kwargs: ax.plot(*args, c="b", **kwargs)
    elif ndim == 2:
        # To preserve the return type of :code:`plot`above
        return lambda *args, **kwargs: [ax.imshow(*args, **kwargs)]
    elif ndim == 3:
        raise NotImplementedError("3D plots not yet supported")
    else:
        # It would be impressive if this is ever executed
        raise ValueError(f"Invalid system dimensionality: {ndim}D")


def _shape_from_int(n: int) -> tuple[int, int]:
    """Get plot shape from n (plot count)"""
    if n < 1:
        raise ValueError("Plot count must be positive")
    shapes = {
        1: (1, 1),
        2: (1, 2),
        3: (1, 3),
        4: (2, 2),
        5: (2, 3),
        6: (2, 3),
        7: (2, 4),
        8: (2, 4),
        9: (3, 3),
        10: (3, 4),
        11: (3, 4),
        12: (3, 4),
    }
    return shapes.get(n, (1, n))


def eigen(E: np.ndarray, psi: np.ndarray):
    """Plot absolute square of wave function

    :param E: Eigenenergies
    :type E: np.ndarray
    :param psi: Eigenfunctions
    :type psi: np.ndarray
    """
    n = E.shape[0]
    ndim = len(psi.shape) - 1
    shape = _shape_from_int(n)

    plt.figure()
    plt.suptitle("$|\\Psi|^2$")
    plot = _get_plot_fun(ndim)
    for i in range(n):
        plt.subplot(*shape, i + 1)
        plt.title(f"E{i} = {E[i] / e_0 :.3f} eV")
        plot(abs(psi[i]) ** 2)
    plt.tight_layout()
    plt.show()
    return


def temporal(t: np.ndarray, psi: np.ndarray, Vt: Callable[[float], np.ndarray]):
    """Create an animation of the wave function, alongside the potential

    :param t: Times correspondind to the wave functions
    :type t: np.ndarray
    :param psi: Wave funcions
    :type psi: np.ndarray
    :param Vt: Function of time, returning the potential at that time
    :type Vt: Callable[[float], np.ndarray]
    """
    ndim = len(psi.shape) - 1

    # Get potential at each timestep
    V = np.array([Vt(tn) for tn in t])

    # We want to plot the probability distribution
    psi2 = np.abs(psi) ** 2

    # Plot the results
    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1_plot = _get_plot_fun(ndim, ax1)
    ax2_plot = _get_plot_fun(ndim, ax2)

    (psi_plot,) = ax1_plot(psi2[0, :])
    (V_plot,) = ax2_plot(V[0, :] / e_0)

    ax1.set_title(f"$|\\Psi|^2$")
    ax2.set_title("Potential [eV]")
    if ndim == 1:
        ax1.set_ylim(0, np.max(psi2) * 1.1)
        ax2.set_ylim(np.min(V / e_0), np.max(V / e_0))
    elif ndim == 2:
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax2.set_xticks([])
        ax2.set_yticks([])
    fig.tight_layout()

    ims = [(psi_plot, V_plot)]
    for n in range(psi2.shape[0]):
        (psi_plot,) = ax1_plot(psi2[n, ...], animated=True)
        (V_plot,) = ax2_plot(V[n, ...] / e_0, animated=True)
        ims.append(
            (
                psi_plot,
                V_plot,
            )
        )

    ani = ArtistAnimation(fig, ims, blit=True, interval=50)
    plt.show()
    return


def potential(V: np.ndarray):
    """Plot the potential

    :param V: Potential array
    :type V: np.ndarray
    """
    plt.figure()
    _get_plot_fun(len(V.shape))(V / e_0)
    plt.title("Potential [eV]")
    plt.show()
    return
