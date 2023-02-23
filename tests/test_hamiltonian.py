
import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

from qm_sim.hamiltonian import Hamiltonian
from qm_sim import nature_constants as const


def test_constant_mass():
    H = Hamiltonian((10,), (1,), 1, spatial_scheme="nine-point")
    plt.figure()
    plt.title("Constant mass hamiltonian")
    plt.imshow(H.asarray())
    plt.show()

def test_constant_mass_2D():
    H = Hamiltonian((10,30), (1,20), 1e-65, spatial_scheme="seven-point")
    plt.figure()
    plt.title("Constant mass hamiltonian")
    H = H.asarray()
    H[H != 0] = np.log(abs(H[H != 0]))
    plt.imshow(H)
    plt.show()

def test_array_mass():
    N = (10,)
    H = Hamiltonian(N, (1, ), 1 + np.arange(np.prod(N)))
    plt.figure()
    plt.title("Varying mass hamiltonian")
    plt.imshow(H.asarray())
    plt.show()

def test_potential():
    N = (100,)
    L = (2e-9,)

    r = np.ones(N)
    m = const.m_e * r
    n = 5

    H0 = Hamiltonian(N, L, m, spatial_scheme="three-point")
    z = np.linspace(-L[0]/2, L[0]/2, N[0])
    V = 100*z**2
    H0.set_potential(V)

    E0, psi0 = H0.eigen(n)

    plt.figure()
    plt.title("psi^2")
    for i in range(n):
        plt.subplot(2, 3, i+1)
        plt.title(f"E{i} = {E0[i] / const.e_0 :.3f} eV")
        plt.yticks([])
        plt.plot(psi0[i].real)
    plt.subplot(2, 3, 6)
    plt.plot(V)
    plt.title("Potential")
    plt.tight_layout()
    plt.show()

def test_eigen():
    N = (100,)
    L = (2e-9,)

    r = np.ones(N)
    m = const.m_e * r
    n = 4

    H0 = Hamiltonian(N, L, m, spatial_scheme="three-point")
    E0, psi0 = H0.eigen(n)

    H1 = Hamiltonian(N, L, m, spatial_scheme="five-point")
    E1, psi1 = H1.eigen(n)

    plt.figure()
    plt.title("psi^2")
    for i in range(n):
        plt.subplot(2, 2, i+1)
        plt.title(f"dE{i} = {E1[i] - E0[i]}")
        plt.plot(abs(psi0[i])**2)
        plt.plot(abs(psi1[i])**2)
    plt.show()

def test_temporal():
    N = (200,)
    L = (2e-9,)
    m = const.m_e
    t_end = 10e-15
    dt = 1e-17

    H = []
    schemes = ["leapfrog", "crank-nicolson", "scipy-Runge-Kutta 3(2)", "scipy-DOP853"]
    for scheme in schemes:
        H.append(Hamiltonian(N, L, m, temporal_scheme=scheme))

    z = np.linspace(-L[0]/2, L[0]/2, N[0])
    Vt = lambda t: 6*z**2 + 3*z*np.abs(z)*np.sin(4e15*t)
    for Hi in H:
        Hi.set_potential(Vt)

    _, _psi_0 = H[0].eigen(2)
    psi_0 = 2**0-5 * (_psi_0[0, :] + _psi_0[1, :])
    t = []
    psi = []
    for Hi in H:
        ti, psii = Hi.temporal_evolution(t_end, dt, psi_0)
        t.append(ti)
        psi.append((abs(psii)**2).real)

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

    V = np.array([Vt(tn) for tn in t[0]])

    psi_plots = []
    for psii in psi:
        ln_psi, = ax1.plot(z / 1e-9, psii[0, :])
        psi_plots.append(ln_psi)
    ln_V, = ax2.plot(z / 1e-9, V[0, :] / const.e_0)

    def init():
        ax1.set_title(f"$|\Psi|^2$")
        ax1.set_xlabel("z [nm]")
        ax1.legend(schemes)
        ax2.set_title("Potential [eV]")
        ax2.set_xlabel("z [nm]")
        ax1.set_ylim(0, np.max(psi[0]) * 1.1)
        ax2.set_ylim(np.min(V / const.e_0), np.max(V / const.e_0))
        fig.tight_layout()
        return *psi_plots, ln_V,

    def frames():
        for n in range(psi[0].shape[0]):
            yield *[psii[n, :] for psii in psi], V[n, :]

    def update(data):
        for i, psi_plot in enumerate(psi_plots):
            psi_plot.set_data(z / 1e-9, data[i])
        ln_V.set_data(z / 1e-9, data[-1] / const.e_0)
        return *psi_plots, ln_V,
    
    ani = FuncAnimation(fig, update, frames=frames, init_func=init, blit=False, interval=50)
    plt.show()
