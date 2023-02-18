
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
    m = 9e-31 * r
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
    m = 9e-31 * r
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
    N = (100,)
    L = (2e-9,)

    m = 9e-31

    H1 = Hamiltonian(N, L, m, temporal_scheme="leapfrog")
    H2 = Hamiltonian(N, L, m, temporal_scheme="RK45")

    z = np.linspace(-L[0]/2, L[0]/2, N[0])
    Vt = lambda t: 6*z**2 + 3*z*np.abs(z)*np.sin(1e15*t)
    H1.set_potential(Vt)
    H2.set_potential(Vt)

    t1, psi1 = H1.temporal_evolution(50e-15, 0.1e-15)
    t2, psi2 = H2.temporal_evolution(50e-15, 0.1e-15)
    psi1 = abs(psi1)**2
    psi2 = abs(psi2)**2

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)

    V = np.array([Vt(tn) for tn in t1])

    ln_psi1, = ax1.plot(z / 1e-9, psi1[0, :])
    ln_psi2, = ax2.plot(z / 1e-9, psi2[0, :])
    ln_V, = ax3.plot(z / 1e-9, V[0, :] / const.e_0)

    def init():
        ax1.set_title("$|\Psi|^2$, Leapfrog")
        ax1.set_xlabel("z [nm]")
        ax2.set_title("$|\Psi|^2$, RK45")
        ax2.set_xlabel("z [nm]")
        ax3.set_title("Potential [eV]")
        ax3.set_xlabel("z [nm]")
        ax1.set_ylim(0, np.max(psi1) * 1.1)
        ax2.set_ylim(0, np.max(psi1) * 1.1) # intentionally use same psi to keep axis same
        ax3.set_ylim(np.min(V / const.e_0), np.max(V / const.e_0))
        fig.tight_layout()
        return ln_psi1, ln_psi2, ln_V,

    def frames():
        for n in range(psi1.shape[0]):
            yield psi1[n, :], psi2[n, :], V[n, :]

    def update(data):
        ln_psi1.set_data(z / 1e-9, data[0])
        ln_psi2.set_data(z / 1e-9, data[1])
        ln_V.set_data(z / 1e-9, data[2] / const.e_0)
        return ln_psi1, ln_psi2, ln_V,
    
    ani = FuncAnimation(fig, update, frames=frames, init_func=init, blit=False, interval=50)
    plt.show()
