import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from qm_sim.hamiltonian import Hamiltonian
from qm_sim.nature_constants import e_0, m_e


N = (200,)              # Discretisation point count
L = (2e-9,)             # System size in meters
m = m_e                 # Mass of the particle, here chosen as electron mass
t_end = 10e-15          # Simulation time
dt = 1e-17              # Simulation data storage stepsize

# Initialize the hamiltonian
H = Hamiltonian(N, L, m, temporal_scheme="leapfrog")

# Set the potential to a quadratic potential oscilating from side to side
z = np.linspace(-L[0]/2, L[0]/2, N[0])
Vt = lambda t: 6*z**2 + 3*z*np.abs(z)*np.sin(4e15*t)
H.set_potential(Vt)

# Calculate the temporal evolution
t, psi = H.temporal_evolution(t_end, dt)

# Get potential at each timestep
V = np.array([H.get_V(tn) for tn in t])

# We want to plot the probability distribution
psi = abs(psi)**2

# Plot the results
fig, (ax1, ax2) = plt.subplots(2, 1)

psi_plot, = ax1.plot(z / 1e-9, psi[0, :])
V_plot, = ax2.plot(z / 1e-9, V[0, :] / e_0)

def init():
    ax1.set_title("$|\Psi|^2$, Leapfrog")
    ax1.set_xlabel("z [nm]")
    ax2.set_title("Potential [eV]")
    ax2.set_xlabel("z [nm]")
    ax1.set_ylim(0, np.max(psi) * 1.1)
    ax2.set_ylim(np.min(V / e_0), np.max(V / e_0))
    fig.tight_layout()

def frames():
    for n in range(psi.shape[0]):
        yield psi[n, :], V[n, :]

def update(data):
    psi_plot.set_data(z / 1e-9, data[0])
    V_plot.set_data(z / 1e-9, data[1] / e_0)

ani = FuncAnimation(fig, update, frames=frames, init_func=init, blit=False, interval=50)
plt.show()
