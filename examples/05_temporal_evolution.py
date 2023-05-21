import numpy as np

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
H.V = Vt

# Plot
H.plot_temporal(t_end, dt)
