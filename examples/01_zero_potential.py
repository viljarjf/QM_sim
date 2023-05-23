"""

Calculate eigenenergies and eigenstates of a 1D system with zero potential

"""
from qm_sim.hamiltonian import Hamiltonian
from qm_sim.nature_constants import m_e

N = 100         # Discretisation point count
L = 1e-9        # System size in meters
m = m_e         # Mass of the particle, here chosen as electron mass
n = 4           # The amount of eigenstates to find

# Set hamiltonian
H = Hamiltonian(N, L, m)

# Solve and plot the system
H.plot_eigen(n)
