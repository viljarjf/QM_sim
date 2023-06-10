"""

Calculate eigenenergies and eigenstates of a 1D system with quadratic potential

"""
from qm_sim.hamiltonian import Hamiltonian
from qm_sim.nature_constants import m_e

import numpy as np

N = 1000        # Discretisation point count
L = 1e-9        # System size in meters
m = m_e         # Mass of the particle, here chosen as electron mass
n = 4           # The amount of eigenstates to find

# Set hamiltonian
H = Hamiltonian(N, L, m)

# Set potential
x, = H.get_coordinate_arrays()

def V(t):
    k = 200
    return 1/2*(k+t)*x**2
H.V = V

# Plot
H.plot_potential()
H.plot_eigen(n)
