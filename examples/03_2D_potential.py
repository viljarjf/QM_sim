"""

Calculate and plot eigenstates in a 2D potential

"""
from qm_sim.hamiltonian import Hamiltonian
from qm_sim.nature_constants import m_e, e_0

import numpy as np

N = (200, 200)          # Discretisation point count
L = (8e-9, 8e-9)        # System size in meters
m = m_e                 # Mass of the particle, here chosen as electron mass
n = 4                   # The amount of eigenstates to find

# Set hamiltonian
H = Hamiltonian(N, L, m)

# Set potential. Here, a square well is used, with dV = 0.15 eV
V = 0.15*e_0 * np.ones(N)
V[
    N[0] // 2 - N[0] // 5 : N[0] // 2 + N[0] // 5, 
    N[1] // 2 - N[1] // 5 : N[1] // 2 + N[1] // 5
] = 0
H.V = V

# Plot potential
H.plot_potential()

# Plot eigenstate
H.plot_eigen(n)
