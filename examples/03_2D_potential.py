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

H.set_potential(V)
print(V)
# Solve the system
energies, states = H.eigen(n)

# Plot results
from matplotlib import pyplot as plt
plt.figure()
plt.suptitle("$|\Psi|^2$")
for i in range(n):
    plt.subplot(2, 2, i+1)
    plt.imshow(abs(states[i])**2)
    plt.title(f"E$_{i}$ = {energies[i] / e_0 :.3f} eV")
    plt.axis("off")
plt.tight_layout()
plt.show()
