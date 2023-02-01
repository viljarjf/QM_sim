"""

Calculate eigenenergies and eigenstates of a 1D system with quadratic potential

"""
from qm_sim.hamiltonian import Hamiltonian
from qm_sim.nature_constants import m_e

import numpy as np

N = (1000,)     # Discretisation point count
L = (1e-9,)     # System size in meters
m = m_e         # Mass of the particle, here chosen as electron mass
n = 4           # The amount of eigenstates to find

# Set hamiltonian
H = Hamiltonian(N, L, m)

# Set potential
x = np.linspace(-L[0]/2, L[0]/2, N[0])

def V(t):
    k = 200
    return 1/2*(k+t)*x**2
H.set_potential(V)


# Solve the system
energies, states = H.eigen(n,t=1000)

# Plot results
from matplotlib import pyplot as plt
plt.figure()
plt.suptitle("$\Psi$")
for i in range(n):
    plt.subplot(2, 2, i+1)
    plt.yticks([])
    plt.plot(states[i].real)
plt.tight_layout()
plt.show()
