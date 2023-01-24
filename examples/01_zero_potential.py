"""

Calculate eigenenergies and eigenstates of a 1D system with zero potential

"""
from qm_sim.hamiltonian import Hamiltonian
from qm_sim.nature_constants import m_e, e_0

N = (100,)      # Discretisation point count
L = (1e-9,)     # System size in meters
m = m_e         # Mass of the particle, here chosen as electron mass
n = 4           # The amount of eigenstates to find

# Set hamiltonian
H = Hamiltonian(N, L, m)

# Solve the system
energies, states = H.eigen(n)

# Plot results
from matplotlib import pyplot as plt
plt.figure()
plt.suptitle("$|\Psi|^2$")
for i in range(n):
    plt.subplot(2, 2, i+1)
    plt.title(f"E{i} = {energies[i] / e_0 :.3f} eV")
    plt.yticks([])
    plt.plot(abs(states[i])**2)
plt.tight_layout()
plt.show()
