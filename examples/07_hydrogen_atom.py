# NOTE: this example uses skimage, available with:
# `pip install scikit-image`
from skimage import measure

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from qm_sim.hamiltonian import Hamiltonian
from qm_sim.nature_constants import e_0, 𝜀_0, m_e, m_p

N = (60, 60, 60)            # Discretisation point count
L = (1e-9, 1e-9, 1e-9)      # System size in meters
m = m_e * m_p / (m_e + m_p) # Mass of the particle, here the reduced mass of the system

nx, ny = 2, 2               # Used for the plot
n = nx * ny                 # Number of orbitals to find

# Initialize the hamiltonian.
H = Hamiltonian(N, L, m)

# Define the potential: -e^2 / (4*pi*𝜀_0*r)
x, y, z = H.get_coordinate_arrays()
r = (x**2 + y**2 + z**2)**0.5
H.V = -e_0**2 / (4 * np.pi * 𝜀_0 * r)

# Find a couple eigenvalues
energies, orbitals = H.eigen(n)
orbitals = np.abs(orbitals**2)

# Plot a iso-surface of each orbital
plt.figure()
plots = []
for i in range(n):
    iso_val = np.mean(orbitals[i])
    vertices, faces, _, _ = measure.marching_cubes(orbitals[i], iso_val, spacing=L)
    ax = plt.subplot(ny, nx, i+1, projection='3d')
    plots.append(ax.plot_trisurf(vertices[:, 0], vertices[:,1], faces, vertices[:, 2],
                    cmap='Spectral', lw=1))
    plt.title(f"E = {energies[i] :.2e}")
plt.tight_layout()
plt.show()
