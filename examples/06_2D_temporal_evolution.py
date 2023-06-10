import numpy as np

from qm_sim.hamiltonian import Hamiltonian
from qm_sim.nature_constants import e_0, m_e
from scipy.signal import convolve2d

N = (100, 100)          # Discretisation point count
L = (15e-9, 15e-9)      # System size in meters
m = m_e                 # Mass of the particle, here chosen as electron mass
t_end = 100e-15         # Simulation time
dt = 1e-16              # Simulation data storage stepsize

# Initialize the hamiltonian.
H = Hamiltonian(N, L, m, temporal_scheme="scipy-Runge-Kutta 3(2)")

# Set the potential. Use a cool elipse cross that rotates over time
a = 2e-9
b = 5e-9
X, Y = H.get_coordinate_arrays()
def V(theta: float) -> np.ndarray:
    ct, st = np.cos(theta), np.sin(theta)

    V = np.ones(N) * e_0
    V[((X*ct + Y*st) / a)**2 + ((X*st - Y*ct) / b)**2 <= 1] = 0
    V[((X*ct + Y*st) / b)**2 + ((X*st - Y*ct) / a)**2 <= 1] = 0

    # Smooth out transition 
    n = 5
    kernel = np.ones((n,n)) / n**2
    V = convolve2d(V, kernel, mode="same", boundary="wrap")
    return V
    
H.V = lambda t: V(1e14*t)

# Set initial condition to a
# superposition of third and fourth eigenstate
_, psi = H.eigen(4)
psi_0 = 1/2**0.5 * (psi[2] + psi[3])

# Plot
H.plot_potential()
H.plot_eigen(4)
H.plot_temporal(t_end, dt, psi_0)
