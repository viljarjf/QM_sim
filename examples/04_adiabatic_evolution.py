import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from scipy.special import expit

from qm_sim.hamiltonian import Hamiltonian
from qm_sim.nature_constants import e_0, m_e

s = 100
N = (s, s)              # Discretisation point count
L = (8e-9, 8e-9)        # System size in meters
m = m_e                 # Mass of the particle, here chosen as electron mass
n = 4                   # The amount of eigenstates to find
steps = 100


H = Hamiltonian(N, L, m)

# Set potential. Here, a square well is used, with dV = 0.15 eV
X, Y = H.get_coordinate_arrays()

# Smoothed rectangular time-dependent potential, assymetric to avoid degeneracy 
def V(t):
    a = 0.15*e_0*expit(5*10**9*(X-2e-9 + 1e-11*t)) + 0.15*e_0*expit(-5*10**9*(X+2e-9-1e-11*t))
    b = 0.15*e_0*expit(5*10**9*(Y-2.5e-9 - 1e-11*t)) + 0.15*e_0*expit(-5*10**9*(Y+2.5e-9+1e-11*t))
    return np.where(a+b>0.15*e_0,0.15*e_0,a+b)

H.V = V

# Solve the system
energies, states = H.eigen(n)
# Choose energy to follow
E = energies[1]

# Evolve adiabatically
Energies, psi = H.adiabatic_evolution(E_n=E, t0=0, dt=1, steps=steps)

# Create animation
fig, (ax, ax2) = plt.subplots(1,2)
im = ax.matshow(psi[:,:,0]**2)
im2 = ax2.matshow(V(0).T)
ims = [im,im2]

def init():
    im.set_data(psi[:,:,0]**2)
    im2.set_data(V(0).T)
    return im

def update(j):
    ims[0].set_data(psi[:,:,j]**2)
    ims[1].set_data(V(j).T)
    return ims

anim = FuncAnimation(fig, func=update, init_func=init, frames=steps, interval=50, blit=False, repeat = True, repeat_delay = 1000)
plt.show()
