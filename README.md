# QM_sim

Python library for simulation of quantum mechanical systems.

## Features 

- 1D and 2D systems
- Choice of finite difference scheme
- Stationary solutions

## Planned features
- 3D systems
- Temporal simulation
- Time-variant potentials
- Example plots

## Installation

`pip install qm-sim`

## Usage

### No potential
~~~python

from qm_sim.hamiltonian import Hamiltonian
from qm_sim.nature_constants import me

N = (1000,) # Discretisation point count
L = (1e-9,) # System size

H = Hamiltonian(N, L, me) # Use electron mass

energies, states = H.eigen(5)

~~~
`energies` is now a 5x1 array of eigenenergies, and `states` is a 5x1000 array of the corresponding eigenstates

### Quadratic potential
~~~python

from qm_sim.hamiltonian import Hamiltonian
from qm_sim.nature_constants import me, e0

import numpy as np

N = (1000,) # Discretisation point count
L = (2e-9,) # System size

H = Hamiltonian(N, L, me) # Use electron mass

V = np.linspace(-L[0]/2, L[0]/2, N[0])**2 * e0

H.set_static_potential(V)

energies, states = H.eigen(5)

~~~
`energies` is now a 5x1 array of eigenenergies, and `states` is a 5x1000 array of the corresponding eigenstates
