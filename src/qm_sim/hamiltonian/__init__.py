"""
Different Hamiltonian classes.

Available options:

- Hamiltonian (proxy of SpatialHamiltonian)
- SpatialHamiltonian

"""

# The spatial Hamiltonian is the default
from .spatial_hamiltonian import SpatialHamiltonian
from .spatial_hamiltonian import SpatialHamiltonian as Hamiltonian
