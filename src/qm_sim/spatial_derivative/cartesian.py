from typing import Iterable

import numpy as np
from scipy import sparse as sp
from typing_extensions import Self


class CartesianDiscretization:
    """Helper class for discretization of cartesian space"""

    def __init__(self, L: float | tuple[float], N: int | tuple[int]):
        """Initialize a discretization objects. Allows non-tuple inputs

        :param L: System size along each dimension
        :type L: float | tuple[float]
        :param N: Discretization count along each dimension
        :type N: int | tuple[int]
        """
        # 1D inputs
        if isinstance(N, int):
            N = (N,)
        if isinstance(L, (float, int)):
            L = (L,)

        # Allow any iterable that can be converted to a tuple
        if isinstance(N, Iterable):
            N = tuple(N)
        if isinstance(L, Iterable):
            L = tuple(L)

        # Check type
        if not isinstance(N, tuple) or not all(isinstance(i, int) for i in N):
            raise ValueError(f"Param `N` must be int or tuple of ints, got {type(N)}")
        if not isinstance(L, tuple) or not all(isinstance(i, (float, int)) for i in L):
            raise ValueError(f"Param `L` must be float or tuple, got {type(L)}")

        if len(N) != len(L):
            raise ValueError("Inputs must have same length")

        self.N = N
        self.L = L
        self.dx = (Li / Ni for Li, Ni in zip(self.L, self.N))

    @classmethod
    def from_dx(cls, dx: float | tuple[float], N: int | tuple[int]) -> Self:
        """Initialize a discretization objects. Allows non-tuple inputs

        :param dx: Discretzation length along each dimension,
        i.e. distance between discretization points
        :type dx: float | tuple[float]
        :param N: Discretization count along each dimension
        :type N: int | tuple[int]
        """
        if isinstance(N, int):
            N = (N,)
        if isinstance(dx, (float, int)):
            dx = (dx,)

        # Allow any iterable that can be converted to a tuple
        if isinstance(N, Iterable):
            N = tuple(N)
        if isinstance(dx, Iterable):
            dx = tuple(dx)

        # Check type
        if not isinstance(N, tuple) or not all(isinstance(i, int) for i in N):
            raise ValueError(f"Param `N` must be int or tuple of ints, got {type(N)}")
        if not isinstance(dx, tuple) or not all(
            isinstance(i, (float, int)) for i in dx
        ):
            raise ValueError(f"Param `dx` must be float or tuple, got {type(dx)}")

        if len(N) != len(dx):
            raise ValueError("Inputs must have same length")

        L = (Ni * dxi for Ni, dxi in zip(N, dx))
        return cls(L, N)

    def get_coordinate_axes(self, centering: str = "middle") -> tuple[np.ndarray]:
        """Return an array of shape (N_i,) for the ith dimension, with corresponding coordinates

        :param centering: Where to place the origin.
            Options:

            - "middle" [default]: Center of system
            - "first": First element in the array

        :type centering: str, optional
        :return: Coordinate axes
        :rtype: tuple[np.ndarray]
        """
        if centering == "middle":
            return (np.linspace(-Li / 2, Li / 2, Ni) for Li, Ni in zip(self.L, self.N))
        elif centering == "first":
            return (
                np.linspace(0, Li, Ni, endpoint=False) for Li, Ni in zip(self.L, self.N)
            )
        else:
            raise ValueError("Invalid centering parameters")

    def get_coordinate_arrays(self, centering: str = "middle") -> tuple[np.ndarray]:
        """Return arrays with shape `N` of coordinates for each point in the system.

        :param centering: Where to place the origin.
            Options:

            - "middle" [default]: Center of system
            - "first": First element in the array

        :type centering: str, optional
        :return: Coordinate arrays
        :rtype: tuple[np.ndarray]
        """
        return np.meshgrid(*self.get_coordinate_axes(centering))


def nabla(
    discretization: CartesianDiscretization,
    order: int = 2,
    dtype: type = np.float64,
    boundary_condition: str = "zero",
) -> sp.dia_matrix:
    """Finite difference derivative in cartesian coordinates.
    Uses central stencil.

    Example: approximate the derivative of sin(x).

    >>> from qm_sim.spatial_derivative.cartesian import nabla, CartesianDiscretization
    >>> import numpy as np
    >>> disc = CartesianDiscretization(2*np.pi, 1000)
    >>> n = nabla( disc, boundary_condition="periodic")
    >>> x, = disc.get_coordinate_arrays( centering="first" )
    >>> y = np.sin(x)
    >>> np.allclose(n @ y, np.cos(x)) # The analytical solution is cos(x)
    True

    :param N: Discretization count along each axis
    :type N: tuple[int] | int
    :param L: System size along each axis
    :type L: tuple[float] | float
    :param order: Numerical order of the differentiation scheme.
        Options are:

            - 2
            - 4
            - 6
            - 8

        Defaults to 2.
    :type order: int, optional
    :param dtype: datatype of matrix.
        Defaults to np.float64
    :type dtype: type, optional
    :raises NotImplementedError: if requested order is not available
    :return: Discretized derivative matrix
    :rtype: sp.dia_matrix
    :param boundary_condition: Which boundary condition to apply.
            Options are:

            - zero
            - periodic

            Defaults to "zero"
    :type boundary_condition: str, optional
    """
    # Lookup for first half of stencil.
    # A 0 is appended for the central element,
    # and the rest is mirrored with a sign change later
    match order:
        case 2:
            stencil = [-1 / 2]
        case 4:
            stencil = [1 / 12, -2 / 3]
        case 6:
            stencil = [-1 / 60, 3 / 20, -3 / 4]
        case 8:
            stencil = [1 / 280, -4 / 105, 1 / 5, -4 / 5]
        case _:
            raise NotImplementedError(
                f"Finite difference scheme not found for {order = }"
            )
    stencil += [0]

    stencil = _mirror_sign_list(stencil)
    N = discretization.N
    L = discretization.L
    return _matrix_from_central_stencil(stencil, 1, N, L, dtype, boundary_condition)


def laplacian(
    discretization: CartesianDiscretization,
    order: int = 2,
    dtype: type = np.float64,
    boundary_condition: str = "zero",
) -> sp.dia_matrix:
    """Finite difference double derivative in cartesian coordinates.
    Uses central stencil

    Example: approximate the second derivative of sin(x).

    >>> from qm_sim.spatial_derivative.cartesian import laplacian, CartesianDiscretization
    >>> import numpy as np
    >>> disc = CartesianDiscretization(2*np.pi, 1000)
    >>> n = laplacian( disc, boundary_condition="periodic")
    >>> x, = disc.get_coordinate_arrays( centering="first" )
    >>> y = np.sin(x)
    >>> np.allclose(n @ y, -np.sin(x)) # The analytical solution is -sin(x)
    True

    :param N: Discretization count along each axis
    :type N: tuple[int] | int
    :param L: System size along each axis
    :type L: tuple[float] | float
    :param order: Numerical order of the differentiation scheme.
        Options are:

            - 2
            - 4
            - 6
            - 8

        Defaults to 2.
    :type order: int, optional
    :param dtype: datatype of matrix.
        Defaults to np.float64
    :type dtype: type, optional
    :raises NotImplementedError: if requested order is not available
    :return: Discretized derivative matrix
    :rtype: sp.dia_matrix
    :param boundary_condition: Which boundary condition to apply.
            Options are:

            - zero
            - periodic

            Defaults to "zero"
    :type boundary_condition: str, optional
    """
    # lookup the first half of the stencil, to be mirrored later
    match order:
        case 2:
            stencil = [1, -2]
        case 4:
            stencil = [-1 / 12, 4 / 3, -5 / 2]
        case 6:
            stencil = [1 / 90, -3 / 20, 3 / 2, -49 / 18]
        case 8:
            stencil = [-1 / 560, 8 / 315, -1 / 5, 8 / 5, -205 / 72]
        case _:
            raise NotImplementedError(
                f"Finite difference scheme not found for {order = }"
            )

    stencil = _mirror_list(stencil)
    N = discretization.N
    L = discretization.L
    return _matrix_from_central_stencil(stencil, 2, N, L, dtype, boundary_condition)


def _matrix_from_central_stencil(
    stencil: list[float],
    power: int,
    N: tuple[int],
    L: tuple[float],
    dtype: type,
    boundary_condition: str = "zero",
) -> sp.dia_matrix:
    """
    Creates a full matrix from a central stencil. Determines indices from stencil.
    """

    available_boundary_conditions = ["zero", "periodic"]
    if boundary_condition not in available_boundary_conditions:
        raise ValueError(
            f"Invalid boundary condition: {boundary_condition}. Options are: "
            + ", ".join(available_boundary_conditions)
        )

    if boundary_condition == "zero":
        indices = np.arange(len(stencil)) - len(stencil) // 2
        axis_indices = [indices for _ in N]

    elif boundary_condition == "periodic":
        indices = np.arange(len(stencil)) - len(stencil) // 2
        # There might be different sizes for each axis.
        # Therefore, we keep the updated indices for each axis
        periodic_indices = [list() for _ in N]
        periodic_stencil = []
        for i, ind in enumerate(indices):
            # Lower diagonals
            if ind < 0:
                for i, n in enumerate(N):
                    periodic_indices[i].append(-n - ind)
                periodic_stencil.append(stencil[i])
            # Upper diagonals
            elif ind > 0:
                for i, n in enumerate(N):
                    periodic_indices[i].append(n - ind)
                periodic_stencil.append(stencil[i])
            # Also keep the central diagonals
            else:
                for i in range(len(N)):
                    periodic_indices[i] += list(indices)
                periodic_stencil += stencil
        axis_indices = np.array(periodic_indices)
        stencil = np.array(periodic_stencil)

    mat = np.zeros((1, 1))
    for l, n, indices in zip(L, N, axis_indices):
        h = l / n
        next_mat = sp.diags(
            stencil,
            indices,
            shape=(n, n),
            dtype=dtype,
            format="dia",
        )
        next_mat *= 1 / h**power
        mat = sp.kronsum(mat, next_mat, format="dia")

    return mat


def _mirror_list(l: list) -> list:
    """
    Mirror a list around the final element.

    Example: [a, b, c] -> [a, b, c, b, a]
    """
    return l + l[-2::-1]


def _mirror_sign_list(l: list) -> list:
    """
    Mirror a list around the final element,
    and flip the sign of each new element.

    Example: [-a, b, c] -> [-a, b, c, -b, a]
    """
    return l + [-i for i in l[-2::-1]]
