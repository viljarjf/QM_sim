"""
Class to handle system discretization
"""
from typing import Iterable

from typing_extensions import Self


class Discretization:
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
