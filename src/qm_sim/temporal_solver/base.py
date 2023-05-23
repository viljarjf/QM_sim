from abc import ABC, abstractmethod
from typing import Callable

import numpy as np
from tqdm import tqdm

from ..nature_constants import h_bar

# Dict to store subclasses of TemporalSolver
_SCHEMES = {}


class TemporalSolver(ABC):
    """Class to solve :math:`y' = H(y)`"""

    #: Name of the solver
    name: str

    #: Integration order
    order: int

    #: Is the method explicit or implicit?
    explicit: bool

    #: Is the method stable?
    #: If only conditionally stable, this will be true
    #: and :code:`dt` will be forced into its stable range
    stable: bool

    _skip_registration: bool = False

    def __init__(
        self, H: Callable[[float], np.ndarray], output_shape: tuple[int] = None
    ):
        """Initialize a temporal solver

        :param H: Function of time, representing the temporal derivative at that time
        :type H: Callable[[float], np.ndarray]
        :param output_shape: Expected shape of the solution, defaults to None
        :type output_shape: tuple[int], optional
        """

        self.H = H

        if output_shape is None:
            output_shape = self.H(0).shape
        self.output_shape = output_shape

    def tqdm(self, t_start: float, t_end: float, enable: bool) -> tqdm:
        pbar = tqdm(
            desc=self.name + " solver", total=t_end - t_start, disable=not enable
        )
        pbar.bar_format = "{l_bar}{bar}| {n:#.02g}/{total:#.02g}"

        # Add func to update with t, not dt
        pbar.progress = lambda t: pbar.update(t - pbar.n)

        return pbar

    @abstractmethod
    def iterate(
        self,
        v_0: np.ndarray,
        t0: float,
        t_final: float,
        dt: float,
        dt_storage: float = None,
        verbose: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Iterate the time propagation scheme.
        Store the current state every :code:`dt_storage`

        Args:
            t_final (float):
                End time for calculations
            dt_storage (float, optional):
                Data storage period.
                If None, store each calculation :code:`dt`
                Defaults to None.

        Returns:
            np.ndarray:
                Time values, shape (:code:`n`,) for :code:`n` storage times
            np.ndarray:
                State at times stored in the other output. shape (:code:`n`, :code:`H`.shape)
        """
        pass

    def __init_subclass__(cls):
        """Register subclasses of :class:`TemporalSolver`"""
        if cls._skip_registration:
            return
        # Register new subclasses of TemporalSolver
        if _SCHEMES.get(cls.name) is None:
            _SCHEMES[cls.name] = cls
        else:
            raise ValueError("Cannot have two schemes with the same name")


def get_temporal_solver(scheme: str) -> type[TemporalSolver]:
    """Get a solver from its name, if it exists

    :param scheme: Name of the solver
    :type scheme: str
    :raises ValueError: If the scheme does not exist
    :return: A temporal solver class, NOT an instance
    :rtype: type[TemporalSolver]
    """
    if scheme in _SCHEMES.keys():
        return _SCHEMES[scheme]
    raise ValueError(
        f"Scheme {scheme} not found. Options are:\n" + "\n".join(_SCHEMES.keys())
    )
