from abc import ABC, abstractmethod
from typing import Callable

import numpy as np
from tqdm import tqdm

from ..nature_constants import h_bar

# Dict to store subclasses of BaseTemporalSolver
_SCHEMES = {}


class BaseTemporalSolver(ABC):

    name: str
    order: int
    explicit: bool
    stable: bool

    _skip_registration: bool = False

    def __init__(self, H: Callable[[float], np.ndarray], output_shape: tuple[int] = None):
        """
        Initialise a temporal solver

        Args:
            H (Callable[[float], np.ndarray]): 
                Function of time, returning a linear operator 
                representing the state function at that time.
        """

        self.H = H

        if output_shape is None:
            output_shape = self.H(0).shape
        self.output_shape = output_shape

    def tqdm(self, t_start: float, t_end: float, enable: bool) -> tqdm:
        pbar = tqdm(desc=self.name + " solver", total=t_end - t_start, disable=not enable)
        pbar.bar_format = "{l_bar}{bar}| {n:#.02g}/{total:#.02g}"

        # Add func to update with t, not dt
        pbar.progress = lambda t: pbar.update(t - pbar.n)

        return pbar


    @abstractmethod
    def iterate(self, v_0: np.ndarray, t0: float, t_final: float, 
        dt: float, dt_storage: float = None, verbose: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """
        Iterate the time propagation scheme.
        Store the current state every `dt_storage`

        Args:
            t_final (float): 
                End time for calculations
            dt_storage (float, optional): 
                Data storage period. 
                If None, store each calculation `dt`
                Defaults to None.

        Returns:
            np.ndarray:
                Time values, shape (n,) for n storage times
            np.ndarray:
                State at times stored in the other output. shape (n, H.shape)
        """
        pass

    def __init_subclass__(cls):
        if cls._skip_registration:
            return
        # Register new subclasses of TemporalSolver
        if _SCHEMES.get(cls.name) is None:
            _SCHEMES[cls.name] = cls
        else:
            raise ValueError("Cannot have two schemes with the same name")


def get_temporal_solver(scheme: str) -> type[BaseTemporalSolver]:
    if scheme in _SCHEMES.keys():
        return _SCHEMES[scheme]
    raise ValueError(f"Scheme {scheme} not found. Options are:\n" 
        + "\n".join(_SCHEMES.keys()))
