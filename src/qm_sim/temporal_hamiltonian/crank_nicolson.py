from .base import BaseTemporalHamiltonian

import numpy as np

class CrankNicolson(BaseTemporalHamiltonian):
    order = 2
    explicit = False
    stable = True
    name = "crank-nicolson"

           
    def _get_dt(self) -> float:
        pass

    def iterate(self, t: float, dt_storage: float = None):
        pass
