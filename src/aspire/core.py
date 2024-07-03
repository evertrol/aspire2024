"""Module for initial conditions"""

import numpy as np

from .integrators.states import OrbitState
from .constants import GM


__all__ = ["GM", "init_circular"]


def init_circular():
    """Set initial conditions for a circular orbit"""

    x0 = 0
    y0 = 1
    u0 = -np.sqrt(GM / y0)
    v0 = 0

    return OrbitState(x0, y0, u0, v0)
