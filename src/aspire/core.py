"""Module for initial conditions"""

import numpy as np

from .integrators.states import OrbitState
from .constants import GM


__all__ = ["GM", "init_circular", "init_elliptical"]


def init_circular() -> OrbitState:
    """Set initial conditions for a circular orbit"""

    x0 = 0
    y0 = 1
    u0 = -np.sqrt(GM / y0)
    v0 = 0

    return OrbitState(x0, y0, u0, v0)


def init_elliptical() -> OrbitState:
    """Set initial conditions for an elliptical orbit"""

    a = 1.0
    e = 0.6

    x0 = 0
    y0 = a * (1 - e)
    u0 = -np.sqrt(GM / a * (1 + e) / (1 - e))
    v0 = 0

    return OrbitState(x0, y0, u0, v0)
