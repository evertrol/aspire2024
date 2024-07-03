import numpy as np

from .integrators.states import OrbitState
from .constants import GM


__all__ = ["rhs"]


def rhs(state: OrbitState) -> OrbitState:
    r"""RHS of the equations of motion

    \dot{\bf r} = {\bf v}

    \dot{\bf v} = -\frac{GM_\star {\bf r}}{r^3}

    where
        {\bf r} = (x, y) and {\bf v} = (u, v).

    """

    # current radius
    r = np.sqrt(state.x**2 + state.y**2)

    # position
    xdot = state.u
    ydot = state.v

    # velocity
    udot = -GM * state.x / r**3
    vdot = -GM * state.y / r**3

    return OrbitState(xdot, ydot, udot, vdot)
