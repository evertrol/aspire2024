"""Module with different ODE integrators"""

from .states import OrbitState
from ..equations import rhs


__all__ = ["integrate_euler", "integrate_rk2", "integrate_rk4"]


def integrate_euler(
    state0: OrbitState, tau: float, tend: float = 1.0
) -> tuple[list[float], list[OrbitState]]:
    """Integrate an orbit given an initial position, pos0, and velocity, vel0,
    using first-order Euler integration"""

    if tau <= 0:
        raise ValueError("tau should be larger than 0")
    if tend < tau:
        raise ValueError("tend should be larger than tau")

    times = []
    history = []

    # initialize time
    t: float = 0

    # store the initial conditions
    times.append(t)
    history.append(state0)

    # main timestep loop
    while t < tend:
        state_old = history[-1]

        # make sure that the last step does not take us past tend
        if t + tau > tend:
            tau = tend - t

        # get the RHS
        ydot = rhs(state_old)

        # do the Euler update
        state_new = state_old + tau * ydot
        t += tau

        # store the state
        times.append(t)
        history.append(state_new)

    return times, history


def integrate_rk2(
    state0: OrbitState, tau: float, tend: float = 1.0
) -> tuple[list[float], list[OrbitState]]:
    """Integrate an orbit given an initial position, pos0, and velocity, vel0,
    using second-order Runge-Kutta integration"""

    times = []
    history = []

    # initialize time
    t = 0.0

    # store the initial conditions
    times.append(t)
    history.append(state0)

    # main timestep loop
    while t < tend:
        state_old = history[-1]

        # make sure that the last step does not take us past tend
        if t + tau > tend:
            tau = tend - t

        # get the RHS
        ydot = rhs(state_old)

        # predict the state at the midpoint
        state_tmp = state_old + 0.5 * tau * ydot

        # evaluate the RHS at the midpoint
        ydot = rhs(state_tmp)

        # do the final update
        state_new = state_old + tau * ydot
        t += tau

        # store the state
        times.append(t)
        history.append(state_new)

    return times, history


def integrate_rk4(
    state0: OrbitState, tau: float, tend: float = 1.0
) -> tuple[list[float], list[OrbitState]]:
    """Integrate an orbit given an initial position, pos0, and velocity, vel0,
    using fourth-order Runge-Kutta integration"""

    times = []
    history = []

    # initialize time
    t = 0.0

    # store the initial conditions
    times.append(t)
    history.append(state0)

    # main timestep loop
    while t < tend:
        state_old = history[-1]

        # make sure that the last step does not take us past tend
        if t + tau > tend:
            tau = tend - t

        # get the RHS
        k1 = rhs(state_old)

        state_tmp = state_old + 0.5 * tau * k1
        k2 = rhs(state_tmp)

        state_tmp = state_old + 0.5 * tau * k2
        k3 = rhs(state_tmp)

        state_tmp = state_old + tau * k3
        k4 = rhs(state_tmp)

        # do the final update
        state_new = state_old + tau / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += tau

        # store the state
        times.append(t)
        history.append(state_new)

    return times, history
