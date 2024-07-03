"""Module with helper runner functions"""

import numpy as np

from .plotting.core import plot
from .core import init_circular, GM
from .integrators import integrate_euler, integrate_rk2, integrate_rk4
from .integrators.states import OrbitState


__all__ = ["run_euler", "run_rk2", "run_rk4", "run_rk4_elliptical"]


def run_euler(taus):
    """Run an Euler orbit integration, for one orbit"""

    state0 = init_circular()

    fig = None
    for tau in taus:
        _, history = integrate_euler(state0, tau, 1)

        label = rf"$\tau = {tau:6.4f}$"
        if not fig:
            fig = plot(history, label=label)
        else:
            plot(history, ax=fig.gca(), label=label)
    fig.gca().legend()

    return fig


def run_rk2(taus):
    """Run a second-order Runge-Kutta orbit integration, for one orbit"""

    state0 = init_circular()

    fig = None
    for tau in taus:
        _, history = integrate_rk2(state0, tau, 1)

        label = rf"$\tau = {tau:6.4f}$"
        if not fig:
            fig = plot(history, label=label)
        else:
            plot(history, ax=fig.gca(), label=label)

    fig.gca().legend()

    return fig


def run_rk4(taus):
    """Run a fourth-order Runge-Kutta orbit integration, for one orbit"""

    state0 = init_circular()

    fig = None
    for tau in taus:
        _, history = integrate_rk4(state0, tau, 1)

        label = rf"$\tau = {tau:6.4f}$"
        if not fig:
            fig = plot(history, label=label)
        else:
            plot(history, ax=fig.gca(), label=label)

    fig.gca().legend()

    return fig


def run_rk4_elliptical():
    """Run a fourth-order Runge-Kutta orbit integration, for a circular orbit"""

    a = 1.0
    e = 0.6

    x0 = 0
    y0 = a * (1 - e)
    u0 = -np.sqrt(GM / a * (1 + e) / (1 - e))
    v0 = 0

    state0 = OrbitState(x0, y0, u0, v0)

    tau = 0.025

    _, history = integrate_rk4(state0, tau, 1)

    fig = plot(history)

    return fig
