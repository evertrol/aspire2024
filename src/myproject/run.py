import numpy as np

from .plotting.core import plot
from .core import initial_conditions, GM
from .integrators import euler_orbit, int_rk2, int_rk4
from .integrators.states import OrbitState


__all__ = ["run_euler", "run_rk2", "run_rk4", "run_rk4_elliptical"]


def run_euler(taus):
    state0 = initial_conditions()

    fig = None
    for tau in taus:
        _, history = euler_orbit(state0, tau, 1)

        label = rf"$\tau = {tau:6.4f}$"
        if not fig:
            fig = plot(history, label=label)
        else:
            plot(history, ax=fig.gca(), label=label)
    fig.gca().legend()

    return fig


def run_rk2(taus):
    state0 = initial_conditions()

    fig = None
    for tau in taus:
        _, history = int_rk2(state0, tau, 1)

        label = rf"$\tau = {tau:6.4f}$"
        if not fig:
            fig = plot(history, label=label)
        else:
            plot(history, ax=fig.gca(), label=label)

    fig.gca().legend()

    return fig


def run_rk4(taus):
    state0 = initial_conditions()

    fig = None
    for tau in taus:
        _, history = int_rk4(state0, tau, 1)

        label = rf"$\tau = {tau:6.4f}$"
        if not fig:
            fig = plot(history, label=label)
        else:
            plot(history, ax=fig.gca(), label=label)

    fig.gca().legend()

    return fig


def run_rk4_elliptical():
    a = 1.0
    e = 0.6

    x0 = 0
    y0 = a * (1 - e)
    u0 = -np.sqrt(GM / a * (1 + e) / (1 - e))
    v0 = 0

    state0 = OrbitState(x0, y0, u0, v0)

    tau = 0.025

    _, history = int_rk4(state0, tau, 1)

    fig = plot(history)

    return fig
