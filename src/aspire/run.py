"""Module with helper runner functions"""

from typing import Any, Optional

from matplotlib.figure import Figure as MPLFigure

from .plotting.core import plot
from .core import init_circular, init_elliptical
from .integrators import integrate_euler, integrate_rk2, integrate_rk4

# this is not yet supported in MyPy 1.10; will be in 1.11
#type Figure = Optional[MPLFigure]  # type: ignore

__all__ = ["run_euler", "run_rk2", "run_rk4", "run_rk4_elliptical"]


def run_euler(taus: list[float]) -> Any:
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

    if fig:
        fig.gca().legend()

    return fig


def run_rk2(taus: list[float]) -> Any:
    """Run a second-order Runge-Kutta orbit integration, for one orbit"""

    state0 = init_circular()

    fig: Any = None
    for tau in taus:
        _, history = integrate_rk2(state0, tau, 1)

        label = rf"$\tau = {tau:6.4f}$"
        if not fig:
            fig = plot(history, label=label)
        else:
            plot(history, ax=fig.gca(), label=label)

    if fig:
        fig.gca().legend()

    return fig


def run_rk4(taus: list[float]) -> Any:
    """Run a fourth-order Runge-Kutta orbit integration, for one orbit"""

    state0 = init_circular()

    fig: Any = None
    for tau in taus:
        _, history = integrate_rk4(state0, tau, 1)

        label = rf"$\tau = {tau:6.4f}$"
        if not fig:
            fig = plot(history, label=label)
        else:
            plot(history, ax=fig.gca(), label=label)

    if fig:
        fig.gca().legend()

    return fig


def run_rk4_elliptical(taus: list[float]) -> None | Any:
    """Run a fourth-order Runge-Kutta orbit integration, for an elliptical orbit"""

    state0 = init_elliptical()

    fig: Any = None
    for tau in taus:
        _, history = integrate_rk4(state0, tau, 1)

        label = rf"$\tau = {tau:6.4f}$"
        if not fig:
            fig = plot(history, label=label)
        else:
            plot(history, ax=fig.gca(), label=label)

    if fig:
        fig.gca().legend()

    return fig
