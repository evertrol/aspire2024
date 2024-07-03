from matplotlib.axes._axes import Axes
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from ..integrators.states import OrbitState


__all__ = ["plot"]


def plot(
    history: list[OrbitState], ax: None | Axes = None, label: None | str = None
) -> None | Figure:
    """make a plot of the solution.  If ax is None we setup a figure
    and make the entire plot returning the figure object, otherwise, we
    just append the plot to a current axis"""

    fig = None

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

        # draw the Sun
        ax.scatter([0], [0], marker=(20, 1, 0), color="y", s=250)  # type: ignore

    # draw the orbit
    xs = [q.x for q in history]
    ys = [q.y for q in history]

    ax.plot(xs, ys, label=label)

    if fig is not None:
        ax.set_aspect("equal")
        ax.set_xlabel("x [AU]")
        ax.set_ylabel("y [AU]")

    return fig
