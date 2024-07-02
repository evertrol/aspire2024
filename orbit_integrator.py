#!/usr/bin/env python

"""Script to write and compare three integrators for ordinary
differential equations. In particular, we create the simplests, using
Euler's method, then create a second and fourth order Runge-Kutta
integrator.

"""

import numpy as np
import matplotlib.pyplot as plt

GM = 4*np.pi**2


class OrbitState:
    """Container to hold the body position and velocity"""

    def __init__(self, x, y, u, v):
        self.x = x
        self.y = y
        self.u = u
        self.v = v

    def __add__(self, other):
        return OrbitState(self.x + other.x, self.y + other.y,
                          self.u + other.u, self.v + other.v)

    def __sub__(self, other):
        return OrbitState(self.x - other.x, self.y - other.y,
                          self.u - other.u, self.v - other.v)

    def __mul__(self, other):
        return OrbitState(other * self.x, other * self.y,
                          other * self.u, other * self.v)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __str__(self):
        return f"{self.x:10.6f} {self.y:10.6f} {self.u:10.6f} {self.v:10.6f}"



def rhs(state):
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


def euler_orbit(state0, tau, T):
    """Integrate an orbit given an initial position, pos0, and velocity, vel0,
    using first-order Euler integration"""

    times = []
    history = []

    # initialize time
    t = 0

    # store the initial conditions
    times.append(t)
    history.append(state0)

    # main timestep loop
    while t < T:

        state_old = history[-1]

        # make sure that the last step does not take us past T
        if t + tau > T:
            tau = T - t

        # get the RHS
        ydot = rhs(state_old)

        # do the Euler update
        state_new = state_old + tau * ydot
        t += tau

        # store the state
        times.append(t)
        history.append(state_new)

    return times, history


def initial_conditions():
    x0 = 0
    y0 = 1
    u0 = -np.sqrt(GM / y0)
    v0 = 0

    return OrbitState(x0, y0, u0, v0)


def plot(history, ax=None, label=None):
    """make a plot of the solution.  If ax is None we setup a figure
    and make the entire plot returning the figure object, otherwise, we
    just append the plot to a current axis"""

    fig = None

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

        # draw the Sun
        ax.scatter([0], [0], marker=(20,1), color="y", s=250)

    # draw the orbit
    xs = [q.x for q in history]
    ys = [q.y for q in history]

    ax.plot(xs, ys, label=label)

    if fig is not None:
        ax.set_aspect("equal")
        ax.set_xlabel("x [AU]")
        ax.set_ylabel("y [AU]")

    return fig



def int_rk2(state0, tau, T):

    times = []
    history = []

    # initialize time
    t = 0

    # store the initial conditions
    times.append(t)
    history.append(state0)

    # main timestep loop
    while t < T:

        state_old = history[-1]

        # make sure that the last step does not take us past T
        if t + tau > T:
            tau = T - t

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


# Try the new integrator with 1 month timestep

# In[11]:


def run_rk2(tau=1/12):

    state0 = initial_conditions()

    times, history = int_rk2(state0, tau, 1)
    fig = plot(history)

    return fig


def run_rk2_multiple():
    taus = [0.1, 0.05, 0.025, 0.0125]

    state0 = initial_conditions()

    for n, tau in enumerate(taus):
        times, history = int_rk2(state0, tau, 1)

        if n == 0:
            fig = plot(history, label=rf"$\tau = {tau:6.4f}$")
        else:
            plot(history, ax=fig.gca(), label=rf"$\tau = {tau:6.4f}$")

    fig.gca().legend()

    return fig


def int_rk4(state0, tau, T):

    times = []
    history = []

    # initialize time
    t = 0

    # store the initial conditions
    times.append(t)
    history.append(state0)

    # main timestep loop
    while t < T:

        state_old = history[-1]

        # make sure that the last step does not take us past T
        if t + tau > T:
            tau = T - t

        # get the RHS
        k1 = rhs(state_old)

        state_tmp = state_old + 0.5 * tau * k1
        k2 = rhs(state_tmp)

        state_tmp = state_old + 0.5 * tau * k2
        k3 = rhs(state_tmp)

        state_tmp = state_old + tau * k3
        k4 = rhs(state_tmp)

        # do the final update
        state_new = state_old + tau / 6.0 * (k1 + 2*k2 + 2*k3 + k4)
        t += tau

        # store the state
        times.append(t)
        history.append(state_new)

    return times, history


def run_rk4():

    state0 = initial_conditions()

    tau = 1.0/12.0

    times, history = int_rk4(state0, tau, 1)

    fig = plot(history)

    return fig



def run_rk4_multiple():
    taus = [0.1, 0.05, 0.025]

    state0 = initial_conditions()

    for n, tau in enumerate(taus):
        times, history = int_rk4(state0, tau, 1)

        if n == 0:
            fig = plot(history, label=rf"$\tau = {tau:6.4f}$")
        else:
            plot(history, ax=fig.gca(), label=rf"$\tau = {tau:6.4f}$")

    fig.gca().legend()

    return fig


def run_rk4_elliptical():

    a = 1.0
    e = 0.6

    x0 = 0
    y0 = a * (1 - e)
    u0 = -np.sqrt(GM / a * (1 + e)/(1 - e))
    v0 = 0

    state0 = OrbitState(x0, y0, u0, v0)



    tau = 0.025
    T = 1

    times, history = int_rk4(state0, tau, 1)

    fig = plot(history)

    return fig


def run():
    run_rk2()
    run_rk2_multiple()
    run_rk4()
    run_rk4_multiple()
    run_rk4_elliptical()


if __name__ == '__main__':
    run()