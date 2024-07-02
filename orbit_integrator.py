#!/usr/bin/env python

"""Script to write and compare three integrators for ordinary
differential equations. In particular, we create the simplests, using
Euler's method, then create a second and fourth order Runge-Kutta
integrator.

"""

import argparse

import integrators


def run(integrator, tau):
    if isinstance(tau, (int, float)):
        tau = [tau]

    if integrator == "euler":
        integrators.run_euler(tau)
    elif integrator == "rk2":
        integrators.run_rk2(tau)
    elif integrator == "rk4":
        integrators.run_rk4(tau)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "integrator", choices=["euler", "rk2", "rk4"], help="Pick an integrator"
    )
    parser.add_argument(
        "--tau", type=float, action="append", help="Specify step size(s)"
    )
    args = parser.parse_args()
    if not args.tau:
        args.tau = [0.1]

    run(args.integrator, tau=args.tau)


if __name__ == "__main__":
    main()
