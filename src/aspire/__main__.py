#!/usr/bin/env python

"""Script to write and compare three integrators for ordinary
differential equations. In particular, we create the simplests, using
Euler's method, then create a second and fourth order Runge-Kutta
integrator.

"""

import argparse
from pathlib import Path

from . import run_euler, run_rk2, run_rk4


def run(integrator: str, tau: list[float], output: str | Path):
    if isinstance(tau, (int, float)):
        tau = [tau]

    fig = None
    if integrator == "euler":
        fig = run_euler(tau)
    elif integrator == "rk2":
        fig = run_rk2(tau)
    elif integrator == "rk4":
        fig = run_rk4(tau)
    if fig:
        fig.savefig(output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "integrator", choices=["euler", "rk2", "rk4"], help="Pick an integrator"
    )
    parser.add_argument(
        "--tau", type=float, action="append", help="Specify step size(s)"
    )
    parser.add_argument("--output", default="aspire.png", help="Output file name")
    args = parser.parse_args()
    if not args.tau:
        args.tau = [0.1]

    run(args.integrator, tau=args.tau, output=args.output)


if __name__ == "__main__":
    main()
