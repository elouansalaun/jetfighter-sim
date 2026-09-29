"""Validation of the 6-DOF model (phase 3): trim points, eigenmodes, control-surface responses.

Prints:
* the Stevens & Lewis reference trim point as recovered by the model;
* the eigenmodes (short period, phugoid, Dutch roll, roll subsidence, spiral);
* the simulation speed.

Plots (results/phase3_validation.png) four responses from steady flight at 3000 m, 200 m/s:
1. elevator doublet (angle of attack and pitch);
2. aileron pulse (roll angle);
3. rudder pulse (sideslip and roll angle: Dutch roll);
4. CG effect: same elevator pulse at x_cg = 0.30 (stable) and 0.35 (unstable).

"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from jetsim.aircraft import analysis as an
from jetsim.aircraft import dynamics_6dof as d6
from jetsim.aircraft.params import load_aircraft
from jetsim.core.integrators import simulate

SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
DEG = math.pi / 180

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "font.size": 10,
        "legend.frameon": False,
        "legend.labelcolor": INK_2,
        "lines.linewidth": 2,
    }
)

H0, V0 = 3000.0, 200.0


def respond(model: d6.F16SixDof, pulse, t_end: float):
    """Response to a command ``u(t) = u_trim + pulse(t)`` from trim."""
    x0, u0 = model.trim(H0, V0)
    res = simulate(
        model.derivatives,
        x0,
        t_end,
        controller=lambda t, _x: u0 + pulse(t),
        post_step=model.post_step,
    )
    data = [model.flight_data(x) for x in res.x]
    return res.t, data


def pulse_on(index: int, amplitude: float, t_on: float, t_off: float):
    def f(t: float) -> np.ndarray:
        du = np.zeros(4)
        if t_on <= t < t_off:
            du[index] = amplitude
        return du

    return f


def doublet(index: int, amplitude: float, t0: float, width: float):
    def f(t: float) -> np.ndarray:
        du = np.zeros(4)
        if t0 <= t < t0 + width:
            du[index] = amplitude
        elif t0 + width <= t < t0 + 2 * width:
            du[index] = -amplitude
        return du

    return f


def label_end(ax, t, y, text, color=INK_2, dy=0):
    ax.annotate(
        text,
        (t[-1], y[-1]),
        xytext=(4, dy),
        textcoords="offset points",
        color=color,
        fontsize=9,
        va="center",
    )


def plot_all(model: d6.F16SixDof, out: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), layout="constrained")

    # 1. Elevator doublet
    ax = axes[0, 0]
    t, data = respond(model, doublet(d6.ELEVATOR, -2 * DEG, 1.0, 1.0), 12.0)
    alpha = np.degrees([d.alpha for d in data])
    theta = np.degrees([d.pitch for d in data])
    ax.plot(t, alpha, color=SERIES[0], label="angle of attack α")
    ax.plot(t, theta, color=SERIES[1], label="pitch θ")
    ax.axvspan(1.0, 3.0, color=GRID, alpha=0.5, linewidth=0)
    ax.set_title("Elevator doublet ±2° (1 s + 1 s)")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Angle [°]")
    ax.legend(loc="upper right")

    # 2. Aileron pulse (δa < 0 = roll right)
    ax = axes[0, 1]
    t, data = respond(model, pulse_on(d6.AILERON, -5 * DEG, 1.0, 2.0), 8.0)
    phi = np.degrees([d.roll for d in data])
    p = np.degrees([d.p for d in data])
    ax.plot(t, phi, color=SERIES[0])
    ax.axvspan(1.0, 2.0, color=GRID, alpha=0.5, linewidth=0)
    i = int(np.argmax(np.abs(p)))
    ax.set_title(f"Aileron pulse −5° for 1 s (max p = {abs(p[i]):.0f}°/s)")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Roll angle φ [°]")
    label_end(ax, t, phi, "roll φ")

    # 3. Rudder pulse: Dutch roll
    ax = axes[1, 0]
    t, data = respond(model, pulse_on(d6.RUDDER, 5 * DEG, 1.0, 1.5), 12.0)
    beta = np.degrees([d.beta for d in data])
    phi = np.degrees([d.roll for d in data])
    ax.plot(t, beta, color=SERIES[0], label="sideslip β")
    ax.plot(t, phi, color=SERIES[1], label="roll φ")
    ax.axvspan(1.0, 1.5, color=GRID, alpha=0.5, linewidth=0)
    ax.set_title("Rudder pulse +5° for 0.5 s")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Angle [°]")
    ax.legend(loc="upper right")

    # 4. CG effect
    ax = axes[1, 1]
    params = model.params
    for xcg, color, name in (
        (0.30, SERIES[0], "x_cg = 0.30 (stable)"),
        (0.35, SERIES[1], "x_cg = 0.35 (nominal S&L)"),
    ):
        m = d6.F16SixDof(params, xcg=xcg)
        t, data = respond(m, pulse_on(d6.ELEVATOR, -1 * DEG, 1.0, 1.5), 10.0)
        alpha = np.degrees([d.alpha for d in data])
        ax.plot(t, alpha, color=color, label=name)
    ax.axvspan(1.0, 1.5, color=GRID, alpha=0.5, linewidth=0)
    ax.set_title("CG position: elevator pulse −1° for 0.5 s")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Angle of attack α [°]")
    ax.legend(loc="upper left")

    fig.suptitle(
        f"F-16 6-DOF: responses from steady flight ({H0:.0f} m, {V0:.0f} m/s)",
        color=INK,
        fontsize=14,
        fontweight="bold",
        x=0.01,
        ha="left",
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"Figure saved: {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="results/phase3_validation.png")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    model = d6.F16SixDof(load_aircraft("f16"))
    print(f"CG position x_cg = {model.xcg:.2f}\n")

    print(f"{'Flight point':<18}{'throttle':>9}{'δe [°]':>9}{'α [°]':>8}{'Mach':>7}")
    for h, v in [(0, 153), (3000, 200), (6000, 250), (9000, 250)]:
        x, u = model.trim(h, v)
        fd = model.flight_data(x)
        print(
            f"{h:>5.0f} m {v:>4.0f} m/s  {u[0]:>9.3f}{math.degrees(u[1]):>9.2f}"
            f"{math.degrees(fd.alpha):>8.2f}{fd.mach:>7.2f}"
        )

    x, u = model.trim(H0, V0)
    modes = an.flight_modes(an.linearize(model, x, u)[0])
    print(f"\nEigenmodes at {H0:.0f} m, {V0:.0f} m/s:")
    for m in modes.values():
        if m.eigenvalue.imag:
            print(
                f"  {m.name:<26} ω = {m.natural_frequency:5.2f} rad/s  ζ = {m.damping:4.2f}"
                f"  period = {m.period:5.1f} s"
            )
        else:
            print(f"  {m.name:<26} τ = {m.time_constant:6.2f} s")

    n = 1000
    t0 = time.perf_counter()
    for _ in range(n):
        x = model.step(x, u)
    rate = n / (time.perf_counter() - t0)
    print(f"\nSpeed: {rate:,.0f} physics steps/s ({rate * 0.01:,.0f}× real time)")

    plot_all(model, Path(args.out))
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
