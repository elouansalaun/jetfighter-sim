"""Validation du modèle 6-DOF (phase 3) : équilibres, modes propres, réponses aux gouvernes.

Affiche :
* le point d'équilibre de référence Stevens & Lewis retrouvé par le modèle ;
* les modes propres (oscillation d'incidence, phugoïde, roulis hollandais, roulis, spirale) ;
* la vitesse de simulation.

Trace (outputs/phase3_validation.png) quatre réponses depuis un vol stabilisé à 3000 m, 200 m/s :
1. doublet de profondeur (incidence et assiette) ;
2. impulsion d'ailerons (gîte) ;
3. impulsion de direction (dérapage et gîte : roulis hollandais) ;
4. effet du centrage : même impulsion de profondeur à x_cg = 0.30 (stable) et 0.35 (instable).

Usage :
    python scripts/phase3_validation.py [--show]
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from jetfighter.aircraft import analysis as an
from jetfighter.aircraft import dynamics_6dof as d6
from jetfighter.aircraft.params import load_aircraft
from jetfighter.core.integrators import simulate

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
    """Réponse à une commande ``u(t) = u_trim + pulse(t)`` depuis l'équilibre."""
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

    # 1. Doublet de profondeur
    ax = axes[0, 0]
    t, data = respond(model, doublet(d6.ELEVATOR, -2 * DEG, 1.0, 1.0), 12.0)
    alpha = np.degrees([d.alpha for d in data])
    theta = np.degrees([d.pitch for d in data])
    ax.plot(t, alpha, color=SERIES[0], label="incidence α")
    ax.plot(t, theta, color=SERIES[1], label="assiette θ")
    ax.axvspan(1.0, 3.0, color=GRID, alpha=0.5, linewidth=0)
    ax.set_title("Doublet de profondeur ±2° (1 s + 1 s)")
    ax.set_xlabel("Temps [s]")
    ax.set_ylabel("Angle [°]")
    ax.legend(loc="upper right")

    # 2. Impulsion d'ailerons (δa < 0 = roulis à droite)
    ax = axes[0, 1]
    t, data = respond(model, pulse_on(d6.AILERON, -5 * DEG, 1.0, 2.0), 8.0)
    phi = np.degrees([d.roll for d in data])
    p = np.degrees([d.p for d in data])
    ax.plot(t, phi, color=SERIES[0])
    ax.axvspan(1.0, 2.0, color=GRID, alpha=0.5, linewidth=0)
    i = int(np.argmax(np.abs(p)))
    ax.set_title(f"Impulsion d'ailerons −5° pendant 1 s (p max = {abs(p[i]):.0f}°/s)")
    ax.set_xlabel("Temps [s]")
    ax.set_ylabel("Gîte φ [°]")
    label_end(ax, t, phi, "gîte φ")

    # 3. Impulsion de direction : roulis hollandais
    ax = axes[1, 0]
    t, data = respond(model, pulse_on(d6.RUDDER, 5 * DEG, 1.0, 1.5), 12.0)
    beta = np.degrees([d.beta for d in data])
    phi = np.degrees([d.roll for d in data])
    ax.plot(t, beta, color=SERIES[0], label="dérapage β")
    ax.plot(t, phi, color=SERIES[1], label="gîte φ")
    ax.axvspan(1.0, 1.5, color=GRID, alpha=0.5, linewidth=0)
    ax.set_title("Impulsion de direction +5° pendant 0.5 s")
    ax.set_xlabel("Temps [s]")
    ax.set_ylabel("Angle [°]")
    ax.legend(loc="upper right")

    # 4. Effet du centrage
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
    ax.set_title("Centrage : impulsion de profondeur −1° pendant 0.5 s")
    ax.set_xlabel("Temps [s]")
    ax.set_ylabel("Incidence α [°]")
    ax.legend(loc="upper left")

    fig.suptitle(
        f"F-16 6-DOF : réponses depuis le vol stabilisé ({H0:.0f} m, {V0:.0f} m/s)",
        color=INK,
        fontsize=14,
        fontweight="bold",
        x=0.01,
        ha="left",
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"Figure enregistrée : {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="outputs/phase3_validation.png")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    model = d6.F16SixDof(load_aircraft("f16"))
    print(f"Centrage x_cg = {model.xcg:.2f}\n")

    print(f"{'Point de vol':<18}{'manette':>9}{'δe [°]':>9}{'α [°]':>8}{'Mach':>7}")
    for h, v in [(0, 153), (3000, 200), (6000, 250), (9000, 250)]:
        x, u = model.trim(h, v)
        fd = model.flight_data(x)
        print(
            f"{h:>5.0f} m {v:>4.0f} m/s  {u[0]:>9.3f}{math.degrees(u[1]):>9.2f}"
            f"{math.degrees(fd.alpha):>8.2f}{fd.mach:>7.2f}"
        )

    x, u = model.trim(H0, V0)
    modes = an.flight_modes(an.linearize(model, x, u)[0])
    print(f"\nModes propres à {H0:.0f} m, {V0:.0f} m/s :")
    for m in modes.values():
        if m.eigenvalue.imag:
            print(
                f"  {m.name:<26} ω = {m.natural_frequency:5.2f} rad/s  ζ = {m.damping:4.2f}"
                f"  période = {m.period:5.1f} s"
            )
        else:
            print(f"  {m.name:<26} τ = {m.time_constant:6.2f} s")

    n = 1000
    t0 = time.perf_counter()
    for _ in range(n):
        x = model.step(x, u)
    rate = n / (time.perf_counter() - t0)
    print(f"\nVitesse : {rate:,.0f} pas physiques/s ({rate * 0.01:,.0f}× le temps réel)")

    plot_all(model, Path(args.out))
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
