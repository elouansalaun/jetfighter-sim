"""Validation visuelle du modèle point-masse (phase 2).

Trace trois vues du F-16 simplifié et affiche un résumé chiffré :

1. le domaine de vol en palier (altitude × Mach), plein gaz sec et pleine PC ;
2. le taux de virage soutenu en fonction du Mach, à trois altitudes ;
3. un looping plein gaz (vue de profil), qui traverse la verticale sans singularité.

Usage :
    python scripts/phase2_performance.py            # enregistre results/phase2_performance.png
    python scripts/phase2_performance.py --show     # et ouvre la fenêtre
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from jetsim.aircraft import performance as perf
from jetsim.aircraft.dynamics_3dof import PN, H, PointMassAircraft
from jetsim.aircraft.params import load_aircraft
from jetsim.core.integrators import simulate

# Palette de référence (mode clair) : séries dans un ordre fixe, texte en encre neutre
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]

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


def envelope(ac: PointMassAircraft, power: float, altitudes: np.ndarray) -> np.ndarray:
    """Tableau (altitude, Mach min, Mach max) du vol en palier stabilisé."""
    rows = []
    for h in altitudes:
        rng = perf.level_speed_range(ac, float(h), power=power)
        if rng is None:
            continue
        a = perf.speed_of_sound(float(h))
        rows.append((h, rng[0] / a, rng[1] / a))
    return np.array(rows)


def plot_envelope(ax, ac: PointMassAircraft) -> None:
    altitudes = np.linspace(0, 19_000, 58)
    for power, label, color in ((1.0, "Pleine PC", SERIES[0]), (0.77, "Plein gaz sec", SERIES[1])):
        env = envelope(ac, power, altitudes)
        mach = np.concatenate([env[:, 1], env[::-1, 2]])
        alt = np.concatenate([env[:, 0], env[::-1, 0]]) / 1000
        ax.plot(mach, alt, color=color, label=label)
        if power == 1.0:
            ax.fill(mach, alt, color=color, alpha=0.08, linewidth=0)
    ax.axhline(ac.params.limits.ceiling / 1000, color=INK_2, linewidth=1, linestyle=(0, (4, 3)))
    ax.text(
        1.72, ac.params.limits.ceiling / 1000 + 0.3, "plafond publié F-16", color=INK_2, fontsize=9
    )
    ax.set_title("Domaine de vol en palier")
    ax.set_xlabel("Mach")
    ax.set_ylabel("Altitude [km]")
    ax.set_xlim(0, 2.3)
    ax.set_ylim(0, 20)
    ax.legend(loc="upper right")


def plot_turn(ax, ac: PointMassAircraft) -> None:
    machs = np.linspace(0.3, 1.6, 53)
    for h, color in zip((0.0, 3000.0, 6000.0), SERIES, strict=True):
        rates = []
        for m in machs:
            t = perf.sustained_turn(ac, h, m * perf.speed_of_sound(h))
            rates.append(math.degrees(t.turn_rate) if t else np.nan)
        rates = np.array(rates)
        ax.plot(machs, rates, color=color, label=f"{h / 1000:.0f} km")
        i = int(np.nanargmax(rates))
        ax.annotate(
            f"{rates[i]:.1f}°/s",
            (machs[i], rates[i]),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            color=INK,
            fontsize=9,
        )
    ax.set_title("Taux de virage soutenu (pleine PC)")
    ax.set_xlabel("Mach")
    ax.set_ylabel("Taux de virage [°/s]")
    ax.set_ylim(0, 26)
    ax.legend(loc="upper right")


def plot_loop(ax, ac: PointMassAircraft) -> None:
    x0, _ = ac.trimmed_state(3000, 280)
    res = simulate(
        ac.derivatives,
        x0,
        22.0,
        u_const=np.array([1.0, math.radians(25), 0.0]),
        post_step=ac.post_step,
    )
    # On coupe quand le vecteur vitesse a fait un tour complet dans le plan vertical
    pitch_path = np.unwrap(np.arctan2(np.gradient(res.x[:, H]), np.gradient(res.x[:, PN])))
    end = int(np.searchsorted(pitch_path - pitch_path[0], 2 * math.pi)) + 1
    dist = res.x[:end, PN] / 1000
    alt = res.x[:end, H] / 1000
    n = np.array([ac.flight_data(x).load_factor for x in res.x[:end]])
    duration = res.t[end - 1]
    ax.plot(dist, alt, color=SERIES[0])
    ax.plot(dist[0], alt[0], "o", color=SERIES[0], markersize=8, markeredgecolor=SURFACE)
    ax.annotate(
        "entrée 280 m/s",
        (dist[0], alt[0]),
        xytext=(8, -14),
        textcoords="offset points",
        color=INK_2,
        fontsize=9,
    )
    top = int(np.argmax(alt))
    ax.annotate(
        f"sommet : {alt[top]:.2f} km, sur le dos",
        (dist[top], alt[top]),
        xytext=(0, 8),
        textcoords="offset points",
        ha="center",
        color=INK,
        fontsize=9,
    )
    ax.annotate(
        f"sortie après {duration:.1f} s",
        (dist[-1], alt[-1]),
        xytext=(8, 4),
        textcoords="offset points",
        color=INK_2,
        fontsize=9,
    )
    ax.set_title(f"Looping plein gaz, α max (n max = {n.max():.1f} g)")
    ax.set_xlabel("Distance vers le Nord [km]")
    ax.set_ylabel("Altitude [km]")
    ax.margins(y=0.12)
    ax.set_aspect("equal", adjustable="datalim")


def summary(ac: PointMassAircraft) -> None:
    a0, a11 = perf.speed_of_sound(0.0), perf.speed_of_sound(11_000.0)
    v3 = 0.8 * perf.speed_of_sound(3000.0)
    rows = [
        ("Mach max au niveau de la mer", f"{perf.level_speed_range(ac, 0.0)[1] / a0:.2f}", "≈ 1.2"),
        ("Mach max à 11 km", f"{perf.level_speed_range(ac, 11_000.0)[1] / a11:.2f}", "≈ 2.0"),
        ("Taux de montée max au sol [m/s]", f"{perf.max_rate_of_climb(ac, 0.0):.0f}", "≈ 250"),
        ("Plafond (0.5 m/s) [km]", f"{perf.ceiling(ac) / 1000:.1f}", "15.2 (opérationnel)"),
        (
            "Virage soutenu 3 km M0.8 [°/s]",
            f"{math.degrees(perf.sustained_turn(ac, 3000, v3).turn_rate):.1f}",
            "≈ 18–20",
        ),
    ]
    print(f"\n{'Grandeur':<36}{'Modèle':>10}   Référence publique")
    for name, val, ref in rows:
        print(f"{name:<36}{val:>10}   {ref}")

    x, u = ac.trimmed_state(3000, 263, load_factor=3)
    n = 5000
    t0 = time.perf_counter()
    for _ in range(n):
        x = ac.step(x, u)
    rate = n / (time.perf_counter() - t0)
    print(
        f"\nVitesse de simulation : {rate:,.0f} pas physiques/s (dt = 0.01 s, soit "
        f"{rate * 0.01:,.0f}× le temps réel)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="results/phase2_performance.png")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    ac = PointMassAircraft(load_aircraft("f16"))
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), layout="constrained")
    plot_envelope(axes[0], ac)
    plot_turn(axes[1], ac)
    plot_loop(axes[2], ac)
    fig.suptitle(
        "F-16 simplifié : modèle point-masse (phase 2)",
        color=INK,
        fontsize=14,
        fontweight="bold",
        x=0.01,
        ha="left",
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"Figure enregistrée : {out}")
    summary(ac)
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
