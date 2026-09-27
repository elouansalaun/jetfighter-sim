"""Tracés d'un vol enregistré : séries temporelles et trajectoire 3D.

Usage ::

    from jetfighter.viz.plots import plot_time_series, plot_trajectory_3d
    fig = plot_time_series(recording)
    fig.savefig("outputs/vol.png")
"""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from jetfighter.aircraft import dynamics_3dof as d3
from jetfighter.aircraft import dynamics_6dof as d6
from jetfighter.viz.recorder import FlightRecording
from jetfighter.viz.style import GRID, INK, INK_2, SERIES, apply_style

DEG = 180.0 / math.pi


def _lines(ax, t, series: list[tuple[str, np.ndarray]], ylabel: str, title: str) -> None:
    for (label, y), color in zip(series, SERIES, strict=False):
        ax.plot(t, y, color=color, label=label)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    if len(series) > 1:
        ax.legend(loc="best", fontsize=8.5)


def _mark_events(axes, rec: FlightRecording) -> None:
    for t_event, _ in rec.events:
        for ax in axes:
            ax.axvline(t_event, color=INK_2, linewidth=1, linestyle=(0, (3, 3)))


def plot_time_series(rec: FlightRecording, title: str | None = None) -> Figure:
    """Tableau de bord temporel : 8 graphiques (altitude, vitesses, α/β, n_z, attitude,
    moteur, commandes, vitesses angulaires)."""
    apply_style()
    ins, t = rec.instruments, rec.t
    fig, axes = plt.subplots(4, 2, figsize=(13, 11), sharex=True, layout="constrained")
    ax = axes.ravel()

    _lines(ax[0], t, [("altitude", ins["altitude"])], "[m]", "Altitude")
    _lines(ax[1], t, [("vraie (TAS)", ins["tas"]), ("corrigée (CAS)", ins["cas"])],
           "[m/s]", "Vitesse")  # fmt: skip
    _lines(ax[2], t, [("incidence α", ins["alpha"] * DEG), ("dérapage β", ins["beta"] * DEG)],
           "[°]", "Incidence et dérapage")  # fmt: skip
    _lines(ax[3], t, [("n_z", ins["nz"])], "[g]", "Facteur de charge")
    _lines(ax[4], t, [("gîte φ", ins["roll"] * DEG), ("assiette θ", ins["pitch"] * DEG)],
           "[°]", "Attitude")  # fmt: skip

    controls = rec.controls
    if rec.model_type == "6dof":
        _lines(ax[5], t, [("manette", controls[:, d6.THROTTLE]), ("puissance", ins["power"])],
               "[0–1]", "Moteur")  # fmt: skip
        _lines(ax[6], t, [("profondeur δe", controls[:, d6.ELEVATOR] * DEG),
                          ("ailerons δa", controls[:, d6.AILERON] * DEG),
                          ("direction δr", controls[:, d6.RUDDER] * DEG)],
               "[°]", "Commandes de gouvernes")  # fmt: skip
    else:
        _lines(ax[5], t, [("manette", controls[:, d3.THROTTLE]), ("puissance", ins["power"])],
               "[0–1]", "Moteur")  # fmt: skip
        _lines(ax[6], t, [("α commandée", controls[:, d3.ALPHA_CMD] * DEG),
                          ("α", ins["alpha"] * DEG)],
               "[°]", "Commande d'incidence")  # fmt: skip
    _lines(ax[7], t, [("p", ins["p"] * DEG), ("q", ins["q"] * DEG), ("r", ins["r"] * DEG)],
           "[°/s]", "Vitesses angulaires")  # fmt: skip

    for a in axes[-1]:
        a.set_xlabel("Temps [s]")
    _mark_events(ax, rec)
    if rec.events:
        ax[0].annotate(rec.events[-1][1], (rec.events[-1][0], ins["altitude"][-1]),
                       xytext=(-6, 8), textcoords="offset points", ha="right",
                       color=INK, fontsize=8.5)  # fmt: skip
    fig.suptitle(title or f"Vol enregistré ({rec.model_type.upper()}, {rec.duration:.1f} s)",
                 color=INK, fontsize=13, fontweight="bold", x=0.01, ha="left")  # fmt: skip
    return fig


def plot_trajectory_3d(rec: FlightRecording, title: str | None = None) -> Figure:
    """Trajectoire dans l'espace (Est, Nord, altitude) avec sa trace au sol."""
    apply_style()
    ins = rec.instruments
    east, north, alt = ins["east"] / 1000, ins["north"] / 1000, ins["altitude"] / 1000
    fig = plt.figure(figsize=(9, 7.5), layout="constrained")
    ax = fig.add_subplot(projection="3d")
    ax.plot(east, north, alt, color=SERIES[0], label="trajectoire")
    ax.plot(east, north, np.zeros_like(alt), color=INK_2, linewidth=1.0,
            linestyle=(0, (4, 3)), label="trace au sol")  # fmt: skip
    # projection verticale tous les ~5 s pour lire l'altitude
    step = max(1, len(east) // 8)
    for k in range(0, len(east), step):
        ax.plot([east[k]] * 2, [north[k]] * 2, [0, alt[k]], color=GRID, linewidth=0.8)
    ax.scatter([east[0]], [north[0]], [alt[0]], color=SERIES[0], s=40, label="départ")
    ax.scatter([east[-1]], [north[-1]], [alt[-1]], color=SERIES[1], s=40, label="fin")
    ax.set_xlabel("Est [km]")
    ax.set_ylabel("Nord [km]")
    ax.set_zlabel("Altitude [km]")
    ax.set_zlim(0, max(1.0, float(alt.max()) * 1.1))
    # Échelles horizontales identiques pour ne pas déformer les virages
    span = max(float(np.ptp(east)), float(np.ptp(north)), 0.5) / 2
    cx, cy = float(east.mean()), float(north.mean())
    ax.set_xlim(cx - span, cx + span)
    ax.set_ylim(cy - span, cy + span)
    ax.set_box_aspect((1, 1, 0.7))
    ax.view_init(elev=18, azim=-55)
    ax.legend(loc="upper left", fontsize=8.5)
    ax.set_title(title or "Trajectoire", loc="left", color=INK)
    return fig
