"""Démonstration de la phase 6 : pilote automatique et commandes de vol électriques.

1. **Mission** volée par le pilote automatique avec les deux modèles (mêmes consignes) :
   * 0–60 s    : montée à 4000 m, cap 090, 230 m/s ;
   * 60–100 s  : virage en palier à 4 g (inclinaison imposée), 250 m/s ;
   * 100–180 s : retour cap 000, descente à 3000 m, 200 m/s.
2. **Réponses indicielles** des commandes de vol électriques 6-DOF (n_z et taux de roulis) à
   trois points de vol, avec l'avion stable (x_cg = 0.30) et instable (x_cg = 0.35).

Sorties dans ``results/`` : ``phase6_mission.png``, ``phase6_fbw.png`` et
``flights/phase6_mission_6dof.acmi`` (Tacview).

Usage :
    python scripts/phase6_autopilot.py
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from jetsim.aircraft import dynamics_3dof as d3
from jetsim.aircraft import dynamics_6dof as d6
from jetsim.aircraft.instruments import read_instruments
from jetsim.aircraft.params import load_aircraft
from jetsim.control.autopilot import Autopilot, AutopilotTargets
from jetsim.control.fbw import HighLevelCommand, make_inner_loop
from jetsim.viz.recorder import FlightRecorder
from jetsim.viz.style import GRID, INK, INK_2, SERIES, apply_style
from jetsim.viz.tacview import export_recording

DEG = math.pi / 180
DT = 0.02  # boucle de commande 50 Hz (2 pas physiques)
OUT = Path("results")


def mission(t: float) -> AutopilotTargets:
    if t < 60:
        return AutopilotTargets(altitude=4000, heading=90 * DEG, airspeed=230)
    if t < 100:
        return AutopilotTargets(altitude=4000, bank=math.acos(1 / 4), airspeed=250)
    return AutopilotTargets(altitude=3000, heading=0.0, airspeed=200)


def fly_mission(kind: str, t_end: float = 180.0):
    params = load_aircraft("f16")
    if kind == "6dof":
        model: d3.PointMassAircraft | d6.F16SixDof = d6.F16SixDof(params)
        x, _ = model.trim(3000, 200)
    else:
        model = d3.PointMassAircraft(params)
        x, _ = model.trimmed_state(3000, 200)
    ap = Autopilot(model)
    ins = read_instruments(model, x)
    ap.reset(x, ins)
    rec = FlightRecorder(model, physics_dt=0.01, substeps=2, metadata={"mission": "phase 6"})
    t = 0.0
    while t < t_end:
        ap.targets = mission(t)
        u = ap.controls(ins, DT)
        rec.record(t, x, u)
        for _ in range(2):
            x = model.step(x, u)
        t = round(t + DT, 9)
        ins = read_instruments(model, x)
    return rec.finish()


def _no_wrap(deg: np.ndarray) -> np.ndarray:
    """Coupe la courbe (NaN) au passage 360° -> 0° pour éviter un trait vertical."""
    out = deg.astype(float).copy()
    out[out > 359.5] -= 360.0  # −0.0° s'affiche 360° : on le ramène à 0
    jumps = np.nonzero(np.abs(np.diff(out)) > 180)[0]
    out[jumps + 1] = np.nan
    return out


def plot_mission(flights: dict) -> None:
    apply_style()
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True, layout="constrained")
    ax = axes.ravel()
    t = next(iter(flights.values())).t
    targets = [mission(ti) for ti in t]
    alt_t = np.array([tg.altitude for tg in targets], dtype=float)
    spd_t = np.array([tg.airspeed for tg in targets], dtype=float)
    hdg_t = np.array([np.nan if tg.heading is None else tg.heading / DEG for tg in targets])

    for (label, rec), color in zip(flights.items(), SERIES, strict=False):
        ins = rec.instruments
        ax[0].plot(rec.t, ins["altitude"], color=color, label=label)
        ax[1].plot(rec.t, _no_wrap(np.degrees(ins["course"]) % 360), color=color, label=label)
        ax[2].plot(rec.t, ins["tas"], color=color, label=label)
        ax[3].plot(rec.t, ins["nz"], color=color, label=label)
    for a, y in ((ax[0], alt_t), (ax[1], _no_wrap(hdg_t % 360)), (ax[2], spd_t)):
        a.plot(t, y, color=INK_2, linewidth=1.2, linestyle=(0, (4, 3)), label="consigne")
    for a in ax:
        a.axvspan(60, 100, color=GRID, alpha=0.6, linewidth=0)
    ax[3].annotate("virage à 4 g", (80, 4.05), ha="center", color=INK, fontsize=9)
    titles = [("Altitude", "[m]"), ("Route", "[°]"), ("Vitesse vraie", "[m/s]"),
              ("Facteur de charge", "[g]")]  # fmt: skip
    for a, (title, unit) in zip(ax, titles, strict=True):
        a.set_title(title)
        a.set_ylabel(unit)
    ax[0].legend(loc="lower right")
    for a in axes[-1]:
        a.set_xlabel("Temps [s]")
    fig.suptitle("Pilote automatique : même mission, modèles 3-DOF et 6-DOF", color=INK,
                 fontsize=13, fontweight="bold", x=0.01, ha="left")  # fmt: skip
    fig.savefig(OUT / "phase6_mission.png", dpi=130)


def step_response(xcg: float, h: float, v: float, cmd):
    model = d6.F16SixDof(load_aircraft("f16"), xcg=xcg)
    x, u0 = model.trim(h, v)
    loop = make_inner_loop(model)
    loop.reset(x)
    t, out = 0.0, []
    while t < 3.0:
        ins = read_instruments(model, x)
        u = loop(ins, cmd(t, u0[0]), DT)
        for _ in range(2):
            x = model.step(x, u)
        t = round(t + DT, 9)
        out.append((t, ins.nz, ins.p / DEG, ins.beta / DEG))
    return np.array(out)


def plot_fbw() -> None:
    apply_style()
    fig, axes = plt.subplots(2, 2, figsize=(13, 7.5), layout="constrained")
    points = [(0, 150), (3000, 200), (6000, 250)]

    def nz_step(t, thr):
        return HighLevelCommand(4.0 if t > 0.5 else 1.0, 0.0, thr)

    def p_step(t, thr):
        return HighLevelCommand(1.0, 90 * DEG if t > 0.5 else 0.0, thr)

    for col, xcg in enumerate((0.30, 0.35)):
        for (h, v), color in zip(points, SERIES, strict=True):
            label = f"{h / 1000:.0f} km, {v} m/s"
            r = step_response(xcg, h, v, nz_step)
            axes[0, col].plot(r[:, 0], r[:, 1], color=color, label=label)
            r = step_response(xcg, h, v, p_step)
            axes[1, col].plot(r[:, 0], r[:, 2], color=color, label=label)
        stab = "stable" if xcg < 0.33 else "instable seul"
        axes[0, col].set_title(f"Échelon de n_z 1 → 4 g — x_cg = {xcg:.2f} ({stab})")
        axes[1, col].set_title(f"Échelon de taux de roulis 0 → 90°/s — x_cg = {xcg:.2f}")
        axes[0, col].set_ylabel("n_z [g]")
        axes[1, col].set_ylabel("p [°/s]")
        axes[1, col].set_xlabel("Temps [s]")
        for row, target in ((0, 4.0), (1, 90.0)):
            axes[row, col].axhline(target, color=INK_2, linewidth=1, linestyle=(0, (4, 3)))
    axes[0, 0].legend(loc="lower right")
    fig.suptitle("Commandes de vol électriques du 6-DOF : réponses indicielles", color=INK,
                 fontsize=13, fontweight="bold", x=0.01, ha="left")  # fmt: skip
    fig.savefig(OUT / "phase6_fbw.png", dpi=130)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    flights = {"3-DOF": fly_mission("3dof"), "6-DOF": fly_mission("6dof")}
    plot_mission(flights)
    export_recording(flights["6-DOF"], OUT / "flights" / "phase6_mission_6dof.acmi",
                     pilot="Pilote automatique", title="Mission pilote automatique")  # fmt: skip
    plot_fbw()

    print(f"{'':8}{'alt. finale':>12}{'route finale':>14}{'vitesse finale':>16}{'n_z max':>9}")
    for name, rec in flights.items():
        ins = rec.instruments
        print(f"{name:8}{ins['altitude'][-1]:10.0f} m{math.degrees(ins['course'][-1]) % 360:12.1f}°"
              f"{ins['tas'][-1]:13.1f} m/s{ins['nz'].max():9.2f}")  # fmt: skip
    print(f"Figures : {OUT}/phase6_mission.png, {OUT}/phase6_fbw.png")


if __name__ == "__main__":
    main()
