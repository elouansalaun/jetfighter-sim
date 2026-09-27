"""Vol de démonstration enregistré : séries temporelles, trajectoire 3D et fichier Tacview.

Le F-16, en boucle ouverte (commandes préprogrammées), enchaîne depuis 3000 m / 250 m/s
un looping plein gaz puis un tonneau. Sans pilote automatique, les commandes sont réglées
« à la main » : c'est justement ce que l'apprentissage devra faire mieux.
Les sorties vont dans ``outputs/flights/`` :

* ``demo.npz``       : enregistrement complet (rejouable, cf. ``jetfighter.viz.recorder.replay``) ;
* ``demo_series.png`` et ``demo_3d.png`` : tracés ;
* ``demo.acmi``      : à ouvrir dans Tacview (https://www.tacview.net, version gratuite).

Usage :
    python scripts/demo_flight.py [--model 3dof|6dof]
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np

from jetfighter.aircraft import dynamics_3dof as d3
from jetfighter.aircraft import dynamics_6dof as d6
from jetfighter.aircraft.envelope import EnvelopeMonitor
from jetfighter.aircraft.instruments import read_instruments
from jetfighter.aircraft.params import load_aircraft, load_envelope
from jetfighter.viz.plots import plot_time_series, plot_trajectory_3d
from jetfighter.viz.recorder import FlightRecorder
from jetfighter.viz.tacview import export_recording

DEG = math.pi / 180
DT, SUBSTEPS = 0.01, 10  # physique 100 Hz, enregistrement 10 Hz


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


class Sequencer:
    """Enchaîne les figures en surveillant les instruments :
    palier (2 s) -> looping -> palier (3 s) -> tonneau -> palier.

    * looping : on tire tant que la trajectoire n'a pas fait un tour complet dans le plan
      vertical (angle atan2(ḣ, vitesse horizontale le long du cap initial), cumulé) ;
    * tonneau : on roule jusqu'à ``roll_stop`` ; l'inertie en roulis termine le tour.
    """

    def __init__(self, model_kind: str, u0: np.ndarray) -> None:
        self.kind, self.u0 = model_kind, u0
        self.phase, self.t_phase = "palier", 0.0
        self.path_angle = self.roll_angle = 0.0
        self._prev: tuple[float, float] | None = None
        self.course0: float | None = None
        # marges d'anticipation (retard de la réponse en incidence et en roulis)
        self.loop_stop = (355 if model_kind == "6dof" else 350) * DEG
        self.roll_stop = (287 if model_kind == "6dof" else 312) * DEG

    def _set(self, phase: str, t: float) -> None:
        self.phase, self.t_phase = phase, t

    def __call__(self, t: float, ins) -> np.ndarray:
        if self.course0 is None:
            self.course0 = ins.course
        v_along = ins.tas * math.cos(ins.gamma) * math.cos(ins.course - self.course0)
        path = math.atan2(ins.vertical_speed, v_along)
        if self._prev is not None:
            self.path_angle += _wrap(path - self._prev[0])
            self.roll_angle += _wrap(ins.roll - self._prev[1])
        self._prev = (path, ins.roll)

        if self.phase == "palier" and t >= 2.0:
            self._set("looping", t)
            self.path_angle = 0.0
        elif self.phase == "looping" and self.path_angle >= self.loop_stop:
            self._set("palier après looping", t)
        elif self.phase == "palier après looping" and t - self.t_phase >= 3.0:
            self._set("tonneau", t)
            self.roll_angle = 0.0
        elif self.phase == "tonneau" and abs(self.roll_angle) >= self.roll_stop:
            self._set("fin", t)

        u = self.u0.copy()
        if self.kind == "6dof":
            if self.phase == "looping":
                u[d6.THROTTLE] = 1.0
                u[d6.ELEVATOR] -= 4 * DEG
            elif self.phase == "tonneau":
                u[d6.AILERON] = -10 * DEG  # δa < 0 : roulis à droite
        else:
            if self.phase == "looping":
                u[d3.THROTTLE] = 1.0
                u[d3.ALPHA_CMD] = 14 * DEG
            elif self.phase == "tonneau":
                u[d3.ROLL_RATE_CMD] = 240 * DEG
        return u


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("3dof", "6dof"), default="6dof")
    parser.add_argument("--out", default="outputs/flights/demo")
    args = parser.parse_args()

    params = load_aircraft("f16")
    six = args.model == "6dof"
    model: d3.PointMassAircraft | d6.F16SixDof
    if six:
        model = d6.F16SixDof(params)
        x, u0 = model.trim(3000.0, 250.0)
    else:
        model = d3.PointMassAircraft(params)
        x, u0 = model.trimmed_state(3000.0, 250.0)
    monitor = EnvelopeMonitor(load_envelope("f16", six_dof=six))
    rec = FlightRecorder(model, physics_dt=DT, substeps=SUBSTEPS, metadata={"demo": True})

    pilot = Sequencer(args.model, u0)
    t, t_end, phase = 0.0, 60.0, pilot.phase
    while t < t_end:
        u = pilot(t, read_instruments(model, x))
        if pilot.phase != phase:
            rec.event(t, pilot.phase)
            phase = pilot.phase
            if phase == "fin":
                t_end = t + 6.0
        rec.record(t, x, u)
        for _ in range(SUBSTEPS):
            x = model.step(x, u, DT)
        t = round(t + DT * SUBSTEPS, 6)
        violation = monitor.check(read_instruments(model, x), DT * SUBSTEPS, state=x)
        if violation is not None:
            rec.record(t, x, u)
            rec.event(t, monitor.message)
            break
    flight = rec.finish()

    out = Path(args.out)
    flight.save(out.with_suffix(".npz"))
    plot_time_series(flight).savefig(out.parent / f"{out.name}_series.png", dpi=130)
    plot_trajectory_3d(flight).savefig(out.parent / f"{out.name}_3d.png", dpi=130)
    export_recording(flight, out.with_suffix(".acmi"))
    print(f"{len(flight)} échantillons, {flight.duration:.1f} s de vol -> {out.parent}/")
    for t_event, msg in flight.events:
        print(f"  événement à {t_event:.1f} s : {msg}")


if __name__ == "__main__":
    main()
