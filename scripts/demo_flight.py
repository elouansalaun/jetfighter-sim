"""Recorded demonstration flight: time series, 3D trajectory and Tacview file.

The F-16, in open loop (pre-programmed controls), flies from 3000 m / 250 m/s
a full-power loop followed by an aileron roll. Without an autopilot, the controls are tuned
"by hand": this is precisely what learning will have to do better.
Outputs go to ``results/flights/``:

* ``demo.npz``       : full recording (replayable, see ``jetsim.viz.recorder.replay``);
* ``demo_series.png`` and ``demo_3d.png``: plots;
* ``demo.acmi``      : to open in Tacview (https://www.tacview.net, free version).

Usage:
    python scripts/demo_flight.py [--model 3dof|6dof]
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np

from jetsim.aircraft import dynamics_3dof as d3
from jetsim.aircraft import dynamics_6dof as d6
from jetsim.aircraft.envelope import EnvelopeMonitor
from jetsim.aircraft.instruments import read_instruments
from jetsim.aircraft.params import load_aircraft, load_envelope
from jetsim.viz.plots import plot_time_series, plot_trajectory_3d
from jetsim.viz.recorder import FlightRecorder
from jetsim.viz.tacview import export_recording

DEG = math.pi / 180
DT, SUBSTEPS = 0.01, 10  # physics 100 Hz, recording 10 Hz


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


class Sequencer:
    """Chains the maneuvers while monitoring the instruments:
    level (2 s) -> loop -> level (3 s) -> roll -> level.

    * loop: pull until the flight path has made a full turn in the vertical
      plane (angle atan2(ḣ, horizontal speed along the initial course), accumulated);
    * roll: roll until ``roll_stop``; roll inertia completes the turn.
    """

    def __init__(self, model_kind: str, u0: np.ndarray) -> None:
        self.kind, self.u0 = model_kind, u0
        self.phase, self.t_phase = "level", 0.0
        self.path_angle = self.roll_angle = 0.0
        self._prev: tuple[float, float] | None = None
        self.course0: float | None = None
        # anticipation margins (lag of the angle-of-attack and roll responses)
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

        if self.phase == "level" and t >= 2.0:
            self._set("loop", t)
            self.path_angle = 0.0
        elif self.phase == "loop" and self.path_angle >= self.loop_stop:
            self._set("level after loop", t)
        elif self.phase == "level after loop" and t - self.t_phase >= 3.0:
            self._set("roll", t)
            self.roll_angle = 0.0
        elif self.phase == "roll" and abs(self.roll_angle) >= self.roll_stop:
            self._set("end", t)

        u = self.u0.copy()
        if self.kind == "6dof":
            if self.phase == "loop":
                u[d6.THROTTLE] = 1.0
                u[d6.ELEVATOR] -= 4 * DEG
            elif self.phase == "roll":
                u[d6.AILERON] = -10 * DEG  # δa < 0: roll right
        else:
            if self.phase == "loop":
                u[d3.THROTTLE] = 1.0
                u[d3.ALPHA_CMD] = 14 * DEG
            elif self.phase == "roll":
                u[d3.ROLL_RATE_CMD] = 240 * DEG
        return u


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("3dof", "6dof"), default="6dof")
    parser.add_argument("--out", default="results/flights/demo")
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
            if phase == "end":
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
    print(f"{len(flight)} samples, {flight.duration:.1f} s of flight -> {out.parent}/")
    for t_event, msg in flight.events:
        print(f"  event at {t_event:.1f} s: {msg}")


if __name__ == "__main__":
    main()
