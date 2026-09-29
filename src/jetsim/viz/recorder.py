"""Recording, saving and replaying flights.

A recording contains, at each recorded instant:

* the time, the raw model state and the control applied until the next
  instant;
* the instruments (phase 4 instrument panel), to plot without recomputing;
* timestamped events (end of flight, message…);
* what is needed to rebuild the model (type, configuration, CG) and replay the flight:
  from the initial state and the controls, the simulation is deterministic and must
  reproduce exactly the same states.

File format: ``.npz`` (numpy), readable without the project using ``np.load``.

Usage ::

    rec = FlightRecorder(model, physics_dt=0.01, substeps=10)
    rec.record(t, x, u)            # at each decision (here every 0.1 s)
    ...
    flight = rec.finish()
    flight.save("outputs/flights/flight.npz")
    states = replay(FlightRecording.load("outputs/flights/flight.npz"))
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from jetsim.aircraft import dynamics_3dof as d3
from jetsim.aircraft import dynamics_6dof as d6
from jetsim.aircraft.instruments import INSTRUMENT_NAMES, read_instruments
from jetsim.aircraft.params import load_aircraft

Array = npt.NDArray[np.float64]
Model = d3.PointMassAircraft | d6.F16SixDof


def model_description(model: Model, aircraft: str = "f16") -> dict[str, Any]:
    """What is needed to rebuild the exact same model."""
    if isinstance(model, d6.F16SixDof):
        return {"model": "6dof", "aircraft": aircraft, "xcg": model.xcg}
    if isinstance(model, d3.PointMassAircraft):
        return {"model": "3dof", "aircraft": aircraft}
    raise TypeError(f"Unsupported model: {type(model).__name__}")


def build_model(description: dict[str, Any]) -> Model:
    params = load_aircraft(description.get("aircraft", "f16"))
    if description["model"] == "6dof":
        return d6.F16SixDof(params, xcg=description.get("xcg"))
    if description["model"] == "3dof":
        return d3.PointMassAircraft(params)
    raise ValueError(f"Unknown model type: {description['model']}")


@dataclass
class FlightRecording:
    t: Array  # (N,)
    states: Array  # (N, n_state)
    controls: Array  # (N, n_control): control applied between t[k] and t[k+1]
    instruments: dict[str, Array]  # name -> (N,)
    physics_dt: float
    substeps: int  # physics steps between two recorded samples
    description: dict[str, Any]  # model (see model_description)
    events: list[tuple[float, str]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.t.size)

    @property
    def model_type(self) -> str:
        return str(self.description["model"])

    @property
    def duration(self) -> float:
        return float(self.t[-1] - self.t[0]) if len(self) else 0.0

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        header = {
            "physics_dt": self.physics_dt,
            "substeps": self.substeps,
            "description": self.description,
            "events": self.events,
            "metadata": self.metadata,
            "format": 1,
        }
        np.savez_compressed(
            path,
            t=self.t,
            states=self.states,
            controls=self.controls,
            instrument_names=np.array(list(self.instruments)),
            instruments=np.array([self.instruments[k] for k in self.instruments]),
            header=np.array(json.dumps(header, ensure_ascii=False)),
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> FlightRecording:
        with np.load(path, allow_pickle=False) as data:
            header = json.loads(str(data["header"]))
            names = [str(n) for n in data["instrument_names"]]
            values = data["instruments"]
            return cls(
                t=data["t"],
                states=data["states"],
                controls=data["controls"],
                instruments={n: values[i] for i, n in enumerate(names)},
                physics_dt=float(header["physics_dt"]),
                substeps=int(header["substeps"]),
                description=header["description"],
                events=[(float(t), str(msg)) for t, msg in header["events"]],
                metadata=header["metadata"],
            )


class FlightRecorder:
    """Accumulates the samples of a flight (see the module docstring)."""

    def __init__(
        self,
        model: Model,
        *,
        physics_dt: float,
        substeps: int,
        aircraft: str = "f16",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.model = model
        self.physics_dt = physics_dt
        self.substeps = substeps
        self.description = model_description(model, aircraft)
        self.metadata = dict(metadata or {})
        self._t: list[float] = []
        self._x: list[Array] = []
        self._u: list[Array] = []
        self._ins: list[tuple[float, ...]] = []
        self.events: list[tuple[float, str]] = []

    def record(self, t: float, x: Array, u: Array) -> None:
        ins = read_instruments(self.model, x)
        self._t.append(float(t))
        self._x.append(np.array(x, dtype=np.float64))
        self._u.append(np.array(u, dtype=np.float64))
        self._ins.append(tuple(getattr(ins, n) for n in INSTRUMENT_NAMES))

    def event(self, t: float, text: str) -> None:
        self.events.append((float(t), text))

    def __len__(self) -> int:
        return len(self._t)

    def finish(self) -> FlightRecording:
        if not self._t:
            raise ValueError("No samples recorded.")
        ins = np.array(self._ins, dtype=np.float64)
        return FlightRecording(
            t=np.array(self._t),
            states=np.array(self._x),
            controls=np.array(self._u),
            instruments={n: ins[:, i] for i, n in enumerate(INSTRUMENT_NAMES)},
            physics_dt=self.physics_dt,
            substeps=self.substeps,
            description=self.description,
            events=list(self.events),
            metadata=dict(self.metadata),
        )


def replay(recording: FlightRecording, model: Model | None = None) -> Array:
    """Re-simulate the flight from the initial state with the recorded controls.

    Returns:
        The re-simulated states, with the same shape as ``recording.states``. Since the
        simulation is deterministic, they must match the recording.
    """
    model = model if model is not None else build_model(recording.description)
    states = np.empty_like(recording.states)
    x = recording.states[0].copy()
    states[0] = x
    for k in range(len(recording) - 1):
        u = recording.controls[k]
        for _ in range(recording.substeps):
            x = model.step(x, u, recording.physics_dt)
        states[k + 1] = x
    return states
