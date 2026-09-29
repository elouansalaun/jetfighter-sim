"""Enregistrement, sauvegarde et rejeu des vols.

Un enregistrement contient, à chaque instant enregistré :

* le **temps**, l'**état** brut du modèle et la **commande** appliquée jusqu'à l'instant
  suivant ;
* les **instruments** (tableau de bord de la phase 4), pour tracer sans recalculer ;
* des **événements** horodatés (fin de vol, message…) ;
* de quoi **reconstruire le modèle** (type, configuration, centrage) et **rejouer** le vol :
  à partir de l'état initial et des commandes, la simulation est déterministe et doit
  retrouver exactement les mêmes états.

Format de fichier : ``.npz`` (numpy), lisible sans le projet avec ``np.load``.

Usage ::

    rec = FlightRecorder(model, physics_dt=0.01, substeps=10)
    rec.record(t, x, u)            # à chaque décision (ici toutes les 0.1 s)
    ...
    flight = rec.finish()
    flight.save("outputs/flights/vol.npz")
    states = replay(FlightRecording.load("outputs/flights/vol.npz"))
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
    """Ce qu'il faut pour reconstruire le modèle à l'identique."""
    if isinstance(model, d6.F16SixDof):
        return {"model": "6dof", "aircraft": aircraft, "xcg": model.xcg}
    if isinstance(model, d3.PointMassAircraft):
        return {"model": "3dof", "aircraft": aircraft}
    raise TypeError(f"Modèle non pris en charge : {type(model).__name__}")


def build_model(description: dict[str, Any]) -> Model:
    params = load_aircraft(description.get("aircraft", "f16"))
    if description["model"] == "6dof":
        return d6.F16SixDof(params, xcg=description.get("xcg"))
    if description["model"] == "3dof":
        return d3.PointMassAircraft(params)
    raise ValueError(f"Type de modèle inconnu : {description['model']}")


@dataclass
class FlightRecording:
    t: Array  # (N,)
    states: Array  # (N, n_état)
    controls: Array  # (N, n_commande) : commande appliquée entre t[k] et t[k+1]
    instruments: dict[str, Array]  # nom -> (N,)
    physics_dt: float
    substeps: int  # pas physiques entre deux enregistrements
    description: dict[str, Any]  # modèle (cf. model_description)
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
    """Accumule les échantillons d'un vol (voir le module)."""

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
            raise ValueError("Aucun échantillon enregistré.")
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
    """Re-simule le vol depuis l'état initial avec les commandes enregistrées.

    Returns:
        Les états re-simulés, de même forme que ``recording.states``. La simulation étant
        déterministe, ils doivent coïncider avec l'enregistrement.
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
