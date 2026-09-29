"""Capteurs imparfaits : bruit blanc et biais sur les instruments.

Chaque voie (``"alpha"``, ``"altitude"``, ``"nz"``…) peut recevoir :

* un **bruit** gaussien d'écart-type ``std``, tiré à chaque lecture ;
* un **biais** constant pendant un épisode, tiré au ``reset()`` avec l'écart-type
  ``bias_std`` (calibrage imparfait, différent d'un vol à l'autre).

Par défaut les capteurs sont parfaits. Le bruit sert à la robustesse des politiques RL
(phase 11) : une politique entraînée sur des mesures parfaites peut être fragile.

Configuration YAML (valeurs SI ; suffixe ``_deg`` pour les angles en degrés) ::

    alpha: {std_deg: 0.2, bias_std_deg: 0.5}
    altitude: {std: 5.0}
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from jetsim.aircraft.instruments import INSTRUMENT_NAMES, Instruments

DEG_TO_RAD = math.pi / 180


@dataclass(frozen=True)
class ChannelNoise:
    std: float = 0.0  # écart-type du bruit à chaque lecture
    bias_std: float = 0.0  # écart-type du biais tiré à chaque épisode

    @classmethod
    def from_dict(cls, d: dict[str, float]) -> ChannelNoise:
        std = d.get("std", 0.0) + d.get("std_deg", 0.0) * DEG_TO_RAD
        bias = d.get("bias_std", 0.0) + d.get("bias_std_deg", 0.0) * DEG_TO_RAD
        return cls(float(std), float(bias))


class SensorSuite:
    """Applique bruit et biais aux instruments (voir le module)."""

    def __init__(
        self,
        noise: dict[str, ChannelNoise] | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.noise = dict(noise or {})
        unknown = set(self.noise) - set(INSTRUMENT_NAMES)
        if unknown:
            raise ValueError(f"Voies inconnues : {sorted(unknown)}")
        self.rng = rng if rng is not None else np.random.default_rng()
        self.bias: dict[str, float] = {}
        self.reset()

    @classmethod
    def perfect(cls) -> SensorSuite:
        return cls({})

    @classmethod
    def from_dict(cls, d: dict[str, Any], rng: np.random.Generator | None = None) -> SensorSuite:
        return cls({k: ChannelNoise.from_dict(v) for k, v in d.items()}, rng)

    @classmethod
    def from_yaml(cls, path: str | Path, rng: np.random.Generator | None = None) -> SensorSuite:
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(yaml.safe_load(f) or {}, rng)

    @property
    def is_perfect(self) -> bool:
        return all(n.std == 0.0 and n.bias_std == 0.0 for n in self.noise.values())

    def reset(self, rng: np.random.Generator | None = None) -> None:
        """Nouvel épisode : tire de nouveaux biais (et change de générateur si fourni)."""
        if rng is not None:
            self.rng = rng
        self.bias = {
            name: float(self.rng.normal(0.0, n.bias_std)) if n.bias_std > 0 else 0.0
            for name, n in self.noise.items()
        }

    def measure(self, truth: Instruments) -> Instruments:
        """Mesures bruitées à partir des valeurs vraies."""
        if not self.noise:
            return truth
        changes = {}
        for name, n in self.noise.items():
            value = getattr(truth, name) + self.bias[name]
            if n.std > 0:
                value += float(self.rng.normal(0.0, n.std))
            changes[name] = value
        return dataclasses.replace(truth, **changes)
