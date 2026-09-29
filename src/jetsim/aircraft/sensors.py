"""Imperfect sensors: white noise and bias on the instruments.

Each channel (``"alpha"``, ``"altitude"``, ``"nz"``…) can receive:

* a Gaussian noise with standard deviation ``std``, drawn at each reading;
* a bias constant over an episode, drawn at ``reset()`` with standard deviation
  ``bias_std`` (imperfect calibration, different from one flight to the next).

By default the sensors are perfect. Noise is used for the robustness of RL policies
(phase 11): a policy trained on perfect measurements can be brittle.

YAML configuration (SI values; ``_deg`` suffix for angles in degrees) ::

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
    std: float = 0.0  # standard deviation of the noise at each reading
    bias_std: float = 0.0  # standard deviation of the bias drawn at each episode

    @classmethod
    def from_dict(cls, d: dict[str, float]) -> ChannelNoise:
        std = d.get("std", 0.0) + d.get("std_deg", 0.0) * DEG_TO_RAD
        bias = d.get("bias_std", 0.0) + d.get("bias_std_deg", 0.0) * DEG_TO_RAD
        return cls(float(std), float(bias))


class SensorSuite:
    """Apply noise and bias to the instruments (see the module docstring)."""

    def __init__(
        self,
        noise: dict[str, ChannelNoise] | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.noise = dict(noise or {})
        unknown = set(self.noise) - set(INSTRUMENT_NAMES)
        if unknown:
            raise ValueError(f"Unknown channels: {sorted(unknown)}")
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
        """New episode: draw new biases (and switch generator if one is provided)."""
        if rng is not None:
            self.rng = rng
        self.bias = {
            name: float(self.rng.normal(0.0, n.bias_std)) if n.bias_std > 0 else 0.0
            for name, n in self.noise.items()
        }

    def measure(self, truth: Instruments) -> Instruments:
        """Noisy measurements from the true values."""
        if not self.noise:
            return truth
        changes = {}
        for name, n in self.noise.items():
            value = getattr(truth, name) + self.bias[name]
            if n.std > 0:
                value += float(self.rng.normal(0.0, n.std))
            changes[name] = value
        return dataclasses.replace(truth, **changes)
