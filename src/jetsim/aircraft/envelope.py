"""Flight envelope monitoring: detects situations that end an episode.

Checks, in priority order:

1. ``NUMERICAL``  : non-finite state (NaN/inf) — numerical divergence;
2. ``GROUND``     : altitude below ground;
3. ``OVERLOAD``   : load factor outside structural limits;
4. ``CEILING``    : altitude above the maximum;
5. ``LOW_SPEED``  : true airspeed below the minimum;
6. ``OVERSPEED``  : Mach above the maximum (or the model's validity domain);
7. ``SIDESLIP``   : sideslip outside the tables' domain;
8. ``STALL``      : angle of attack beyond the stall angle for more than
   ``stall_duration`` seconds (a brief excursion is tolerated: it is a maneuver,
   not a loss of control).


"""

from __future__ import annotations

import math
from enum import Enum

import numpy as np
import numpy.typing as npt

from jetsim.aircraft.instruments import Instruments
from jetsim.aircraft.params import EnvelopeLimits


class Violation(Enum):
    NUMERICAL = "numerical divergence"
    GROUND = "ground collision"
    OVERLOAD = "structural overload"
    CEILING = "ceiling exceeded"
    LOW_SPEED = "airspeed too low"
    OVERSPEED = "overspeed"
    SIDESLIP = "excessive sideslip"
    STALL = "sustained stall"


class EnvelopeMonitor:
    """Check the envelope at each step; keeps track of the time spent stalled."""

    def __init__(self, limits: EnvelopeLimits) -> None:
        self.limits = limits
        self.stall_time = 0.0
        self.message = ""

    def reset(self) -> None:
        self.stall_time = 0.0
        self.message = ""

    def check(
        self,
        ins: Instruments,
        dt: float,
        state: npt.NDArray[np.float64] | None = None,
    ) -> Violation | None:
        """Return the detected violation, or ``None`` if the flight continues.

        Args:
            ins: **true** instruments (not the noisy measurements).
            dt: time elapsed since the previous check [s].
            state: raw state vector, checked for NaN (optional).
        """
        lim = self.limits
        values = (ins.altitude, ins.tas, ins.mach, ins.nz, ins.alpha, ins.beta)
        if (state is not None and not np.all(np.isfinite(state))) or not all(
            math.isfinite(v) for v in values
        ):
            return self._fail(Violation.NUMERICAL, "non-finite state (NaN or infinity)")
        if ins.altitude < lim.min_altitude:
            return self._fail(Violation.GROUND, f"altitude {ins.altitude:.0f} m")
        if not lim.n_min <= ins.nz <= lim.n_max:
            return self._fail(
                Violation.OVERLOAD,
                f"n = {ins.nz:.1f} g outside [{lim.n_min:.0f}, {lim.n_max:.0f}] g",
            )
        if ins.altitude > lim.max_altitude:
            return self._fail(Violation.CEILING, f"altitude {ins.altitude:.0f} m")
        if ins.tas < lim.min_airspeed:
            return self._fail(Violation.LOW_SPEED, f"V = {ins.tas:.0f} m/s")
        if ins.mach > lim.max_mach:
            return self._fail(Violation.OVERSPEED, f"Mach {ins.mach:.3f} > {lim.max_mach:.2f}")
        if abs(ins.beta) > lim.beta_max:
            return self._fail(Violation.SIDESLIP, f"β = {math.degrees(ins.beta):.0f}°")

        if ins.alpha > lim.alpha_stall:
            self.stall_time += dt
            if self.stall_time > lim.stall_duration + 1e-9:  # rounding tolerance
                return self._fail(
                    Violation.STALL,
                    f"α = {math.degrees(ins.alpha):.0f}° for {self.stall_time:.1f} s",
                )
        else:
            self.stall_time = 0.0
        return None

    def _fail(self, violation: Violation, detail: str) -> Violation:
        self.message = f"{violation.value}: {detail}"
        return violation
