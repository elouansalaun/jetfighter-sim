"""Control building blocks: PID with anti-windup, first-order filters.

"Parallel" PID ::

    u = Kp·e + Ki·∫e dt − Kd·dy/dt        (derivative on the MEASUREMENT, filtered)

* derivative on the measurement rather than on the error: no derivative kick when the
  setpoint changes abruptly;
* anti-windup by clamping: the integrator is frozen while the output is saturated
  and the error pushes in the direction of saturation;
* conditional integration (``integral_zone``): integrate only near the setpoint,
  to keep the integrator from winding up during large transients;
* ``reset(output=…)`` initializes the integrator so that the first output equals ``output``
  (bumpless transfer when engaging the autopilot in flight).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class PID:
    kp: float
    ki: float = 0.0
    kd: float = 0.0
    out_min: float = -math.inf
    out_max: float = math.inf
    derivative_tau: float = 0.05  # [s] derivative filter
    integral_zone: float = math.inf  # only integrate if |error| ≤ this value
    integral: float = field(default=0.0, init=False)
    _last_y: float | None = field(default=None, init=False)
    _d_filt: float = field(default=0.0, init=False)

    def reset(self, output: float = 0.0, error: float = 0.0) -> None:
        """Reset; the integral is chosen so that the output equals ``output``."""
        self._last_y = None
        self._d_filt = 0.0
        self.integral = (output - self.kp * error) / self.ki if self.ki else 0.0

    def __call__(self, error: float, dt: float, measurement: float | None = None,
                 gain: float = 1.0) -> float:  # fmt: skip
        """Controller output.

        Args:
            error: setpoint − measurement.
            dt: time step [s].
            measurement: measurement (for the derivative term); ``None`` = no derivative term.
            gain: overall multiplicative factor (gain scheduling).
        """
        d_term = 0.0
        if self.kd and measurement is not None:
            if self._last_y is not None and dt > 0:
                raw = (measurement - self._last_y) / dt
                a = dt / (self.derivative_tau + dt)
                self._d_filt += a * (raw - self._d_filt)
            self._last_y = measurement
            d_term = -self.kd * self._d_filt

        candidate = self.integral + error * dt
        unsat = gain * (self.kp * error + self.ki * candidate + d_term)
        out = min(max(unsat, self.out_min), self.out_max)
        saturated_high = unsat > self.out_max and error * self.ki * gain > 0
        saturated_low = unsat < self.out_min and error * self.ki * gain < 0
        if not (saturated_high or saturated_low) and abs(error) <= self.integral_zone:
            self.integral = candidate
        return out


@dataclass
class Washout:
    """First-order high-pass filter: passes variations, removes the steady component.

    Used by the yaw damper: we want to damp oscillations of r, not oppose the
    steady yaw rate of a coordinated turn.
    """

    tau: float
    _state: float = field(default=0.0, init=False)

    def reset(self, value: float = 0.0) -> None:
        self._state = value

    def __call__(self, value: float, dt: float) -> float:
        self._state += (value - self._state) * dt / (self.tau + dt)
        return value - self._state


def wrap_angle(angle: float) -> float:
    """Angle in [−π, π) (heading errors)."""
    return (angle + math.pi) % (2 * math.pi) - math.pi
