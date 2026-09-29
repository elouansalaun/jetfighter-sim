"""Simplified propulsion: afterburning turbofan.

Thrust as a function of the power setting P ∈ [0, 1]:

* 0 → ``mil_power``: from idle to full dry power (MIL), linear interpolation;
* ``mil_power`` → 1: from full dry power to full afterburner (MAX).

Corrections:

* altitude: (ρ/ρ0)^n. With n ≈ 1.15, this matches within a few % the static
  (Mach 0) thrusts of the Stevens & Lewis model tables between 0 and 50,000 ft;
* ram effect: factor (1 + k·M), k interpolated between ``ram_factor_mil``
  (dry settings) and ``ram_factor_max`` (full afterburner);
* engine limit: thrust never exceeds ``ram_limit`` × the static sea-level thrust
  at the same setting. At low altitude and high speed, a real engine is limited by
  temperature and pressure; without this bound, the ram effect would give an
  unrealistic max Mach at sea level.

Power follows the throttle with a first-order lag (time constant
``engine_time_constant``): the RL agent cannot get full thrust instantly.

This simple model is used by the point-mass model (phase 2). The 6-DOF model uses
``TabulatedTurbofan`` (Stevens & Lewis altitude × Mach thrust tables), below.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from jetsim.aircraft.f16_aero import UniformGrid, lerp2
from jetsim.aircraft.params import PropulsionParams
from jetsim.core.constants import RHO0


class SimpleTurbofan:
    """Parametric thrust model (see the module docstring)."""

    def __init__(self, params: PropulsionParams) -> None:
        self.p = params

    def altitude_factor(self, rho: float) -> float:
        """Ratio thrust(altitude) / thrust(sea level) at zero Mach."""
        return float((rho / RHO0) ** self.p.density_exponent)

    def thrust(self, power: float, rho: float, mach: float) -> float:
        """Thrust [N].

        Args:
            power: effective power setting, in [0, 1] (clamped).
            rho: air density [kg/m³].
            mach: Mach number.
        """
        p = self.p
        power = min(max(power, 0.0), 1.0)
        if power <= p.mil_power:
            frac = power / p.mil_power
            t_sl = p.thrust_idle_sl + (p.thrust_mil_sl - p.thrust_idle_sl) * frac
            ram = p.ram_factor_mil
        else:
            frac = (power - p.mil_power) / (1.0 - p.mil_power)
            t_sl = p.thrust_mil_sl + (p.thrust_max_sl - p.thrust_mil_sl) * frac
            ram = p.ram_factor_mil + (p.ram_factor_max - p.ram_factor_mil) * frac
        factor = self.altitude_factor(rho) * (1.0 + ram * max(mach, 0.0))
        return t_sl * min(factor, p.ram_limit)

    def power_rate(self, power: float, throttle: float) -> float:
        """Rate of the effective power toward the throttle command (first order)."""
        throttle = min(max(throttle, 0.0), 1.0)
        return (throttle - power) / self.p.engine_time_constant


# ==========================================================================
# Tabulated engine from the Stevens & Lewis model (used by the 6-DOF model)
# ==========================================================================
class TabulatedTurbofan:
    """F100 engine from the Stevens & Lewis model: thrust tables (altitude × Mach) for
    idle, full dry power and full afterburner, plus power dynamics.

    **Power** is expressed here as a fraction [0, 1] (0.5 = full dry power, above that:
    afterburner), i.e. the original model's "percentage" divided by 100.

    * power command: "throttle gearing" law (knee at 77 % throttle);
    * dynamics: first order whose time constant depends on the error (slow response
      for large errors), 5 s⁻¹ in afterburner, dry <-> afterburner transitions in steps.
    """

    def __init__(self, engine: dict[str, Any], *, lbf_to_n: float, ft_to_m: float) -> None:
        self.alt = UniformGrid(
            engine["altitude_ft"]["start"] * ft_to_m,
            engine["altitude_ft"]["step"] * ft_to_m,
            engine["altitude_ft"]["count"],
        )
        self.mach = UniformGrid.from_dict(engine["mach"])
        self.idle = np.asarray(engine["idle_lbf"], dtype=float) * lbf_to_n
        self.mil = np.asarray(engine["mil_lbf"], dtype=float) * lbf_to_n
        self.max = np.asarray(engine["max_lbf"], dtype=float) * lbf_to_n
        gear = engine["throttle_gear"]
        self._gear = (
            float(gear["breakpoint"]),
            float(gear["slope_dry"]),
            float(gear["slope_ab"]),
            float(gear["offset_ab"]),
        )
        self._ab = float(engine["afterburner_threshold"]) / 100.0

    def commanded_power(self, throttle: float) -> float:
        """Power command [0, 1] as a function of throttle [0, 1]."""
        throttle = min(max(throttle, 0.0), 1.0)
        bp, s_dry, s_ab, off_ab = self._gear
        pct = s_dry * throttle if throttle <= bp else s_ab * throttle + off_ab
        return pct / 100.0

    def throttle_for_power(self, power: float) -> float:
        """Inverse of ``commanded_power`` (useful for trimming)."""
        bp, s_dry, s_ab, off_ab = self._gear
        pct = power * 100.0
        return pct / s_dry if pct <= s_dry * bp else (pct - off_ab) / s_ab

    def thrust(self, power: float, altitude: float, mach: float) -> float:
        """Thrust [N] for a power [0, 1], an altitude [m] and a Mach number."""
        h = max(altitude, 0.0)
        t_mil = lerp2(self.mil, self.alt, h, self.mach, mach)
        if power < self._ab:
            t_idle = lerp2(self.idle, self.alt, h, self.mach, mach)
            return t_idle + (t_mil - t_idle) * power / self._ab
        t_max = lerp2(self.max, self.alt, h, self.mach, mach)
        return t_mil + (t_max - t_mil) * (power - self._ab) / (1.0 - self._ab)

    def power_rate(self, power: float, commanded: float) -> float:
        """Power rate [1/s] (fractions [0, 1])."""
        ab = self._ab
        if commanded >= ab:
            if power >= ab:
                inv_tau, target = 5.0, commanded
            else:
                target = 0.6
                inv_tau = _inverse_time_constant(target - power)
        elif power >= ab:
            inv_tau, target = 5.0, 0.4
        else:
            target = commanded
            inv_tau = _inverse_time_constant(target - power)
        return inv_tau * (target - power)


def _inverse_time_constant(delta: float) -> float:
    """Engine 1/τ as a function of the power error (fraction): 1 if ≤ 25 %, 0.1 if ≥ 50 %."""
    dp = delta * 100.0
    if dp <= 25.0:
        return 1.0
    if dp >= 50.0:
        return 0.1
    return 1.9 - 0.036 * dp
