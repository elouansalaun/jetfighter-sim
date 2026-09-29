"""Point-mass model performance: flight envelope, climb, turn.

These computations are used to validate the model (comparison with the public
orders of magnitude of the F-16) and to choose realistic initial conditions for learning.

All functions assume quasi-steady symmetric flight at full power
(afterburner) unless stated otherwise.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from jetsim.aircraft.dynamics_3dof import PointMassAircraft, TrimError
from jetsim.core.atmosphere import isa_scalar
from jetsim.core.constants import G0


@dataclass(frozen=True)
class TurnPerformance:
    load_factor: float  # n
    turn_rate: float  # ω [rad/s]
    radius: float  # R [m]


def speed_of_sound(h: float) -> float:
    return isa_scalar(h)[3]


def excess_power(ac: PointMassAircraft, h: float, v: float, power: float = 1.0) -> float:
    """Specific excess power Ps [m/s] in level flight (n = 1) at the given power.

    Ps = V·(T·cos α − D)/(m·g): achievable climb rate at constant speed (or, divided
    by V/g, achievable acceleration in level flight). Returns ``-inf`` if level flight is
    impossible (angle of attack out of limits).
    """
    _, _, rho, a = isa_scalar(h)
    mach = v / a
    qs = 0.5 * rho * v * v * ac.S
    thrust = ac.engine.thrust(power, rho, mach)
    a_lo, a_hi = ac.params.limits.alpha_min, ac.params.limits.alpha_max

    def normal_eq(alpha: float) -> float:
        return qs * ac.aero.cl(alpha, mach) + thrust * math.sin(alpha) - ac.weight

    if normal_eq(a_lo) * normal_eq(a_hi) > 0:
        return -math.inf
    alpha = brentq(normal_eq, a_lo, a_hi, xtol=1e-10)
    drag = qs * ac.aero.cd(ac.aero.cl(alpha, mach), mach)
    return v * (thrust * math.cos(alpha) - drag) / ac.weight


def level_speed_range(
    ac: PointMassAircraft, h: float, power: float = 1.0, mach_max: float = 3.0
) -> tuple[float, float] | None:
    """Min and max speeds [m/s] in steady level flight (Ps ≥ 0), or ``None`` if impossible."""
    a = speed_of_sound(h)
    machs = np.linspace(0.05, mach_max, 600)
    ps = np.array([excess_power(ac, h, m * a, power) for m in machs])
    ok = np.flatnonzero(ps >= 0)
    if ok.size == 0:
        return None

    def f(m: float) -> float:
        return excess_power(ac, h, m * a, power)

    i_lo, i_hi = ok[0], ok[-1]
    v_min = machs[i_lo] if i_lo == 0 or not np.isfinite(ps[i_lo - 1]) else None
    if v_min is None:
        v_min = brentq(f, machs[i_lo - 1], machs[i_lo])
    v_max = machs[i_hi] if i_hi == machs.size - 1 else brentq(f, machs[i_hi], machs[i_hi + 1])
    return float(v_min) * a, float(v_max) * a


def max_rate_of_climb(ac: PointMassAircraft, h: float, power: float = 1.0) -> float:
    """Max rate of climb [m/s] at altitude ``h`` (max of Ps over speed)."""
    a = speed_of_sound(h)
    return max(excess_power(ac, h, m * a, power) for m in np.linspace(0.3, 2.5, 221))


def ceiling(ac: PointMassAircraft, climb_rate: float = 0.5, power: float = 1.0) -> float:
    """Ceiling [m]: altitude where the max rate of climb drops to ``climb_rate``
    (0.5 m/s ≈ 100 ft/min, the service ceiling definition)."""
    return brentq(lambda h: max_rate_of_climb(ac, h, power) - climb_rate, 0.0, 25_000.0, xtol=1.0)


def sustained_turn(ac: PointMassAircraft, h: float, v: float) -> TurnPerformance | None:
    """Tightest sustained level turn (thrust = drag) at (h, V)."""
    n_max = ac.params.limits.n_max

    def feasible(n: float) -> bool:
        try:
            ac.trim(h, v, load_factor=n)
        except TrimError:
            return False
        return True

    if not feasible(1.0):
        return None
    if feasible(n_max):
        n = n_max
    else:
        lo, hi = 1.0, n_max
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if feasible(mid) else (lo, mid)
        n = lo
    return _turn(n, v)


def instantaneous_turn(ac: PointMassAircraft, h: float, v: float) -> TurnPerformance:
    """Max instantaneous turn (α_max or n_max), without holding speed."""
    _, _, rho, a = isa_scalar(h)
    mach = v / a
    qs = 0.5 * rho * v * v * ac.S
    alpha = ac.params.limits.alpha_max
    thrust = ac.engine.thrust(1.0, rho, mach)
    n = (qs * ac.aero.cl(alpha, mach) + thrust * math.sin(alpha)) / ac.weight
    return _turn(min(n, ac.params.limits.n_max), v)


def _turn(n: float, v: float) -> TurnPerformance:
    if n <= 1.0:
        return TurnPerformance(n, 0.0, math.inf)
    omega = G0 * math.sqrt(n * n - 1.0) / v
    return TurnPerformance(n, omega, v / omega)
