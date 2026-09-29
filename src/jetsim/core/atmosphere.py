"""International Standard Atmosphere (ISA / US Standard Atmosphere 1976), from 0 to 32 km.

The model is defined by layers with a constant lapse rate, in geopotential altitude H:

=========  ===========  =============  ====================
Layer      H base [m]   T base [K]     lapse [K/m]
=========  ===========  =============  ====================
Tropo.     0            288.15         -0.0065
Tropopause 11 000       216.65          0.0   (isothermal)
Strato. 1  20 000       216.65         +0.001
=========  ===========  =============  ====================

Within each layer:
    - non-zero lapse : T = Tb + L·(H − Hb) ;  P = Pb·(T/Tb)^(−g0/(L·R))
    - isothermal     : T = Tb ;               P = Pb·exp(−g0·(H − Hb)/(R·Tb))
then ρ = P/(R·T) and a = √(γ·R·T).

The geometric altitude h (the one in the aircraft state) is converted to geopotential altitude
with H = r0·h/(r0 + h). The difference is small (≈ 60 m at 20 km) but costs nothing.

Altitudes outside the domain [H_MIN, H_MAX] are clamped (no exception): an RL agent
leaving the domain must be handled by the episode termination conditions, not by a code crash.

Functions accept a scalar (returns a ``float``) or a numpy array (returns an array).
"""

from __future__ import annotations

import math
from typing import NamedTuple, Union

import numpy as np
import numpy.typing as npt
from scipy.optimize import brentq

from jetsim.core.constants import A0, G0, GAMMA_AIR, P0, R_AIR, RHO0, T0

FloatOrArray = Union[float, npt.NDArray[np.float64]]  # noqa: UP007 (alias runtime, py3.10)

EARTH_RADIUS: float = 6_356_766.0
"""Earth radius used for geopotential altitude (US Std Atm 1976) [m]."""

H_MIN: float = -1_000.0
"""Minimum model altitude [m] (the tropospheric lapse rate is extended below 0)."""

H_MAX: float = 32_000.0
"""Maximum geopotential altitude of the model [m]."""

# Layers: base geopotential altitude [m], lapse rate [K/m]
_LAYER_BASES: tuple[float, ...] = (0.0, 11_000.0, 20_000.0)
_LAYER_LAPSES: tuple[float, ...] = (-0.0065, 0.0, 0.001)


def _build_layers() -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Compute T and P at the base of each layer by continuity from sea level."""
    temps = [T0]
    press = [P0]
    for i in range(1, len(_LAYER_BASES)):
        dh = _LAYER_BASES[i] - _LAYER_BASES[i - 1]
        tb, pb, lapse = temps[-1], press[-1], _LAYER_LAPSES[i - 1]
        if lapse == 0.0:
            t = tb
            p = pb * math.exp(-G0 * dh / (R_AIR * tb))
        else:
            t = tb + lapse * dh
            p = pb * (t / tb) ** (-G0 / (lapse * R_AIR))
        temps.append(t)
        press.append(p)
    return tuple(temps), tuple(press)


_LAYER_TEMPS, _LAYER_PRESSURES = _build_layers()


class AtmosphereState(NamedTuple):
    """Air properties at a given altitude (SI units)."""

    temperature: FloatOrArray  # T [K]
    pressure: FloatOrArray  # P [Pa]
    density: FloatOrArray  # ρ [kg/m³]
    speed_of_sound: FloatOrArray  # a [m/s]


# --------------------------------------------------------------------------
# Geometric <-> geopotential altitude
# --------------------------------------------------------------------------
def geopotential_altitude(h: FloatOrArray) -> FloatOrArray:
    """Geometric altitude h [m] -> geopotential altitude H [m]."""
    return EARTH_RADIUS * h / (EARTH_RADIUS + h)


def geometric_altitude(H: FloatOrArray) -> FloatOrArray:
    """Geopotential altitude H [m] -> geometric altitude h [m]."""
    return EARTH_RADIUS * H / (EARTH_RADIUS - H)


# --------------------------------------------------------------------------
# ISA model
# --------------------------------------------------------------------------
def _isa_scalar(H: float) -> tuple[float, float, float, float]:
    """Fast scalar version (``math`` module), used in the simulation loop."""
    H = min(max(H, H_MIN), H_MAX)
    i = 2 if H >= _LAYER_BASES[2] else (1 if H >= _LAYER_BASES[1] else 0)
    hb, tb, pb, lapse = _LAYER_BASES[i], _LAYER_TEMPS[i], _LAYER_PRESSURES[i], _LAYER_LAPSES[i]
    if lapse == 0.0:
        t = tb
        p = pb * math.exp(-G0 * (H - hb) / (R_AIR * tb))
    else:
        t = tb + lapse * (H - hb)
        p = pb * (t / tb) ** (-G0 / (lapse * R_AIR))
    rho = p / (R_AIR * t)
    a = math.sqrt(GAMMA_AIR * R_AIR * t)
    return t, p, rho, a


def isa_scalar(h: float) -> tuple[float, float, float, float]:
    """Fast path for the simulation loop: scalar geometric altitude [m].

    Returns:
        ``(T, P, ρ, a)`` as floats, without building an ``AtmosphereState``.
    """
    return _isa_scalar(EARTH_RADIUS * h / (EARTH_RADIUS + h))


def _isa_array(H: npt.NDArray[np.float64]) -> tuple[npt.NDArray[np.float64], ...]:
    """Vectorized (numpy) version for plots and analyses."""
    H = np.clip(H, H_MIN, H_MAX)
    t = np.empty_like(H)
    p = np.empty_like(H)
    upper = (*_LAYER_BASES[1:], np.inf)
    for i, (hb, tb, pb, lapse) in enumerate(
        zip(_LAYER_BASES, _LAYER_TEMPS, _LAYER_PRESSURES, _LAYER_LAPSES, strict=True)
    ):
        m = (H >= hb) & (H < upper[i]) if i > 0 else H < upper[i]
        dh = H[m] - hb
        if lapse == 0.0:
            t[m] = tb
            p[m] = pb * np.exp(-G0 * dh / (R_AIR * tb))
        else:
            t[m] = tb + lapse * dh
            p[m] = pb * (t[m] / tb) ** (-G0 / (lapse * R_AIR))
    rho = p / (R_AIR * t)
    a = np.sqrt(GAMMA_AIR * R_AIR * t)
    return t, p, rho, a


def isa(h: FloatOrArray, *, geometric: bool = True) -> AtmosphereState:
    """Standard atmosphere properties.

    Args:
        h: altitude [m], scalar or array.
        geometric: ``True`` (default) if ``h`` is a geometric altitude (aircraft state),
            ``False`` if it is already a geopotential altitude (ISA tables).

    Returns:
        ``AtmosphereState(temperature, pressure, density, speed_of_sound)``.
    """
    if np.ndim(h) == 0:
        hf = float(h)  # type: ignore[arg-type]
        H = EARTH_RADIUS * hf / (EARTH_RADIUS + hf) if geometric else hf
        return AtmosphereState(*_isa_scalar(H))
    arr = np.asarray(h, dtype=np.float64)
    H_arr = EARTH_RADIUS * arr / (EARTH_RADIUS + arr) if geometric else arr
    return AtmosphereState(*_isa_array(H_arr))


# --------------------------------------------------------------------------
# Derived quantities used by the aircraft model
# --------------------------------------------------------------------------
def mach_number(V: FloatOrArray, h: FloatOrArray) -> FloatOrArray:
    """Mach number M = V / a(h), V being the true airspeed [m/s]."""
    return V / isa(h).speed_of_sound


def dynamic_pressure(V: FloatOrArray, h: FloatOrArray) -> FloatOrArray:
    """Dynamic pressure q̄ = ½·ρ(h)·V² [Pa]."""
    return 0.5 * isa(h).density * V * V


def equivalent_airspeed(V: FloatOrArray, h: FloatOrArray) -> FloatOrArray:
    """Equivalent airspeed EAS = V·√(ρ/ρ0) [m/s].

    It is the speed that would give the same dynamic pressure at sea level.
    """
    return V * np.sqrt(isa(h).density / RHO0)


# --------------------------------------------------------------------------
# Airspeeds (what an airspeed indicator shows)
# --------------------------------------------------------------------------
def impact_pressure(mach: float, pressure: float) -> float:
    """Impact pressure qc = P_total − P_static measured by a Pitot tube [Pa].

    * subsonic: qc = P·[(1 + 0.2·M²)^3.5 − 1] (isentropic);
    * supersonic: normal shock in front of the tube (Rayleigh formula),
      qc = P·[166.92158·M⁷ / (7·M² − 1)^2.5 − 1].
    """
    m2 = mach * mach
    if mach <= 1.0:
        return pressure * ((1.0 + 0.2 * m2) ** 3.5 - 1.0)
    return pressure * (166.92158 * mach**7 / (7.0 * m2 - 1.0) ** 2.5 - 1.0)


def _mach_from_impact_ratio(ratio: float) -> float:
    """Mach such that qc/P = ``ratio`` (inverse of ``impact_pressure``)."""
    sub = math.sqrt(5.0 * ((ratio + 1.0) ** (2.0 / 7.0) - 1.0))
    if sub <= 1.0:
        return sub
    return float(brentq(lambda m: impact_pressure(m, 1.0) - ratio, 1.0, 20.0, xtol=1e-14))


def calibrated_airspeed(V: float, h: float) -> float:
    """Calibrated airspeed CAS [m/s]: the speed that would give the same impact pressure at
    sea level in the standard atmosphere. Equal to the true airspeed at sea level;
    lower at altitude. It is the speed the aircraft "feels" (lift, limits)."""
    _, pressure, _, a = isa_scalar(h)
    qc = impact_pressure(abs(V) / a, pressure)
    return A0 * _mach_from_impact_ratio(qc / P0)


def true_airspeed_from_calibrated(cas: float, h: float) -> float:
    """Inverse of ``calibrated_airspeed``: true airspeed [m/s] for a CAS at altitude h."""
    _, pressure, _, a = isa_scalar(h)
    qc = impact_pressure(cas / A0, P0)
    return a * _mach_from_impact_ratio(qc / pressure)


__all__ = [
    "A0",
    "EARTH_RADIUS",
    "H_MAX",
    "H_MIN",
    "P0",
    "RHO0",
    "T0",
    "AtmosphereState",
    "calibrated_airspeed",
    "dynamic_pressure",
    "equivalent_airspeed",
    "geometric_altitude",
    "geopotential_altitude",
    "impact_pressure",
    "isa",
    "isa_scalar",
    "mach_number",
    "true_airspeed_from_calibrated",
]
