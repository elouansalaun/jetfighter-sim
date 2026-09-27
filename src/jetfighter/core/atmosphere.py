"""Atmosphère standard internationale (ISA / US Standard Atmosphere 1976), de 0 à 32 km.

Le modèle est défini par couches à gradient thermique constant, en **altitude géopotentielle** H :

=========  ===========  =============  ====================
Couche     H base [m]   T base [K]     gradient [K/m]
=========  ===========  =============  ====================
Tropo.     0            288.15         -0.0065
Tropopause 11 000       216.65          0.0   (isotherme)
Strato. 1  20 000       216.65         +0.001
=========  ===========  =============  ====================

Dans chaque couche :
    - gradient non nul : T = Tb + L·(H − Hb) ;  P = Pb·(T/Tb)^(−g0/(L·R))
    - isotherme        : T = Tb ;               P = Pb·exp(−g0·(H − Hb)/(R·Tb))
puis ρ = P/(R·T) et a = √(γ·R·T).

L'altitude **géométrique** h (celle de l'état de l'avion) est convertie en altitude géopotentielle
par H = r0·h/(r0 + h). L'écart est faible (≈ 60 m à 20 km) mais le coût est nul.

Les altitudes hors du domaine [H_MIN, H_MAX] sont **saturées** (pas d'exception) : un agent RL
qui sort du domaine doit être géré par les conditions de fin d'épisode, pas par un crash du code.

Les fonctions acceptent un scalaire (retour en ``float``) ou un tableau numpy (retour en tableau).
"""

from __future__ import annotations

import math
from typing import NamedTuple, Union

import numpy as np
import numpy.typing as npt

from jetfighter.core.constants import A0, G0, GAMMA_AIR, P0, R_AIR, RHO0, T0

FloatOrArray = Union[float, npt.NDArray[np.float64]]  # noqa: UP007 (alias runtime, py3.10)

EARTH_RADIUS: float = 6_356_766.0
"""Rayon terrestre utilisé pour l'altitude géopotentielle (US Std Atm 1976) [m]."""

H_MIN: float = -1_000.0
"""Altitude minimale du modèle [m] (le gradient troposphérique est prolongé en dessous de 0)."""

H_MAX: float = 32_000.0
"""Altitude géopotentielle maximale du modèle [m]."""

# Couches : altitude géopotentielle de base [m], gradient [K/m]
_LAYER_BASES: tuple[float, ...] = (0.0, 11_000.0, 20_000.0)
_LAYER_LAPSES: tuple[float, ...] = (-0.0065, 0.0, 0.001)


def _build_layers() -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Calcule T et P à la base de chaque couche par continuité à partir du niveau de la mer."""
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
    """Propriétés de l'air à une altitude donnée (unités SI)."""

    temperature: FloatOrArray  # T [K]
    pressure: FloatOrArray  # P [Pa]
    density: FloatOrArray  # ρ [kg/m³]
    speed_of_sound: FloatOrArray  # a [m/s]


# --------------------------------------------------------------------------
# Altitude géométrique <-> géopotentielle
# --------------------------------------------------------------------------
def geopotential_altitude(h: FloatOrArray) -> FloatOrArray:
    """Altitude géométrique h [m] -> altitude géopotentielle H [m]."""
    return EARTH_RADIUS * h / (EARTH_RADIUS + h)


def geometric_altitude(H: FloatOrArray) -> FloatOrArray:
    """Altitude géopotentielle H [m] -> altitude géométrique h [m]."""
    return EARTH_RADIUS * H / (EARTH_RADIUS - H)


# --------------------------------------------------------------------------
# Modèle ISA
# --------------------------------------------------------------------------
def _isa_scalar(H: float) -> tuple[float, float, float, float]:
    """Version scalaire rapide (module ``math``), utilisée dans la boucle de simulation."""
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


def _isa_array(H: npt.NDArray[np.float64]) -> tuple[npt.NDArray[np.float64], ...]:
    """Version vectorisée (numpy) pour les tracés et les analyses."""
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
    """Propriétés de l'atmosphère standard.

    Args:
        h: altitude [m], scalaire ou tableau.
        geometric: ``True`` (défaut) si ``h`` est une altitude géométrique (cas de l'état avion),
            ``False`` si c'est déjà une altitude géopotentielle (cas des tables ISA).

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
# Grandeurs dérivées utiles au modèle avion
# --------------------------------------------------------------------------
def mach_number(V: FloatOrArray, h: FloatOrArray) -> FloatOrArray:
    """Nombre de Mach M = V / a(h), V vitesse air vraie [m/s]."""
    return V / isa(h).speed_of_sound


def dynamic_pressure(V: FloatOrArray, h: FloatOrArray) -> FloatOrArray:
    """Pression dynamique q̄ = ½·ρ(h)·V² [Pa]."""
    return 0.5 * isa(h).density * V * V


def equivalent_airspeed(V: FloatOrArray, h: FloatOrArray) -> FloatOrArray:
    """Vitesse équivalente EAS = V·√(ρ/ρ0) [m/s].

    C'est la vitesse qui donnerait la même pression dynamique au niveau de la mer.
    """
    return V * np.sqrt(isa(h).density / RHO0)


__all__ = [
    "A0",
    "EARTH_RADIUS",
    "H_MAX",
    "H_MIN",
    "P0",
    "RHO0",
    "T0",
    "AtmosphereState",
    "dynamic_pressure",
    "equivalent_airspeed",
    "geometric_altitude",
    "geopotential_altitude",
    "isa",
    "mach_number",
]
