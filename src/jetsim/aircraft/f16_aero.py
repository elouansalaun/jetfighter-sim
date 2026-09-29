"""Aérodynamique tabulée du F-16 (modèle Stevens & Lewis, annexe A).

Coefficients en **axes corps** (x avant, y droite, z bas), fonctions de l'incidence α,
du dérapage β, des gouvernes (δe profondeur, δa ailerons, δr direction) et des vitesses
angulaires (p, q, r) via les dérivées d'amortissement :

    CX = CX(α, δe) + (c̄q/2V)·CXq(α)
    CY = −0.02β + 0.021·δa/20 + 0.086·δr/30 + (b/2V)·(CYr·r + CYp·p)
    CZ = CZ(α)·(1 − (β/57.3)²) − 0.19·δe/25 + (c̄q/2V)·CZq(α)
    Cl = Cl(α, β) + ΔCl_δa(α, β)·δa/20 + ΔCl_δr(α, β)·δr/30 + (b/2V)·(Clr·r + Clp·p)
    Cm = Cm(α, δe) + (c̄q/2V)·Cmq(α) + CZ·(x_cg,ref − x_cg)
    Cn = Cn(α, β) + ΔCn_δa·δa/20 + ΔCn_δr·δr/30 + (b/2V)·(Cnr·r + Cnp·p) − CY·(x_cg,ref − x_cg)·c̄/b

(angles α, β, δ en degrés dans ces formules, comme dans les tables.)

Les deux derniers termes transportent les moments du centre de gravité de référence
(35 % de c̄) vers le centrage réel ``x_cg`` : reculer le centrage rend l'avion moins stable.

Interface : angles en **radians**, vitesses en m/s, conversion en degrés en interne.
Domaine des tables : α ∈ [−10°, 45°], |β| ≤ 30°, |δe| ≤ 24° ; au-delà, extrapolation
linéaire (comme le modèle d'origine). Pas d'effet du Mach.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import yaml

from jetsim.core.constants import RAD_TO_DEG

Array = npt.NDArray[np.float64]


# --------------------------------------------------------------------------
# Interpolation sur grilles régulières (linéaire, extrapolation par le dernier intervalle)
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class UniformGrid:
    start: float
    step: float
    count: int

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> UniformGrid:
        return cls(float(d["start"]), float(d["step"]), int(d["count"]))

    @property
    def values(self) -> Array:
        return np.asarray(self.start + self.step * np.arange(self.count), dtype=np.float64)

    def locate(self, x: float) -> tuple[int, float]:
        """Indice de cellule ``i`` ∈ [0, n−2] et fraction ``f`` (hors de [0, 1] = extrapolation)."""
        s = (x - self.start) / self.step
        i = math.floor(s)
        if i < 0:
            i = 0
        elif i > self.count - 2:
            i = self.count - 2
        return i, s - i


def lerp1(table: Array, grid: UniformGrid, x: float) -> float:
    i, f = grid.locate(x)
    return float(table[i] + f * (table[i + 1] - table[i]))


def lerp2(table: Array, gx: UniformGrid, x: float, gy: UniformGrid, y: float) -> float:
    """Interpolation bilinéaire de ``table[ix, iy]``."""
    i, fx = gx.locate(x)
    j, fy = gy.locate(y)
    t00, t01 = table[i, j], table[i, j + 1]
    t10, t11 = table[i + 1, j], table[i + 1, j + 1]
    a = t00 + fx * (t10 - t00)
    b = t01 + fx * (t11 - t01)
    return float(a + fy * (b - a))


# --------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class F16AeroTables:
    alpha: UniformGrid
    elevator: UniformGrid
    beta_abs: UniformGrid
    beta: UniformGrid
    cx: Array  # [α, δe]
    cz_alpha: Array  # [α]
    cz_elevator_gain: float
    cy_beta: float
    cy_aileron: float
    cy_rudder: float
    cm: Array  # [α, δe]
    cl: Array  # [α, |β|]
    cn: Array  # [α, |β|]
    dlda: Array  # [α, β]
    dldr: Array
    dnda: Array
    dndr: Array
    damping: Array  # [α, 9] : CXq, CYr, CYp, CZq, Clr, Clp, Cmq, Cnr, Cnp

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> F16AeroTables:
        g, a = d["grids"], d["aero"]

        def arr(key: str) -> Array:
            return np.asarray(a[key], dtype=np.float64)

        tables = cls(
            alpha=UniformGrid.from_dict(g["alpha_deg"]),
            elevator=UniformGrid.from_dict(g["elevator_deg"]),
            beta_abs=UniformGrid.from_dict(g["beta_abs_deg"]),
            beta=UniformGrid.from_dict(g["beta_deg"]),
            cx=arr("cx"),
            cz_alpha=arr("cz_alpha"),
            cz_elevator_gain=float(a["cz_elevator_gain"]),
            cy_beta=float(a["cy_beta"]),
            cy_aileron=float(a["cy_aileron"]),
            cy_rudder=float(a["cy_rudder"]),
            cm=arr("cm"),
            cl=arr("cl"),
            cn=arr("cn"),
            dlda=arr("dlda"),
            dldr=arr("dldr"),
            dnda=arr("dnda"),
            dndr=arr("dndr"),
            damping=arr("damping"),
        )
        tables.validate()
        return tables

    def validate(self) -> None:
        na, ne, nba, nb = (
            self.alpha.count,
            self.elevator.count,
            self.beta_abs.count,
            self.beta.count,
        )
        expected = {
            "cx": (na, ne),
            "cz_alpha": (na,),
            "cm": (na, ne),
            "cl": (na, nba),
            "cn": (na, nba),
            "dlda": (na, nb),
            "dldr": (na, nb),
            "dnda": (na, nb),
            "dndr": (na, nb),
            "damping": (na, 9),
        }
        for name, shape in expected.items():
            if getattr(self, name).shape != shape:
                raise ValueError(
                    f"Table {name} : forme {getattr(self, name).shape}, {shape} attendue."
                )


def load_tables_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        data: dict[str, Any] = yaml.safe_load(f)
    return data


# --------------------------------------------------------------------------
# Coefficients
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class AeroCoefficients:
    CX: float
    CY: float
    CZ: float
    Cl: float
    Cm: float
    Cn: float


class F16Aero:
    """Coefficients aérodynamiques en axes corps (voir le module)."""

    def __init__(
        self, tables: F16AeroTables, wing_span: float, mean_chord: float, xcg_ref: float = 0.35
    ) -> None:
        self.t = tables
        self.b = wing_span
        self.cbar = mean_chord
        self.xcg_ref = xcg_ref

    def coefficients(
        self,
        alpha: float,
        beta: float,
        elevator: float,
        aileron: float,
        rudder: float,
        p: float,
        q: float,
        r: float,
        airspeed: float,
        xcg: float,
    ) -> AeroCoefficients:
        """Coefficients totaux. Angles et vitesses angulaires en rad et rad/s, V en m/s."""
        t = self.t
        a = alpha * RAD_TO_DEG
        b = beta * RAD_TO_DEG
        el = elevator * RAD_TO_DEG
        dail = aileron * RAD_TO_DEG / 20.0
        drdr = rudder * RAD_TO_DEG / 30.0
        abs_b = abs(b)
        sign_b = 1.0 if b > 0 else (-1.0 if b < 0 else 0.0)

        # Coefficients statiques
        cx = lerp2(t.cx, t.alpha, a, t.elevator, el)
        cy = t.cy_beta * b + t.cy_aileron * dail + t.cy_rudder * drdr
        cz = lerp1(t.cz_alpha, t.alpha, a) * (1.0 - (b / 57.3) ** 2) + t.cz_elevator_gain * (
            el / 25.0
        )
        cl = (
            lerp2(t.cl, t.alpha, a, t.beta_abs, abs_b) * sign_b
            + lerp2(t.dlda, t.alpha, a, t.beta, b) * dail
            + lerp2(t.dldr, t.alpha, a, t.beta, b) * drdr
        )
        cm = lerp2(t.cm, t.alpha, a, t.elevator, el)
        cn = (
            lerp2(t.cn, t.alpha, a, t.beta_abs, abs_b) * sign_b
            + lerp2(t.dnda, t.alpha, a, t.beta, b) * dail
            + lerp2(t.dndr, t.alpha, a, t.beta, b) * drdr
        )

        # Amortissement (vitesses angulaires adimensionnées)
        i, f = t.alpha.locate(a)
        d = t.damping[i] + f * (t.damping[i + 1] - t.damping[i])
        v_safe = max(airspeed, 1.0)
        cq = self.cbar * q / (2.0 * v_safe)
        b2v = self.b / (2.0 * v_safe)
        cx += cq * d[0]
        cy += b2v * (d[1] * r + d[2] * p)
        cz += cq * d[3]
        cl += b2v * (d[4] * r + d[5] * p)
        dx_cg = self.xcg_ref - xcg
        cm += cq * d[6] + cz * dx_cg
        cn += b2v * (d[7] * r + d[8] * p) - cy * dx_cg * self.cbar / self.b
        return AeroCoefficients(cx, cy, cz, cl, cm, cn)
