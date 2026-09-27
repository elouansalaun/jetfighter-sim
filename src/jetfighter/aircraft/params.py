"""Paramètres avion chargés depuis un fichier YAML (``configs/aircraft/*.yaml``).

Le YAML est en unités SI, sauf les clés suffixées ``_deg`` / ``_deg_s`` qui sont converties
ici en radians. Le reste du code ne manipule que les dataclasses ci-dessous : changer d'avion
revient à changer de fichier YAML.

Usage ::

    from jetfighter.aircraft.params import load_aircraft
    p = load_aircraft("f16")          # cherche configs/aircraft/f16.yaml
    p = load_aircraft("mon/fichier.yaml")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import yaml

from jetfighter.core.constants import DEG_TO_RAD

CONFIG_DIR: Path = Path(__file__).resolve().parents[3] / "configs" / "aircraft"
"""Dossier des configurations avion (installation en mode éditable)."""


@dataclass(frozen=True)
class Geometry:
    wing_area: float  # S [m²]
    wing_span: float  # b [m]
    mean_chord: float  # c̄ [m]

    @property
    def aspect_ratio(self) -> float:
        return self.wing_span**2 / self.wing_area


@dataclass(frozen=True)
class MassProperties:
    mass: float  # [kg]
    Ixx: float  # [kg·m²]
    Iyy: float
    Izz: float
    Ixz: float

    @property
    def inertia_tensor(self) -> npt.NDArray[np.float64]:
        """Tenseur d'inertie en repère corps (plan de symétrie xz : Ixy = Iyz = 0)."""
        return np.array(
            [
                [self.Ixx, 0.0, -self.Ixz],
                [0.0, self.Iyy, 0.0],
                [-self.Ixz, 0.0, self.Izz],
            ]
        )


@dataclass(frozen=True)
class PropulsionParams:
    thrust_idle_sl: float  # [N]
    thrust_mil_sl: float  # [N]
    thrust_max_sl: float  # [N]
    mil_power: float  # position de manette du plein gaz sec, dans ]0, 1[
    density_exponent: float
    ram_factor_mil: float
    ram_factor_max: float
    ram_limit: float  # poussée max / poussée statique au sol, même régime
    engine_time_constant: float  # [s]


@dataclass(frozen=True)
class PolarAeroParams:
    CL0: float
    CLalpha: float  # [1/rad]
    mach: tuple[float, ...]
    clalpha_scale: tuple[float, ...]
    cd0: tuple[float, ...]
    k_induced: tuple[float, ...]

    def __post_init__(self) -> None:
        n = len(self.mach)
        for name in ("clalpha_scale", "cd0", "k_induced"):
            if len(getattr(self, name)) != n:
                raise ValueError(f"aero_polar.{name} doit avoir {n} valeurs (une par Mach).")
        if any(b <= a for a, b in zip(self.mach, self.mach[1:], strict=False)):
            raise ValueError("aero_polar.mach doit être strictement croissant.")


@dataclass(frozen=True)
class FlightLimits:
    n_max: float
    n_min: float
    alpha_max: float  # [rad]
    alpha_min: float  # [rad]
    roll_rate_max: float  # [rad/s]
    mach_max: float
    ceiling: float  # [m]


@dataclass(frozen=True)
class PointMassResponse:
    alpha_time_constant: float  # [s]
    alpha_rate_max: float  # [rad/s]
    roll_time_constant: float  # [s]


@dataclass(frozen=True)
class SurfaceLimits:
    min: float  # [rad]
    max: float  # [rad]
    rate_max: float  # [rad/s]


@dataclass(frozen=True)
class SixDofParams:
    tables_path: Path  # fichier des tables aéro + moteur (chemin absolu)
    xcg_ref: float  # centrage de référence des tables [fraction de c̄]
    xcg: float  # centrage réel [fraction de c̄]
    actuator_time_constant: float  # [s]


@dataclass(frozen=True)
class AircraftParams:
    name: str
    geometry: Geometry
    mass: MassProperties
    propulsion: PropulsionParams
    aero_polar: PolarAeroParams
    limits: FlightLimits
    point_mass_response: PointMassResponse
    control_surfaces: dict[str, SurfaceLimits] = field(default_factory=dict)
    six_dof: SixDofParams | None = None

    @classmethod
    def from_dict(cls, d: dict[str, Any], base_dir: Path | None = None) -> AircraftParams:
        """Construit les paramètres ; ``base_dir`` sert à résoudre les fichiers annexes."""
        lim = d["limits"]
        pmr = d["point_mass_response"]
        ap = d["aero_polar"]
        return cls(
            name=d.get("name", "avion"),
            geometry=Geometry(**d["geometry"]),
            mass=MassProperties(**d["mass"]),
            propulsion=PropulsionParams(**d["propulsion"]),
            aero_polar=PolarAeroParams(
                CL0=ap["CL0"],
                CLalpha=ap["CLalpha"],
                mach=tuple(ap["mach"]),
                clalpha_scale=tuple(ap["clalpha_scale"]),
                cd0=tuple(ap["cd0"]),
                k_induced=tuple(ap["k_induced"]),
            ),
            limits=FlightLimits(
                n_max=lim["n_max"],
                n_min=lim["n_min"],
                alpha_max=lim["alpha_max_deg"] * DEG_TO_RAD,
                alpha_min=lim["alpha_min_deg"] * DEG_TO_RAD,
                roll_rate_max=lim["roll_rate_max_deg_s"] * DEG_TO_RAD,
                mach_max=lim["mach_max"],
                ceiling=lim["ceiling"],
            ),
            point_mass_response=PointMassResponse(
                alpha_time_constant=pmr["alpha_time_constant"],
                alpha_rate_max=pmr["alpha_rate_max_deg_s"] * DEG_TO_RAD,
                roll_time_constant=pmr["roll_time_constant"],
            ),
            control_surfaces={
                k: SurfaceLimits(
                    min=v["min_deg"] * DEG_TO_RAD,
                    max=v["max_deg"] * DEG_TO_RAD,
                    rate_max=v["rate_max_deg_s"] * DEG_TO_RAD,
                )
                for k, v in d.get("control_surfaces", {}).items()
            },
            six_dof=_six_dof(d.get("six_dof"), base_dir),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> AircraftParams:
        path = Path(path)
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(yaml.safe_load(f), base_dir=path.resolve().parent)


def _six_dof(d: dict[str, Any] | None, base_dir: Path | None) -> SixDofParams | None:
    if d is None:
        return None
    tables = Path(d["tables_file"])
    if not tables.is_absolute():
        tables = (base_dir or CONFIG_DIR) / tables
    return SixDofParams(
        tables_path=tables,
        xcg_ref=float(d["xcg_ref"]),
        xcg=float(d["xcg"]),
        actuator_time_constant=float(d["actuator_time_constant"]),
    )


def load_aircraft(name_or_path: str | Path = "f16") -> AircraftParams:
    """Charge un avion par nom (``configs/aircraft/<nom>.yaml``) ou par chemin de fichier."""
    path = Path(name_or_path)
    if not path.suffix:
        path = CONFIG_DIR / f"{name_or_path}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Configuration avion introuvable : {path}")
    return AircraftParams.from_yaml(path)


# --------------------------------------------------------------------------
# Enveloppe de vol (conditions de fin d'épisode)
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class EnvelopeLimits:
    """Limites au-delà desquelles un vol est considéré comme perdu (fin d'épisode)."""

    min_altitude: float  # [m] sol
    max_altitude: float  # [m]
    min_airspeed: float  # [m/s] vitesse vraie
    max_mach: float
    n_max: float  # facteur de charge structural
    n_min: float
    alpha_stall: float  # [rad]
    stall_duration: float  # [s] durée tolérée au-delà de alpha_stall
    beta_max: float  # [rad]

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> EnvelopeLimits:
        return cls(
            min_altitude=float(d["min_altitude"]),
            max_altitude=float(d["max_altitude"]),
            min_airspeed=float(d["min_airspeed"]),
            max_mach=float(d["max_mach"]),
            n_max=float(d["n_max"]),
            n_min=float(d["n_min"]),
            alpha_stall=float(d["alpha_stall_deg"]) * DEG_TO_RAD,
            stall_duration=float(d["stall_duration"]),
            beta_max=float(d["beta_max_deg"]) * DEG_TO_RAD,
        )


def load_envelope(name_or_path: str | Path = "f16", *, six_dof: bool = False) -> EnvelopeLimits:
    """Limites d'enveloppe de la configuration avion (section ``envelope``).

    Avec ``six_dof=True``, les valeurs de ``envelope.six_dof_overrides`` remplacent les
    valeurs communes (le modèle 6-DOF a un domaine de validité plus étroit).
    """
    path = Path(name_or_path)
    if not path.suffix:
        path = CONFIG_DIR / f"{name_or_path}.yaml"
    with open(path, encoding="utf-8") as f:
        env = dict(yaml.safe_load(f)["envelope"])
    overrides = env.pop("six_dof_overrides", {}) or {}
    if six_dof:
        env.update(overrides)
    return EnvelopeLimits.from_dict(env)
