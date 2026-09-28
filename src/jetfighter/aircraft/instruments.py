"""Tableau de bord commun aux modèles 3-DOF et 6-DOF.

``read_instruments(model, x)`` renvoie les mêmes grandeurs quel que soit le modèle, ce qui
permettra aux environnements RL (phase 7) de construire leurs observations sans savoir
quel modèle tourne derrière.

Conventions :

* **attitude** (roulis φ, assiette θ, cap ψ) : angles d'Euler 3-2-1 du **repère corps** ;
* **trajectoire** (pente γ, route χ) : direction du vecteur vitesse ;
* **inclinaison μ** (*bank*) : rotation du plan de portance autour du vecteur vitesse.
  C'est elle qui fixe l'équilibre d'un virage (n·cos μ = cos γ en palier). Elle diffère
  de la gîte φ du fuselage dès que l'incidence est grande ;
* **facteurs de charge** n = f/g, avec f la force spécifique (ce que mesure un
  accéléromètre : efforts aéro + poussée divisés par la masse, sans la gravité), en axes
  corps. ``nz`` est compté **positif vers le haut** (1 en palier, 9 en ressource à 9 g) ;
* **énergie spécifique** E = h + V²/2g et **puissance spécifique excédentaire** Ps = Ė.

Pour le 3-DOF, l'attitude et les vitesses angulaires corps sont reconstruites exactement
à partir du repère vent et de l'incidence (hypothèse du modèle : dérapage nul).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields

import numpy as np
import numpy.typing as npt

from jetfighter.aircraft import dynamics_3dof as d3
from jetfighter.aircraft import dynamics_6dof as d6
from jetfighter.core.atmosphere import calibrated_airspeed, equivalent_airspeed
from jetfighter.core.frames import (
    dcm_from_quat,
    dcm_wind_to_body,
    euler_from_dcm,
    quat_conjugate,
    quat_multiply,
)

Vec = npt.NDArray[np.float64]


@dataclass(frozen=True)
class Instruments:
    """Grandeurs observables de l'avion (SI, radians)."""

    # Position
    north: float
    east: float
    altitude: float
    # Anémométrie
    tas: float  # vitesse vraie [m/s]
    cas: float  # vitesse corrigée [m/s]
    eas: float  # vitesse équivalente [m/s]
    mach: float
    dynamic_pressure: float  # [Pa]
    # Variomètre et trajectoire
    vertical_speed: float  # ḣ [m/s]
    gamma: float  # pente
    course: float  # route χ
    bank: float  # inclinaison μ du vecteur portance autour de la vitesse
    # Attitude (repère corps)
    roll: float  # φ
    pitch: float  # θ
    heading: float  # ψ
    # Aérodynamique
    alpha: float
    beta: float
    # Gyromètres (repère corps)
    p: float
    q: float
    r: float
    # Accéléromètres (facteurs de charge, repère corps)
    nx: float
    ny: float
    nz: float  # positif vers le haut
    # Énergie
    specific_energy: float  # E [m]
    specific_excess_power: float  # Ps [m/s]
    airspeed_rate: float  # V̇ [m/s²]
    # Moteur
    thrust: float  # [N]
    power: float  # [0-1]

    def to_dict(self) -> dict[str, float]:
        return asdict(self)

    def vector(self, names: tuple[str, ...]) -> Vec:
        """Sous-ensemble des grandeurs, dans l'ordre demandé (pour les observations RL)."""
        return np.array([getattr(self, n) for n in names], dtype=np.float64)


INSTRUMENT_NAMES: tuple[str, ...] = tuple(f.name for f in fields(Instruments))


def _airspeeds(tas: float, h: float) -> tuple[float, float]:
    return calibrated_airspeed(tas, h), float(equivalent_airspeed(tas, h))


def read_instruments(model: d3.PointMassAircraft | d6.F16SixDof, x: Vec) -> Instruments:
    """Lit le tableau de bord pour l'état ``x`` du modèle donné."""
    if isinstance(model, d6.F16SixDof):
        return _read_6dof(model, x)
    if isinstance(model, d3.PointMassAircraft):
        return _read_3dof(model, x)
    raise TypeError(f"Modèle non pris en charge : {type(model).__name__}")


# --------------------------------------------------------------------------
# 6-DOF
# --------------------------------------------------------------------------
def _read_6dof(model: d6.F16SixDof, x: Vec) -> Instruments:
    fd = model.flight_data(x)
    force, _, _, _, _ = model.forces_moments(x)
    dx = model.derivatives(0.0, x, np.array([0.0, x[d6.DE], x[d6.DA], x[d6.DR]]))
    uvw, duvw = x[d6.VEL], dx[d6.VEL]
    v = fd.airspeed
    v_dot = float(uvw @ duvw) / v if v > 0 else 0.0
    g = model.g
    specific_force = force / model.mass
    cas, eas = _airspeeds(v, fd.altitude)
    c_nw = dcm_from_quat(x[d6.QUAT]) @ dcm_wind_to_body(fd.alpha, fd.beta)
    bank, _, _ = euler_from_dcm(c_nw)
    return Instruments(
        north=fd.north,
        east=fd.east,
        altitude=fd.altitude,
        tas=v,
        cas=cas,
        eas=eas,
        mach=fd.mach,
        dynamic_pressure=fd.dynamic_pressure,
        vertical_speed=fd.climb_rate,
        gamma=fd.gamma,
        course=fd.course,
        bank=bank,
        roll=fd.roll,
        pitch=fd.pitch,
        heading=fd.yaw,
        alpha=fd.alpha,
        beta=fd.beta,
        p=fd.p,
        q=fd.q,
        r=fd.r,
        nx=specific_force[0] / g,
        ny=specific_force[1] / g,
        nz=-specific_force[2] / g,
        specific_energy=fd.altitude + v * v / (2 * g),
        specific_excess_power=fd.climb_rate + v * v_dot / g,
        airspeed_rate=v_dot,
        thrust=fd.thrust,
        power=fd.power,
    )


# --------------------------------------------------------------------------
# 3-DOF : reconstruction de l'attitude corps à partir du repère vent et de α
# --------------------------------------------------------------------------
def _read_3dof(model: d3.PointMassAircraft, x: Vec) -> Instruments:
    fd = model.flight_data(x)
    alpha = fd.alpha
    u = np.array([x[d3.POWER], alpha, x[d3.ROLL_RATE]])
    dx = model.derivatives(0.0, x, u)
    # Rotation du repère vent : ω_w = 2·(q_w* ⊗ q̇_w), partie vectorielle
    q_w = x[d3.QUAT]
    omega_w = 2.0 * quat_multiply(quat_conjugate(q_w), dx[d3.QUAT])[1:]
    c_bw = dcm_wind_to_body(alpha, 0.0)
    c_nb = dcm_from_quat(q_w) @ c_bw.T
    roll, pitch, heading = euler_from_dcm(c_nb)
    # ω_corps = ω_vent + [0, α̇, 0]. α̇ dépend de la commande (inconnue du tableau de bord) :
    # on l'omet, l'écart est nul en régime établi et borné par la dynamique d'incidence.
    omega_b = c_bw @ omega_w

    ca, sa = math.cos(alpha), math.sin(alpha)
    f_w = np.array(
        [(fd.thrust * ca - fd.drag) / model.mass, 0.0, -(fd.lift + fd.thrust * sa) / model.mass]
    )
    f_b = c_bw @ f_w
    g = model.weight / model.mass
    cas, eas = _airspeeds(fd.airspeed, fd.altitude)
    return Instruments(
        north=fd.north,
        east=fd.east,
        altitude=fd.altitude,
        tas=fd.airspeed,
        cas=cas,
        eas=eas,
        mach=fd.mach,
        dynamic_pressure=fd.dynamic_pressure,
        vertical_speed=fd.climb_rate,
        gamma=fd.gamma,
        course=fd.heading,
        bank=fd.bank,
        roll=roll,
        pitch=pitch,
        heading=heading,
        alpha=alpha,
        beta=0.0,
        p=float(omega_b[0]),
        q=float(omega_b[1]),
        r=float(omega_b[2]),
        nx=f_b[0] / g,
        ny=f_b[1] / g,
        nz=-f_b[2] / g,
        specific_energy=fd.specific_energy,
        specific_excess_power=fd.specific_excess_power,
        airspeed_rate=fd.airspeed_rate,
        thrust=fd.thrust,
        power=fd.power,
    )
