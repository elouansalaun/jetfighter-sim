"""Pilote automatique : tenue d'altitude, de cap, de vitesse, virage à inclinaison donnée.

Boucles externes qui produisent une ``HighLevelCommand`` (n_z, taux de roulis, manette),
ensuite exécutée par la boucle interne (``fbw.py``). Mêmes réglages pour 3-DOF et 6-DOF ::

    altitude --(P)--> vitesse verticale --> pente γ --(PI)--> n_z  (+ 1/cos φ en virage)
    cap      --(P)--> inclinaison μ --(PI)--> taux de roulis
    vitesse  --(PI)--> manette

Consignes (``AutopilotTargets``) : chaque tenue peut être désactivée (``None``) ;
``bank`` (inclinaison imposée) a priorité sur ``heading``.

Usage ::

    ap = Autopilot(model)
    ap.reset(x, instruments)
    ap.targets = AutopilotTargets(altitude=5000, heading=math.radians(90), airspeed=250)
    u = ap.controls(instruments, dt)      # commande du modèle
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from jetsim.aircraft import dynamics_3dof as d3
from jetsim.aircraft import dynamics_6dof as d6
from jetsim.aircraft.instruments import Instruments
from jetsim.control.fbw import HighLevelCommand, make_inner_loop
from jetsim.control.pid import PID, wrap_angle
from jetsim.core.constants import G0

Vec = npt.NDArray[np.float64]
DEG = math.pi / 180


@dataclass
class AutopilotTargets:
    altitude: float | None = None  # [m]
    heading: float | None = None  # route χ [rad]
    airspeed: float | None = None  # vitesse vraie [m/s]
    bank: float | None = None  # inclinaison imposée [rad] (prioritaire sur heading)
    throttle: float | None = None  # manette fixe si airspeed est None


@dataclass(frozen=True)
class AutopilotGains:
    k_altitude: float = 0.12  # [1/s] erreur d'altitude -> vitesse verticale
    max_vertical_speed: float = 40.0  # [m/s]
    k_gamma: float = 0.6  # [1/s] erreur de pente -> taux de variation de pente γ̇
    ki_gamma: float = 0.1  # [1/s²]
    max_gamma_rate: float = 0.1  # [rad/s] ≈ ±2 g de correction à 200 m/s
    nz_min: float = -0.5  # confort : le pilote automatique ne pousse pas au-delà
    nz_max: float = 7.5
    k_heading: float = 0.4  # [1/s] erreur de cap -> taux de virage
    max_bank: float = 60.0 * DEG  # en tenue de cap
    max_bank_explicit: float = 85.0 * DEG  # inclinaison imposée (virages serrés)
    k_bank: float = 2.0  # [1/s] erreur d'inclinaison -> taux de roulis
    ki_bank: float = 0.2  # [1/s²] supprime l'écart d'inclinaison permanent en virage serré
    max_roll_rate: float = 90.0 * DEG
    k_speed: float = 0.03  # [1/(m/s)]
    ki_speed: float = 0.01  # [1/(m·s)]


class Autopilot:
    """Pilote automatique complet : boucles externes + boucle interne (voir le module)."""

    def __init__(
        self,
        model: d3.PointMassAircraft | d6.F16SixDof,
        gains: AutopilotGains | None = None,
    ) -> None:
        self.model = model
        self.g = gains or AutopilotGains()
        self.inner = make_inner_loop(model)
        self.targets = AutopilotTargets()
        self.gamma_pid = PID(self.g.k_gamma, self.g.ki_gamma, out_min=-self.g.max_gamma_rate,
                             out_max=self.g.max_gamma_rate)  # fmt: skip
        self.speed_pid = PID(self.g.k_speed, self.g.ki_speed, out_min=0.0, out_max=1.0)
        self.bank_pid = PID(self.g.k_bank, self.g.ki_bank, out_min=-self.g.max_roll_rate,
                            out_max=self.g.max_roll_rate, integral_zone=5 * DEG)  # fmt: skip
        self.command = HighLevelCommand()

    def reset(self, x: Vec, ins: Instruments) -> None:
        """Engage le pilote automatique sur l'état courant, sans à-coup."""
        self.inner.reset(x)
        self.gamma_pid.reset(0.0)
        self.bank_pid.reset(0.0)
        if isinstance(self.model, d6.F16SixDof):
            thr = self.model.engine.throttle_for_power(float(x[d6.POWER]))
        else:
            thr = float(x[d3.POWER])
        self.speed_pid.reset(output=min(max(thr, 0.0), 1.0))
        self.command = HighLevelCommand(nz=ins.nz, roll_rate=0.0, throttle=0.5)

    # ------------------------------------------------------------------
    def high_level(self, ins: Instruments, dt: float) -> HighLevelCommand:
        """Boucles externes : consignes -> (n_z, taux de roulis, manette)."""
        g, tg = self.g, self.targets
        v = max(ins.tas, 1.0)
        cos_bank = max(math.cos(ins.bank), 0.12)  # compensation de virage jusqu'à ≈ 83°

        # Tangage : altitude -> vitesse verticale -> pente -> n_z
        if tg.altitude is not None:
            vs = min(max(g.k_altitude * (tg.altitude - ins.altitude), -g.max_vertical_speed),
                     g.max_vertical_speed)  # fmt: skip
            gamma_cmd = math.asin(min(max(vs / v, -0.8), 0.8))
            gamma_rate = self.gamma_pid(gamma_cmd - ins.gamma, dt)
            correction = gamma_rate * v / G0  # n_z supplémentaire pour courber la trajectoire
            nz = math.cos(ins.gamma) / cos_bank + correction
        else:
            nz = 1.0 / cos_bank
        nz = min(max(nz, g.nz_min), g.nz_max)

        # Roulis : cap -> inclinaison -> taux de roulis
        if tg.bank is not None:
            bank_cmd = min(max(tg.bank, -g.max_bank_explicit), g.max_bank_explicit)
        elif tg.heading is not None:
            turn_rate = g.k_heading * wrap_angle(tg.heading - ins.course)
            bank_cmd = min(max(math.atan(turn_rate * v / G0), -g.max_bank), g.max_bank)
        else:
            bank_cmd = 0.0
        roll_rate = self.bank_pid(wrap_angle(bank_cmd - ins.bank), dt)

        # Moteur
        if tg.airspeed is not None:
            throttle = self.speed_pid(tg.airspeed - ins.tas, dt)
        elif tg.throttle is not None:
            throttle = tg.throttle
        else:
            throttle = self.command.throttle
        self.command = HighLevelCommand(nz=nz, roll_rate=roll_rate, throttle=throttle)
        return self.command

    def controls(self, ins: Instruments, dt: float) -> Vec:
        """Commande du modèle (consignes -> boucles externes -> boucle interne)."""
        return self.inner(ins, self.high_level(ins, dt), dt)
