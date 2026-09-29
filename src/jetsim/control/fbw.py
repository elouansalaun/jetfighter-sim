"""Boucle interne (« commandes de vol électriques ») : commandes de haut niveau -> gouvernes.

Tous les contrôleurs de haut niveau du projet (pilote automatique, heuristiques
d'évitement, et l'agent RL en mode hiérarchique) parlent le même langage ::

    HighLevelCommand(nz=…, roll_rate=…, throttle=…)

* ``nz``        : facteur de charge demandé [g] (1 = palier ; 9 = ressource à 9 g) ;
* ``roll_rate`` : taux de roulis demandé [rad/s] (> 0 : vers la droite) ;
* ``throttle``  : manette [0, 1].

``make_inner_loop(model)`` renvoie la boucle interne adaptée au modèle, qui convertit ces
consignes en commandes du modèle.

6-DOF — lois de pilotage (gains dans ``FbwGains``, multipliés par q̄_ref/q̄ pour garder la
même dynamique quand la pression dynamique varie) :

* **tangage** : PI sur l'erreur de n_z + amortissement en q ; δe < 0 fait cabrer ;
* **limiteur d'incidence** : dès que α approche α_max, la consigne de n_z est réduite à ce qui
  maintient α ≤ α_max (idem côté négatif) — l'avion ne peut pas être mis en décrochage ;
* **limiteur de facteur de charge** : n_z ∈ [n_min, n_max] ;
* **roulis** : PI sur l'erreur de taux de roulis ;
* **lacet** : dérapage ramené à zéro (β → 0) + amortisseur de lacet (r filtré passe-haut, pour
  ne pas contrer le taux de lacet normal d'un virage).

Réglage (par simulation sur 0–9 km, 150–300 m/s) : échelon 1 -> 4 g en 0.4–0.9 s,
dépassement ≤ 5 % (17 % à 9 km) ; échelon de 90°/s en roulis en 0.2–0.3 s, |β| < 1°.
Ces lois **stabilisent aussi l'avion au centrage instable** (x_cg = 0.35 ou 0.40).

3-DOF : n_z -> α commandée par inversion de la polaire + correction intégrale ; le taux de
roulis et la manette sont transmis directement (le modèle point-masse a déjà ses propres
lois de réponse et limiteurs).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from jetsim.aircraft import dynamics_3dof as d3
from jetsim.aircraft import dynamics_6dof as d6
from jetsim.aircraft.instruments import Instruments
from jetsim.control.pid import PID, Washout

Vec = npt.NDArray[np.float64]
DEG = math.pi / 180


@dataclass(frozen=True)
class HighLevelCommand:
    nz: float = 1.0  # [g]
    roll_rate: float = 0.0  # [rad/s]
    throttle: float = 0.5  # [0-1]


@dataclass(frozen=True)
class FbwGains:
    """Gains des commandes de vol 6-DOF (au point de référence q̄_ref)."""

    q_ref: float = 18_000.0  # [Pa] ≈ 200 m/s à 3000 m
    schedule_min: float = 0.3
    schedule_max: float = 3.0
    k_nz: float = 3.0 * DEG  # [rad/g]
    ki_nz: float = 6.0 * DEG  # [rad/(g·s)]
    k_q: float = 0.4  # [rad/(rad/s)]
    k_p: float = 0.35  # [rad/(rad/s)]
    ki_p: float = 1.0  # [rad/rad]
    k_beta: float = 3.0  # [rad/rad]
    k_r: float = 0.5  # [rad/(rad/s)]
    yaw_washout: float = 1.0  # [s]
    alpha_max: float = 25.0 * DEG
    alpha_min: float = -8.0 * DEG
    k_alpha: float = 0.5  # [g/°] limiteur d'incidence
    k_alpha_q: float = 0.05  # [g/(°/s)] amortissement du limiteur
    nz_max: float = 9.0
    nz_min: float = -3.0
    k_nz_q: float = 0.05  # [g/(°/s)] anticipation du limiteur de n_z (dépassement)
    nz_slew_rate: float = 20.0  # [g/s] vitesse max de variation de la consigne de n_z
    damping_exponent: float = 0.5  # amortissement en q programmé en (q̄_ref/q̄)^0.5
    roll_rate_max: float = 240.0 * DEG


class FlyByWire:
    """Boucle interne du modèle 6-DOF (voir le module)."""

    def __init__(self, model: d6.F16SixDof, gains: FbwGains | None = None) -> None:
        self.model = model
        self.g = gains or FbwGains()
        cs = model.params.control_surfaces
        self._de = (cs["elevator"].min, cs["elevator"].max)
        self._da = (cs["aileron"].min, cs["aileron"].max)
        self._dr = (cs["rudder"].min, cs["rudder"].max)
        self.pitch = PID(self.g.k_nz, self.g.ki_nz, out_min=-self._de[1], out_max=-self._de[0])
        self.roll = PID(self.g.k_p, self.g.ki_p, out_min=-self._da[1], out_max=-self._da[0])
        self.yaw_filter = Washout(self.g.yaw_washout)
        self.nz_command = 1.0  # consigne effectivement suivie (après limiteurs)
        self._nz_ref = 1.0  # consigne après limitation de vitesse de variation

    def reset(self, x: Vec) -> None:
        """Engage la boucle sans à-coup : les intégrateurs reprennent les gouvernes actuelles."""
        self.pitch.reset(output=-x[d6.DE])
        self.roll.reset(output=-x[d6.DA])
        self.yaw_filter.reset(x[d6.R])
        self.nz_command = 1.0
        self._nz_ref = 1.0

    def limited_nz(self, ins: Instruments, nz_cmd: float) -> float:
        """Consigne de n_z après limiteurs de facteur de charge et d'incidence."""
        g = self.g
        q_deg = ins.q / DEG
        # Limiteur de facteur de charge avec anticipation par la vitesse de tangage : sans
        # elle, un échelon 1 -> −3 g à grande vitesse dépasse −4 g (limite structurale)
        # (l'anticipation ne fait que resserrer les limites, jamais les élargir)
        nz = min(max(nz_cmd, g.nz_min - g.k_nz_q * min(q_deg, 0.0)),
                 g.nz_max - g.k_nz_q * max(q_deg, 0.0))  # fmt: skip
        upper = ins.nz + g.k_alpha * (g.alpha_max - ins.alpha) / DEG - g.k_alpha_q * q_deg
        lower = ins.nz + g.k_alpha * (g.alpha_min - ins.alpha) / DEG - g.k_alpha_q * q_deg
        return min(max(nz, lower), upper)

    def __call__(self, ins: Instruments, cmd: HighLevelCommand, dt: float) -> Vec:
        g = self.g
        sched = min(max(g.q_ref / max(ins.dynamic_pressure, 1.0), g.schedule_min),
                    g.schedule_max)  # fmt: skip
        # consigne à vitesse de variation limitée : une inversion brutale (+9 -> −3 g) ne
        # doit pas emballer la boucle (dépassement mesuré jusqu'à −5 g sans ce filtre)
        step = g.nz_slew_rate * dt
        self._nz_ref = min(max(cmd.nz, self._nz_ref - step), self._nz_ref + step)
        self.nz_command = self.limited_nz(ins, self._nz_ref)
        damping = g.k_q * sched**g.damping_exponent
        nose_up = self.pitch(self.nz_command - ins.nz, dt, gain=sched) - damping * ins.q
        p_cmd = min(max(cmd.roll_rate, -g.roll_rate_max), g.roll_rate_max)
        roll_right = self.roll(p_cmd - ins.p, dt, gain=sched)
        rudder = sched * (g.k_r * self.yaw_filter(ins.r, dt) - g.k_beta * ins.beta)
        u = np.empty(4)
        u[d6.THROTTLE] = min(max(cmd.throttle, 0.0), 1.0)
        u[d6.ELEVATOR] = min(max(-nose_up, self._de[0]), self._de[1])
        u[d6.AILERON] = min(max(-roll_right, self._da[0]), self._da[1])
        u[d6.RUDDER] = min(max(rudder, self._dr[0]), self._dr[1])
        return u


class PointMassInnerLoop:
    """Boucle interne du modèle 3-DOF : n_z -> α commandée (inversion de la polaire + PI)."""

    def __init__(self, model: d3.PointMassAircraft, ki: float = 1.0 * DEG) -> None:
        self.model = model
        lim = model.params.limits
        self.correction = PID(0.0, ki, out_min=-5 * DEG, out_max=5 * DEG)
        self._alpha_bounds = (lim.alpha_min, lim.alpha_max)
        self.nz_command = 1.0

    def reset(self, x: Vec) -> None:
        self.correction.reset(0.0)
        self.nz_command = 1.0

    def __call__(self, ins: Instruments, cmd: HighLevelCommand, dt: float) -> Vec:
        m = self.model
        qs = max(ins.dynamic_pressure, 1.0) * m.S
        lim = m.params.limits
        self.nz_command = min(max(cmd.nz, lim.n_min), lim.n_max)
        alpha_ff = m.aero.alpha_for_cl(self.nz_command * m.weight / qs, ins.mach)
        alpha = alpha_ff + self.correction(self.nz_command - ins.nz, dt)
        u = np.empty(3)
        u[d3.THROTTLE] = min(max(cmd.throttle, 0.0), 1.0)
        u[d3.ALPHA_CMD] = min(max(alpha, self._alpha_bounds[0]), self._alpha_bounds[1])
        u[d3.ROLL_RATE_CMD] = cmd.roll_rate
        return u


InnerLoop = FlyByWire | PointMassInnerLoop


def make_inner_loop(model: d3.PointMassAircraft | d6.F16SixDof) -> InnerLoop:
    if isinstance(model, d6.F16SixDof):
        return FlyByWire(model)
    if isinstance(model, d3.PointMassAircraft):
        return PointMassInnerLoop(model)
    raise TypeError(f"Modèle non pris en charge : {type(model).__name__}")
