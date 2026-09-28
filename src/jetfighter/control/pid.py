"""Briques de commande : PID avec anti-emballement, filtres du premier ordre.

PID « parallèle » ::

    u = Kp·e + Ki·∫e dt − Kd·dy/dt        (dérivée sur la MESURE, filtrée)

* dérivée sur la mesure plutôt que sur l'erreur : pas de coup de bélier quand la consigne
  change brutalement ;
* **anti-emballement** (anti-windup) par blocage : l'intégrateur est gelé tant que la
  sortie est saturée et que l'erreur pousse dans le sens de la saturation ;
* **intégration conditionnelle** (``integral_zone``) : on n'intègre que près de la consigne,
  pour éviter que l'intégrateur se charge pendant les grands transitoires ;
* ``reset(output=…)`` initialise l'intégrateur pour que la première sortie vaille ``output``
  (transfert sans à-coup quand on engage le pilote automatique en vol).
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
    derivative_tau: float = 0.05  # [s] filtre de la dérivée
    integral_zone: float = math.inf  # n'intègre que si |erreur| ≤ cette valeur
    integral: float = field(default=0.0, init=False)
    _last_y: float | None = field(default=None, init=False)
    _d_filt: float = field(default=0.0, init=False)

    def reset(self, output: float = 0.0, error: float = 0.0) -> None:
        """Remet à zéro ; l'intégrale est choisie pour que la sortie vaille ``output``."""
        self._last_y = None
        self._d_filt = 0.0
        self.integral = (output - self.kp * error) / self.ki if self.ki else 0.0

    def __call__(self, error: float, dt: float, measurement: float | None = None,
                 gain: float = 1.0) -> float:  # fmt: skip
        """Sortie du régulateur.

        Args:
            error: consigne − mesure.
            dt: pas de temps [s].
            measurement: mesure (pour le terme dérivé) ; ``None`` = pas de terme dérivé.
            gain: facteur multiplicatif global (programmation de gains).
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
    """Filtre passe-haut du 1er ordre : laisse passer les variations, efface le continu.

    Sert à l'amortisseur de lacet : on veut amortir les oscillations de r, pas s'opposer au
    taux de lacet permanent d'un virage coordonné.
    """

    tau: float
    _state: float = field(default=0.0, init=False)

    def reset(self, value: float = 0.0) -> None:
        self._state = value

    def __call__(self, value: float, dt: float) -> float:
        self._state += (value - self._state) * dt / (self.tau + dt)
        return value - self._state


def wrap_angle(angle: float) -> float:
    """Angle dans [−π, π[ (erreurs de cap)."""
    return (angle + math.pi) % (2 * math.pi) - math.pi
