"""Surveillance de l'enveloppe de vol : détecte les situations qui terminent un épisode.

Les vérifications, dans l'ordre de priorité :

1. ``NUMERICAL``  : état non fini (NaN/inf) — divergence numérique ;
2. ``GROUND``     : altitude sous le sol ;
3. ``OVERLOAD``   : facteur de charge hors des limites structurales ;
4. ``CEILING``    : altitude au-dessus du maximum ;
5. ``LOW_SPEED``  : vitesse vraie sous le minimum ;
6. ``OVERSPEED``  : Mach au-dessus du maximum (ou du domaine de validité du modèle) ;
7. ``SIDESLIP``   : dérapage hors du domaine des tables ;
8. ``STALL``      : incidence au-delà de l'incidence de décrochage **pendant plus de**
   ``stall_duration`` secondes (un dépassement bref est toléré : c'est une manœuvre,
   pas une perte de contrôle).

Usage ::

    monitor = EnvelopeMonitor(load_envelope("f16", six_dof=True))
    monitor.reset()
    violation = monitor.check(instruments, dt=0.1, state=x)
    if violation is not None:
        print(monitor.message)
"""

from __future__ import annotations

import math
from enum import Enum

import numpy as np
import numpy.typing as npt

from jetfighter.aircraft.instruments import Instruments
from jetfighter.aircraft.params import EnvelopeLimits


class Violation(Enum):
    NUMERICAL = "divergence numérique"
    GROUND = "collision avec le sol"
    OVERLOAD = "surcharge structurale"
    CEILING = "plafond dépassé"
    LOW_SPEED = "vitesse trop faible"
    OVERSPEED = "survitesse"
    SIDESLIP = "dérapage excessif"
    STALL = "décrochage prolongé"


class EnvelopeMonitor:
    """Vérifie l'enveloppe à chaque pas ; garde en mémoire la durée passée en décrochage."""

    def __init__(self, limits: EnvelopeLimits) -> None:
        self.limits = limits
        self.stall_time = 0.0
        self.message = ""

    def reset(self) -> None:
        self.stall_time = 0.0
        self.message = ""

    def check(
        self,
        ins: Instruments,
        dt: float,
        state: npt.NDArray[np.float64] | None = None,
    ) -> Violation | None:
        """Renvoie la violation détectée, ou ``None`` si le vol continue.

        Args:
            ins: instruments **vrais** (pas les mesures bruitées).
            dt: temps écoulé depuis la vérification précédente [s].
            state: vecteur d'état brut, contrôlé pour les NaN (optionnel).
        """
        lim = self.limits
        values = (ins.altitude, ins.tas, ins.mach, ins.nz, ins.alpha, ins.beta)
        if (state is not None and not np.all(np.isfinite(state))) or not all(
            math.isfinite(v) for v in values
        ):
            return self._fail(Violation.NUMERICAL, "état non fini (NaN ou infini)")
        if ins.altitude < lim.min_altitude:
            return self._fail(Violation.GROUND, f"altitude {ins.altitude:.0f} m")
        if not lim.n_min <= ins.nz <= lim.n_max:
            return self._fail(
                Violation.OVERLOAD,
                f"n = {ins.nz:.1f} g hors de [{lim.n_min:.0f}, {lim.n_max:.0f}] g",
            )
        if ins.altitude > lim.max_altitude:
            return self._fail(Violation.CEILING, f"altitude {ins.altitude:.0f} m")
        if ins.tas < lim.min_airspeed:
            return self._fail(Violation.LOW_SPEED, f"V = {ins.tas:.0f} m/s")
        if ins.mach > lim.max_mach:
            return self._fail(Violation.OVERSPEED, f"Mach {ins.mach:.3f} > {lim.max_mach:.2f}")
        if abs(ins.beta) > lim.beta_max:
            return self._fail(Violation.SIDESLIP, f"β = {math.degrees(ins.beta):.0f}°")

        if ins.alpha > lim.alpha_stall:
            self.stall_time += dt
            if self.stall_time > lim.stall_duration + 1e-9:  # tolérance d'arrondi
                return self._fail(
                    Violation.STALL,
                    f"α = {math.degrees(ins.alpha):.0f}° depuis {self.stall_time:.1f} s",
                )
        else:
            self.stall_time = 0.0
        return None

    def _fail(self, violation: Violation, detail: str) -> Violation:
        self.message = f"{violation.value} : {detail}"
        return violation
