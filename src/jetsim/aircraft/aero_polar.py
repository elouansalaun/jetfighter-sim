"""Aérodynamique simplifiée « polaire » pour le modèle point-masse (phase 2).

    CL(α, M) = CL0 + CLα·s(M)·α
    CD(CL, M) = CD0(M) + k(M)·CL²

* s(M) : facteur de compressibilité sur la pente de portance (hausse en transsonique,
  baisse en supersonique) ;
* CD0(M) : traînée à portance nulle, avec la montée transsonique autour de Mach 1 ;
* k(M) : coefficient de traînée induite, qui augmente en supersonique.

Les tables sont interpolées linéairement en Mach et **saturées** hors de la plage.
Le modèle n'a pas de décrochage : l'incidence est bornée en amont par le limiteur
(α ≤ α_max = 25°, dans la zone où la portance du F-16 reste quasi linéaire).
"""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Sequence

from jetsim.aircraft.params import PolarAeroParams


def interp1(x: float, xs: Sequence[float], ys: Sequence[float]) -> float:
    """Interpolation linéaire scalaire, saturée aux bornes (plus rapide que ``np.interp``)."""
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    i = bisect_right(xs, x) - 1
    x0, x1 = xs[i], xs[i + 1]
    return ys[i] + (ys[i + 1] - ys[i]) * (x - x0) / (x1 - x0)


class PolarAero:
    """Coefficients aérodynamiques de la polaire (voir le module)."""

    def __init__(self, params: PolarAeroParams) -> None:
        self.p = params

    def clalpha(self, mach: float) -> float:
        """Pente de portance CLα(M) [1/rad]."""
        return self.p.CLalpha * interp1(mach, self.p.mach, self.p.clalpha_scale)

    def cl(self, alpha: float, mach: float) -> float:
        """Coefficient de portance."""
        return self.p.CL0 + self.clalpha(mach) * alpha

    def alpha_for_cl(self, cl: float, mach: float) -> float:
        """Incidence [rad] donnant le coefficient de portance ``cl``."""
        return (cl - self.p.CL0) / self.clalpha(mach)

    def cd0(self, mach: float) -> float:
        return interp1(mach, self.p.mach, self.p.cd0)

    def k_induced(self, mach: float) -> float:
        return interp1(mach, self.p.mach, self.p.k_induced)

    def cd(self, cl: float, mach: float) -> float:
        """Coefficient de traînée."""
        return self.cd0(mach) + self.k_induced(mach) * cl * cl

    def max_lift_to_drag(self, mach: float) -> float:
        """Finesse max de la polaire (en négligeant CL0 dans l'optimum) : 1 / (2·√(CD0·k))."""
        return 1.0 / (2.0 * (self.cd0(mach) * self.k_induced(mach)) ** 0.5)
