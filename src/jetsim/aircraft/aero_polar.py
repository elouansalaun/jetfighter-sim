"""Simplified "drag polar" aerodynamics for the point-mass model (phase 2).

    CL(α, M) = CL0 + CLα·s(M)·α
    CD(CL, M) = CD0(M) + k(M)·CL²

* s(M): compressibility factor on the lift slope (rises in the transonic regime,
  drops in the supersonic regime);
* CD0(M): zero-lift drag, with the transonic drag rise around Mach 1;
* k(M): induced-drag coefficient, which grows in the supersonic regime.

The tables are linearly interpolated in Mach and clamped outside their range.
The model has no stall: angle of attack is bounded upstream by the limiter
(α ≤ α_max = 25°, in the region where the F-16's lift stays nearly linear).
"""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Sequence

from jetsim.aircraft.params import PolarAeroParams


def interp1(x: float, xs: Sequence[float], ys: Sequence[float]) -> float:
    """Scalar linear interpolation, clamped at the bounds (faster than ``np.interp``)."""
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    i = bisect_right(xs, x) - 1
    x0, x1 = xs[i], xs[i + 1]
    return ys[i] + (ys[i + 1] - ys[i]) * (x - x0) / (x1 - x0)


class PolarAero:
    """Aerodynamic coefficients of the drag polar (see the module docstring)."""

    def __init__(self, params: PolarAeroParams) -> None:
        self.p = params

    def clalpha(self, mach: float) -> float:
        """Lift-curve slope CLα(M) [1/rad]."""
        return self.p.CLalpha * interp1(mach, self.p.mach, self.p.clalpha_scale)

    def cl(self, alpha: float, mach: float) -> float:
        """Lift coefficient."""
        return self.p.CL0 + self.clalpha(mach) * alpha

    def alpha_for_cl(self, cl: float, mach: float) -> float:
        """Angle of attack [rad] giving the lift coefficient ``cl``."""
        return (cl - self.p.CL0) / self.clalpha(mach)

    def cd0(self, mach: float) -> float:
        return interp1(mach, self.p.mach, self.p.cd0)

    def k_induced(self, mach: float) -> float:
        return interp1(mach, self.p.mach, self.p.k_induced)

    def cd(self, cl: float, mach: float) -> float:
        """Drag coefficient."""
        return self.cd0(mach) + self.k_induced(mach) * cl * cl

    def max_lift_to_drag(self, mach: float) -> float:
        """Max lift-to-drag ratio of the polar (neglecting CL0 at the optimum): 1 / (2·√(CD0·k))."""
        return 1.0 / (2.0 * (self.cd0(mach) * self.k_induced(mach)) ** 0.5)
