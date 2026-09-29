"""LQR optimal control on the linearized model (phase 6 option).

The linear-quadratic regulator minimizes ∫ (δxᵀ·Q·δx + δuᵀ·R·δu) dt for ẋ = A·x + B·u;
the solution is a state feedback u = −K·δx, with K = R⁻¹·Bᵀ·S and S the solution of the
algebraic Riccati equation.

``LongitudinalLQR`` stabilizes the longitudinal motion (V, α, θ, q) around level
flight with the elevator alone, the throttle staying at its trim value. It makes the aircraft
stable even at the unstable CG position (x_cg = 0.35), but only around its design
point: it is a good local regulator, not a flight control law (see ``fbw.py``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.linalg import solve_continuous_are

from jetsim.aircraft import analysis as an
from jetsim.aircraft import dynamics_6dof as d6
from jetsim.aircraft.instruments import Instruments

Vec = npt.NDArray[np.float64]
CVec = npt.NDArray[Any]  # eigenvalues (complex)
DEG = np.pi / 180


def lqr(A: Vec, B: Vec, Q: Vec, R: Vec) -> tuple[Vec, Vec, CVec]:
    """Continuous-time LQR gain.

    Returns:
        ``(K, S, closed-loop eigenvalues)``.
    """
    S = np.asarray(solve_continuous_are(A, B, Q, R), dtype=np.float64)
    K = np.asarray(np.linalg.solve(R, B.T @ S), dtype=np.float64)
    return K, S, np.linalg.eigvals(A - B @ K)


@dataclass
class LongitudinalLQR:
    """Elevator regulator δe = δe_eq − K·(s − s_eq), s = [V, α, θ, q]."""

    K: Vec  # (1, 4)
    s_eq: Vec  # (4,)
    elevator_eq: float
    throttle_eq: float
    open_loop: CVec  # longitudinal eigenvalues without the regulator
    closed_loop: CVec  # with the regulator
    elevator_limits: tuple[float, float]

    @classmethod
    def design(
        cls,
        model: d6.F16SixDof,
        altitude: float,
        airspeed: float,
        state_weights: tuple[float, float, float, float] = (1e-2, 100.0, 10.0, 10.0),
        control_weight: float = 100.0,
    ) -> LongitudinalLQR:
        """Design at the trim point (altitude, airspeed). Default weights:
        V lightly penalized, α and θ heavily, δe costly (smooth commands)."""
        x, u = model.trim(altitude, airspeed)
        A, B = an.linearize(model, x, u)
        lon = list(an.LONGITUDINAL)
        A_lon = A[np.ix_(lon, lon)]
        B_lon = B[lon, 1:2]  # elevator column
        K, _, closed = lqr(A_lon, B_lon, np.diag(state_weights), np.array([[control_weight]]))
        s = an.reduced_state(x)[lon]
        cs = model.params.control_surfaces["elevator"]
        return cls(
            K=K,
            s_eq=s,
            elevator_eq=float(u[d6.ELEVATOR]),
            throttle_eq=float(u[d6.THROTTLE]),
            open_loop=np.linalg.eigvals(A_lon),
            closed_loop=closed,
            elevator_limits=(cs.min, cs.max),
        )

    def __call__(self, ins: Instruments) -> Vec:
        """6-DOF command [throttle, δe, δa, δr] (ailerons and rudder neutral)."""
        s = np.array([ins.tas, ins.alpha, ins.pitch, ins.q])
        de = self.elevator_eq - float((self.K @ (s - self.s_eq))[0])
        de = min(max(de, self.elevator_limits[0]), self.elevator_limits[1])
        return np.array([self.throttle_eq, de, 0.0, 0.0])
