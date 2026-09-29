"""Singularity-free "3-DOF" point-mass aircraft model (loops are possible).

Principle
---------
The aircraft is a point mass. Its orientation is described only by the **wind frame**
(x_w along the velocity, z_w opposite to lift), represented by a quaternion
``q_w`` (wind -> NED). The Euler angles of this frame are directly the course χ,
the flight-path angle γ and the bank angle μ (bank about the velocity vector).

The classical equations in (V, γ, χ) ::

    V̇ = (T·cos α − D)/m − g·sin γ
    γ̇ = [(L + T·sin α)·cos μ − m·g·cos γ] / (m·V)
    χ̇ = (L + T·sin α)·sin μ / (m·V·cos γ)

are singular at γ = ±90°. We therefore integrate the rotation of the wind frame instead:
with a = [a_x, a_y, a_z] the acceleration (thrust + aero + gravity) in wind axes,

    V̇ = a_x ;   ω_w = [p_w, −a_z/V, a_y/V]  ;  q̇_w = ½·q_w ⊗ ω_w

where p_w is the roll rate about the velocity vector (commanded by the pilot). This is
strictly equivalent to the equations above away from γ = ±90°, and remains valid beyond.

Assumptions: symmetric flight (sideslip β = 0), thrust along the fuselage axis, constant
mass, flat Earth, no wind.

State (11): ``[x_N, y_E, h, V, q0, q1, q2, q3, α, p_w, P]``
    horizontal NED position and altitude [m], airspeed [m/s], wind -> NED quaternion,
    angle of attack [rad], roll rate about the velocity [rad/s], engine power [0-1].

Controls (3): ``[throttle, α_cmd, p_cmd]``
    throttle ∈ [0, 1] (> ``mil_power`` = afterburner), commanded angle of attack [rad],
    commanded roll rate [rad/s].

Control dynamics (what the aircraft + its fly-by-wire flight controls would do):
    * α follows α_cmd with a first-order lag (τ_α), rate limited, with a **limiter**:
      α_min ≤ α ≤ α_max and n_min ≤ n ≤ n_max (load-factor limiter);
    * p_w follows p_cmd with a first-order lag (τ_p), bounded to ± p_max;
    * engine power follows the throttle with a first-order lag.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.optimize import brentq

from jetsim.aircraft.aero_polar import PolarAero
from jetsim.aircraft.params import AircraftParams
from jetsim.aircraft.propulsion import SimpleTurbofan
from jetsim.core.atmosphere import isa_scalar
from jetsim.core.constants import G0
from jetsim.core.frames import euler_from_quat, quat_derivative, quat_from_euler
from jetsim.core.integrators import DEFAULT_DT, rk4_step

Vec = npt.NDArray[np.float64]

# State indices
PN, PE, H, V, Q0, Q1, Q2, Q3, ALPHA, ROLL_RATE, POWER = range(11)
N_STATE = 11
QUAT = slice(Q0, Q3 + 1)

# Control indices
THROTTLE, ALPHA_CMD, ROLL_RATE_CMD = range(3)
N_CONTROL = 3

V_EPS: float = 1.0
"""Speed floor [m/s] in divisions by V (avoids blowing up at zero speed)."""


class TrimError(RuntimeError):
    """No equilibrium exists (e.g. insufficient thrust, angle of attack out of limits)."""


@dataclass(frozen=True)
class FlightData:
    """Instrument quantities computed from the state (SI units, radians)."""

    north: float
    east: float
    altitude: float
    airspeed: float  # V (true) [m/s]
    mach: float
    dynamic_pressure: float  # q̄ [Pa]
    alpha: float  # angle of attack
    gamma: float  # flight-path angle
    heading: float  # course χ
    bank: float  # bank angle μ
    roll_rate: float  # p_w
    load_factor: float  # n = (L + T·sin α)/(m·g)
    lift: float  # [N]
    drag: float  # [N]
    thrust: float  # [N]
    power: float  # engine power [0-1]
    airspeed_rate: float  # V̇ [m/s²]
    climb_rate: float  # ḣ [m/s]
    turn_rate: float  # χ̇ [rad/s]
    specific_energy: float  # E = h + V²/2g [m]
    specific_excess_power: float  # Ps = Ė = V·(T·cos α − D)/(m·g) [m/s]


class PointMassAircraft:
    """Point-mass aircraft (see the module docstring for equations and conventions)."""

    def __init__(self, params: AircraftParams) -> None:
        self.params = params
        self.aero = PolarAero(params.aero_polar)
        self.engine = SimpleTurbofan(params.propulsion)
        self.mass = params.mass.mass
        self.weight = self.mass * G0
        self.S = params.geometry.wing_area
        lim = params.limits
        rsp = params.point_mass_response
        self._alpha_bounds = (lim.alpha_min, lim.alpha_max)
        self._n_bounds = (lim.n_min, lim.n_max)
        self._p_max = lim.roll_rate_max
        self._tau_alpha = rsp.alpha_time_constant
        self._alpha_rate_max = rsp.alpha_rate_max
        self._tau_p = rsp.roll_time_constant

    # ------------------------------------------------------------------
    # State construction
    # ------------------------------------------------------------------
    @staticmethod
    def make_state(
        *,
        altitude: float,
        airspeed: float,
        gamma: float = 0.0,
        heading: float = 0.0,
        bank: float = 0.0,
        alpha: float = 0.0,
        power: float = 0.5,
        roll_rate: float = 0.0,
        north: float = 0.0,
        east: float = 0.0,
    ) -> Vec:
        """Build a state vector from readable quantities (angles in radians)."""
        x = np.zeros(N_STATE)
        x[PN], x[PE], x[H], x[V] = north, east, altitude, airspeed
        x[QUAT] = quat_from_euler(bank, gamma, heading)
        x[ALPHA], x[ROLL_RATE], x[POWER] = alpha, roll_rate, power
        return x

    # ------------------------------------------------------------------
    # Forces
    # ------------------------------------------------------------------
    def _aero_propulsion(
        self, h: float, v: float, alpha: float, power: float
    ) -> tuple[float, float, float, float, float, float]:
        """Return (mach, q̄, L, D, T, a_sound)."""
        _, _, rho, a_sound = isa_scalar(h)
        mach = v / a_sound
        qbar = 0.5 * rho * v * v
        cl = self.aero.cl(alpha, mach)
        cd = self.aero.cd(cl, mach)
        qs = qbar * self.S
        thrust = self.engine.thrust(power, rho, mach)
        return mach, qbar, qs * cl, qs * cd, thrust, a_sound

    def alpha_limits(self, qbar: float, mach: float, thrust: float) -> tuple[float, float]:
        """Allowed angle-of-attack range: AoA bounds ∩ load-factor bounds.

        n(α) ≈ [q̄S·(CL0 + CLα·α) + T·α] / (m·g) is linearized in α to invert the limit.
        """
        a_lo, a_hi = self._alpha_bounds
        n_lo, n_hi = self._n_bounds
        qs = qbar * self.S
        slope = qs * self.aero.clalpha(mach) + thrust
        if slope > 1e-9:
            offset = qs * self.aero.p.CL0
            a_hi = min(a_hi, (n_hi * self.weight - offset) / slope)
            a_lo = max(a_lo, (n_lo * self.weight - offset) / slope)
        return a_lo, max(a_lo, a_hi)

    # ------------------------------------------------------------------
    # Dynamics
    # ------------------------------------------------------------------
    def derivatives(self, t: float, x: Vec, u: Vec) -> Vec:
        """ẋ = f(t, x, u)."""
        h, v = x[H], x[V]
        q0, q1, q2, q3 = x[Q0], x[Q1], x[Q2], x[Q3]
        alpha, p_w, power = x[ALPHA], x[ROLL_RATE], x[POWER]

        mach, qbar, lift, drag, thrust, _ = self._aero_propulsion(h, v, alpha, power)

        # Gravity in wind axes: g · (3rd row of C_nw)
        g_x = G0 * 2.0 * (q1 * q3 - q0 * q2)
        g_y = G0 * 2.0 * (q2 * q3 + q0 * q1)
        g_z = G0 * (q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3)

        ca, sa = math.cos(alpha), math.sin(alpha)
        a_x = (thrust * ca - drag) / self.mass + g_x
        a_y = g_y
        a_z = -(lift + thrust * sa) / self.mass + g_z

        v_safe = max(v, V_EPS)
        omega_w = (p_w, -a_z / v_safe, a_y / v_safe)

        dx = np.empty(N_STATE)
        # Position: velocity = V · x_w (1st column of C_nw)
        dx[PN] = v * (q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3)
        dx[PE] = v * 2.0 * (q1 * q2 + q0 * q3)
        dx[H] = -v * 2.0 * (q1 * q3 - q0 * q2)
        dx[V] = a_x
        dx[QUAT] = quat_derivative(x[QUAT], omega_w)

        # Controls: limited angle of attack, roll, engine
        a_lo, a_hi = self.alpha_limits(qbar, mach, thrust)
        alpha_target = min(max(u[ALPHA_CMD], a_lo), a_hi)
        alpha_rate = (alpha_target - alpha) / self._tau_alpha
        dx[ALPHA] = min(max(alpha_rate, -self._alpha_rate_max), self._alpha_rate_max)
        p_target = min(max(u[ROLL_RATE_CMD], -self._p_max), self._p_max)
        dx[ROLL_RATE] = (p_target - p_w) / self._tau_p
        dx[POWER] = self.engine.power_rate(power, u[THROTTLE])
        return dx

    @staticmethod
    def post_step(x: Vec) -> Vec:
        """Renormalize the quaternion after an integration step (modifies ``x`` in place)."""
        q = x[QUAT]
        x[QUAT] = q / math.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3])
        return x

    def step(self, x: Vec, u: Vec, dt: float = DEFAULT_DT) -> Vec:
        """Advance one physics step (RK4 + renormalization). Returns a new state."""
        return self.post_step(rk4_step(self.derivatives, 0.0, x, np.asarray(u, float), dt))

    # ------------------------------------------------------------------
    # Instruments
    # ------------------------------------------------------------------
    def flight_data(self, x: Vec) -> FlightData:
        """Compute the observable flight quantities from the state."""
        h, v, alpha, power = x[H], x[V], x[ALPHA], x[POWER]
        mach, qbar, lift, drag, thrust, _ = self._aero_propulsion(h, v, alpha, power)
        mu, gamma, chi = euler_from_quat(x[QUAT])
        dx = self.derivatives(0.0, x, np.array([power, alpha, x[ROLL_RATE]]))

        # χ̇ = (v_N·a_E − v_E·a_N)/(v_N² + v_E²), with a_NED = V̇·x_w + V·ẋ_w
        v_h2 = dx[PN] ** 2 + dx[PE] ** 2
        if v_h2 > 1e-6:
            xw_n_rate, xw_e_rate = _xw_horizontal_rate(x[QUAT], dx[QUAT])
            a_n = dx[V] * dx[PN] / max(v, V_EPS) + v * xw_n_rate
            a_e = dx[V] * dx[PE] / max(v, V_EPS) + v * xw_e_rate
            turn_rate = (dx[PN] * a_e - dx[PE] * a_n) / v_h2
        else:
            turn_rate = 0.0

        excess = v * (thrust * math.cos(alpha) - drag) / self.weight
        return FlightData(
            north=x[PN],
            east=x[PE],
            altitude=h,
            airspeed=v,
            mach=mach,
            dynamic_pressure=qbar,
            alpha=alpha,
            gamma=gamma,
            heading=chi,
            bank=mu,
            roll_rate=x[ROLL_RATE],
            load_factor=(lift + thrust * math.sin(alpha)) / self.weight,
            lift=lift,
            drag=drag,
            thrust=thrust,
            power=power,
            airspeed_rate=dx[V],
            climb_rate=dx[H],
            turn_rate=turn_rate,
            specific_energy=h + v * v / (2.0 * G0),
            specific_excess_power=excess,
        )

    # ------------------------------------------------------------------
    # Equilibrium (trim)
    # ------------------------------------------------------------------
    def trim(
        self, altitude: float, airspeed: float, gamma: float = 0.0, load_factor: float | None = None
    ) -> tuple[float, float]:
        """Trim angle of attack and power at constant speed.

        Solves ``V̇ = 0`` and "load factor = n":
            T·cos α − D − m·g·sin γ = 0
            L + T·sin α − n·m·g = 0
        with n = cos γ by default (straight flight), or n > 1 for a level turn.

        Returns:
            ``(α, power)``.

        Raises:
            TrimError: if the equilibrium leaves the envelope (max thrust, α_max, n_max).
        """
        n = math.cos(gamma) if load_factor is None else load_factor
        _, _, rho, a_sound = isa_scalar(altitude)
        mach = airspeed / a_sound
        qs = 0.5 * rho * airspeed**2 * self.S
        a_lo, a_hi = self._alpha_bounds
        if not self._n_bounds[0] <= n <= self._n_bounds[1]:
            raise TrimError(f"Load factor {n:.2f} outside limits {self._n_bounds}.")

        power = 0.5
        alpha = 0.0
        for _ in range(50):
            thrust = self.engine.thrust(power, rho, mach)

            def normal_eq(a: float, thrust: float = thrust) -> float:
                return qs * self.aero.cl(a, mach) + thrust * math.sin(a) - n * self.weight

            if normal_eq(a_lo) * normal_eq(a_hi) > 0:
                raise TrimError(
                    f"No angle of attack in [{math.degrees(a_lo):.0f}°, {math.degrees(a_hi):.0f}°]"
                    f" for n = {n:.2f} at V = {airspeed:.0f} m/s, h = {altitude:.0f} m."
                )
            alpha_new = brentq(normal_eq, a_lo, a_hi, xtol=1e-12)
            drag = qs * self.aero.cd(self.aero.cl(alpha_new, mach), mach)
            thrust_req = (drag + self.weight * math.sin(gamma)) / math.cos(alpha_new)

            def axial_eq(pw: float, thrust_req: float = thrust_req) -> float:
                return self.engine.thrust(pw, rho, mach) - thrust_req

            if axial_eq(1.0) < 0:
                raise TrimError(
                    f"Insufficient thrust: {thrust_req / 1e3:.1f} kN required, "
                    f"{self.engine.thrust(1.0, rho, mach) / 1e3:.1f} kN available "
                    f"(V = {airspeed:.0f} m/s, h = {altitude:.0f} m)."
                )
            if axial_eq(0.0) > 0:
                raise TrimError(
                    "Idle thrust exceeds the requirement: the aircraft accelerates "
                    f"(V = {airspeed:.0f} m/s, h = {altitude:.0f} m, "
                    f"flight-path angle {math.degrees(gamma):.1f}°)."
                )
            power_new = brentq(axial_eq, 0.0, 1.0, xtol=1e-12)
            converged = abs(alpha_new - alpha) < 1e-11 and abs(power_new - power) < 1e-11
            alpha, power = alpha_new, power_new
            if converged:
                return alpha, power
        raise TrimError("The trim computation did not converge.")

    def trimmed_state(
        self,
        altitude: float,
        airspeed: float,
        *,
        gamma: float = 0.0,
        heading: float = 0.0,
        load_factor: float = 1.0,
        north: float = 0.0,
        east: float = 0.0,
    ) -> tuple[Vec, Vec]:
        """Trim state and control.

        * ``load_factor = 1``: straight flight (level, or climbing at flight-path angle ``gamma``);
        * ``load_factor > 1`` with ``gamma = 0``: steady level turn, bank angle
          μ = acos(1/n).

        Returns:
            ``(x, u)``: state and the constant control that holds it.
        """
        if load_factor > 1.0 and gamma != 0.0:
            raise ValueError("Steady turn: only the level turn (gamma = 0) is supported.")
        n = math.cos(gamma) if load_factor == 1.0 else load_factor
        bank = 0.0 if load_factor == 1.0 else math.acos(1.0 / load_factor)
        alpha, power = self.trim(altitude, airspeed, gamma, n)
        x = self.make_state(
            altitude=altitude,
            airspeed=airspeed,
            gamma=gamma,
            heading=heading,
            bank=bank,
            alpha=alpha,
            power=power,
            north=north,
            east=east,
        )
        u = np.array([power, alpha, 0.0])
        return x, u


# ----------------------------------------------------------------------
# Internal helpers
# ----------------------------------------------------------------------
def _xw_horizontal_rate(q: Vec, dq: Vec) -> tuple[float, float]:
    """Rates of the North and East components of x_w (1st column of C_nw)."""
    q0, q1, q2, q3 = q
    d0, d1, d2, d3 = dq
    dn = 2.0 * (q0 * d0 + q1 * d1 - q2 * d2 - q3 * d3)
    de = 2.0 * (d1 * q2 + q1 * d2 + d0 * q3 + q0 * d3)
    return dn, de
