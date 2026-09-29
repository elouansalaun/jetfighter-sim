"""Rigid-body "6-DOF" F-16 model, flown through its control surfaces and throttle.

Equations (body axes, flat Earth, no wind) ::

    m·(v̇ + ω × v) = F_aero + F_thrust + m·g
    I·ω̇ + ω × (I·ω + h_engine) = M_aero
    q̇ = ½·q ⊗ [0, ω]                   (attitude, body -> NED quaternion)
    ṗ_NED = C_nb(q)·v

with v = [u, v, w] the velocity (= airspeed), ω = [p, q, r], I the inertia tensor
(including the Ixz product) and h_engine = [h_e, 0, 0] the engine angular momentum
(gyroscopic effect). Aerodynamic forces: ``F16Aero`` (Stevens & Lewis tables).

State (17):
    ``[x_N, y_E, h, u, v, w, q0, q1, q2, q3, p, q, r, P, δe, δa, δr]``
    position [m], body velocity [m/s], quaternion, angular rates [rad/s],
    engine power [0-1] (0.5 = full dry power), effective control surfaces [rad].

Controls (4): ``[throttle, δe_cmd, δa_cmd, δr_cmd]``
    throttle ∈ [0, 1], surface commands [rad]. Standard sign conventions:
    δe > 0 pitches nose down, δa > 0 rolls left, δr > 0 yaws left.

Actuators: first order (τ ≈ 0.05 s), position and rate limited.

Validity domain: that of the tables (−10° ≤ α ≤ 45°, |β| ≤ 30°), subsonic
(no Mach effect in the aero tables; thrust, however, does depend on Mach).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.optimize import least_squares

from jetsim.aircraft.f16_aero import F16Aero, F16AeroTables, load_tables_yaml
from jetsim.aircraft.params import AircraftParams
from jetsim.aircraft.propulsion import TabulatedTurbofan
from jetsim.core.atmosphere import isa_scalar
from jetsim.core.constants import FT_TO_M, G0, LBF_TO_N, SLUGFT2_TO_KGM2
from jetsim.core.frames import (
    aero_angles,
    body_velocity_from_aero,
    euler_from_quat,
    flight_path_angles,
    quat_derivative,
    quat_from_euler,
)
from jetsim.core.integrators import DEFAULT_DT, rk4_step

Vec = npt.NDArray[np.float64]
Atmosphere = Callable[[float], tuple[float, float, float, float]]

# State indices
PN, PE, H, U, V, W, Q0, Q1, Q2, Q3, P, Q, R, POWER, DE, DA, DR = range(17)
N_STATE = 17
VEL = slice(U, W + 1)
QUAT = slice(Q0, Q3 + 1)
RATES = slice(P, R + 1)
SURF = slice(DE, DR + 1)

# Control indices
THROTTLE, ELEVATOR, AILERON, RUDDER = range(4)
N_CONTROL = 4


class TrimError(RuntimeError):
    """Trimming did not converge (flight point outside the envelope?)."""


@dataclass(frozen=True)
class FlightData6:
    """Instrument quantities of the 6-DOF model (SI, radians)."""

    north: float
    east: float
    altitude: float
    airspeed: float
    mach: float
    dynamic_pressure: float
    alpha: float
    beta: float
    roll: float  # φ
    pitch: float  # θ
    yaw: float  # ψ
    p: float
    q: float
    r: float
    gamma: float  # flight-path angle
    course: float  # course χ
    climb_rate: float
    load_factor: float  # n_z = −Z_aero/(m·g), positive in a pull-up
    lateral_load_factor: float  # n_y = Y_aero/(m·g)
    thrust: float
    power: float
    elevator: float
    aileron: float
    rudder: float


class F16SixDof:
    """6-DOF F-16 model (see the module docstring)."""

    def __init__(
        self,
        params: AircraftParams,
        *,
        xcg: float | None = None,
        atmosphere: Atmosphere = isa_scalar,
        gravity: float = G0,
    ) -> None:
        if params.six_dof is None:
            raise ValueError("The aircraft configuration has no six_dof section.")
        sd = params.six_dof
        raw = load_tables_yaml(sd.tables_path)
        tables = F16AeroTables.from_dict(raw)
        geo = params.geometry
        self.params = params
        self.aero = F16Aero(tables, geo.wing_span, geo.mean_chord, sd.xcg_ref)
        self.engine = TabulatedTurbofan(raw["engine"], lbf_to_n=LBF_TO_N, ft_to_m=FT_TO_M)
        self.xcg = sd.xcg if xcg is None else xcg
        self.atmosphere = atmosphere
        self.g = gravity

        self.mass = params.mass.mass
        self.S = geo.wing_area
        self.b = geo.wing_span
        self.cbar = geo.mean_chord
        self.inertia = params.mass.inertia_tensor
        self.inertia_inv = np.linalg.inv(self.inertia)
        self.h_engine = np.array(
            [raw["engine"]["angular_momentum_slugft2_s"] * SLUGFT2_TO_KGM2, 0.0, 0.0]
        )

        self.tau_act = sd.actuator_time_constant
        cs = params.control_surfaces
        self._surf_limits = [
            (cs[name].min, cs[name].max, cs[name].rate_max)
            for name in ("elevator", "aileron", "rudder")
        ]

    # ------------------------------------------------------------------
    # State construction
    # ------------------------------------------------------------------
    @staticmethod
    def make_state(
        *,
        altitude: float,
        airspeed: float,
        alpha: float = 0.0,
        beta: float = 0.0,
        roll: float = 0.0,
        pitch: float | None = None,
        yaw: float = 0.0,
        rates: tuple[float, float, float] = (0.0, 0.0, 0.0),
        power: float = 0.3,
        surfaces: tuple[float, float, float] = (0.0, 0.0, 0.0),
        north: float = 0.0,
        east: float = 0.0,
    ) -> Vec:
        """State from readable quantities. By default, pitch θ = α (level flight)."""
        x = np.zeros(N_STATE)
        x[PN], x[PE], x[H] = north, east, altitude
        x[VEL] = body_velocity_from_aero(airspeed, alpha, beta)
        x[QUAT] = quat_from_euler(roll, alpha if pitch is None else pitch, yaw)
        x[RATES] = rates
        x[POWER] = power
        x[SURF] = surfaces
        return x

    # ------------------------------------------------------------------
    # Dynamics
    # ------------------------------------------------------------------
    def forces_moments(self, x: Vec) -> tuple[Vec, Vec, float, float, float]:
        """Body-axis forces: (F_aero + thrust [N], M_aero [N·m], Mach, q̄, thrust)."""
        V_air, alpha, beta = aero_angles(x[VEL])
        _, _, rho, a_sound = self.atmosphere(x[H])
        mach = V_air / a_sound
        qbar = 0.5 * rho * V_air * V_air
        c = self.aero.coefficients(
            alpha, beta, x[DE], x[DA], x[DR], x[P], x[Q], x[R], V_air, self.xcg
        )
        thrust = self.engine.thrust(x[POWER], x[H], mach)
        qs = qbar * self.S
        force = np.array([qs * c.CX + thrust, qs * c.CY, qs * c.CZ])
        moment = np.array([qs * self.b * c.Cl, qs * self.cbar * c.Cm, qs * self.b * c.Cn])
        return force, moment, mach, qbar, thrust

    def derivatives(self, t: float, x: Vec, u: Vec) -> Vec:
        """ẋ = f(t, x, u)."""
        force, moment, _, _, _ = self.forces_moments(x)
        uvw = x[VEL]
        omega = x[RATES]
        q0, q1, q2, q3 = x[QUAT]
        g = self.g

        # Gravity in body axes: g · (3rd row of C_nb)
        grav = g * np.array(
            [
                2.0 * (q1 * q3 - q0 * q2),
                2.0 * (q2 * q3 + q0 * q1),
                q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3,
            ]
        )

        dx = np.empty(N_STATE)
        dx[VEL] = force / self.mass + grav - np.cross(omega, uvw)
        h_total = self.inertia @ omega + self.h_engine
        dx[RATES] = self.inertia_inv @ (moment - np.cross(omega, h_total))
        dx[QUAT] = quat_derivative(x[QUAT], omega)

        # Navigation: NED velocity = C_nb · v
        u_, v_, w_ = uvw
        dx[PN] = (
            (q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3) * u_
            + 2.0 * (q1 * q2 - q0 * q3) * v_
            + 2.0 * (q1 * q3 + q0 * q2) * w_
        )
        dx[PE] = (
            2.0 * (q1 * q2 + q0 * q3) * u_
            + (q0 * q0 - q1 * q1 + q2 * q2 - q3 * q3) * v_
            + 2.0 * (q2 * q3 - q0 * q1) * w_
        )
        dx[H] = -(
            2.0 * (q1 * q3 - q0 * q2) * u_
            + 2.0 * (q2 * q3 + q0 * q1) * v_
            + (q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3) * w_
        )

        # Engine and actuators
        dx[POWER] = self.engine.power_rate(x[POWER], self.engine.commanded_power(u[THROTTLE]))
        for k, (lo, hi, rate_max) in enumerate(self._surf_limits):
            target = min(max(u[ELEVATOR + k], lo), hi)
            rate = (target - x[DE + k]) / self.tau_act
            dx[DE + k] = min(max(rate, -rate_max), rate_max)
        return dx

    @staticmethod
    def post_step(x: Vec) -> Vec:
        """Renormalize the quaternion (modifies ``x`` in place)."""
        q = x[QUAT]
        x[QUAT] = q / math.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3])
        return x

    def step(self, x: Vec, u: Vec, dt: float = DEFAULT_DT) -> Vec:
        """One RK4 physics step + renormalization."""
        return self.post_step(rk4_step(self.derivatives, 0.0, x, np.asarray(u, float), dt))

    # ------------------------------------------------------------------
    # Instruments
    # ------------------------------------------------------------------
    def flight_data(self, x: Vec) -> FlightData6:
        force, _, mach, qbar, thrust = self.forces_moments(x)
        V_air, alpha, beta = aero_angles(x[VEL])
        roll, pitch, yaw = euler_from_quat(x[QUAT])
        dx = self.derivatives(0.0, x, np.array([0.0, x[DE], x[DA], x[DR]]))
        _, gamma, course = flight_path_angles(np.array([dx[PN], dx[PE], -dx[H]]))
        weight = self.mass * self.g
        return FlightData6(
            north=x[PN],
            east=x[PE],
            altitude=x[H],
            airspeed=V_air,
            mach=mach,
            dynamic_pressure=qbar,
            alpha=alpha,
            beta=beta,
            roll=roll,
            pitch=pitch,
            yaw=yaw,
            p=x[P],
            q=x[Q],
            r=x[R],
            gamma=gamma,
            course=course,
            climb_rate=dx[H],
            load_factor=-(force[2]) / weight,
            lateral_load_factor=force[1] / weight,
            thrust=thrust,
            power=x[POWER],
            elevator=x[DE],
            aileron=x[DA],
            rudder=x[DR],
        )

    # ------------------------------------------------------------------
    # Trimming
    # ------------------------------------------------------------------
    def trim(
        self, altitude: float, airspeed: float, gamma: float = 0.0, yaw: float = 0.0
    ) -> tuple[Vec, Vec]:
        """Steady straight flight, wings level, flight-path angle ``gamma`` (0 = level).

        Unknowns: α, β, throttle, δe, δa, δr. Equations: u̇ = v̇ = ẇ = ṗ = q̇ = ṙ = 0,
        with p = q = r = 0, engine power at equilibrium, surfaces at their command and
        θ such that the flight-path angle is ``gamma``.

        Returns:
            ``(x, u)``: trim state and control.

        Raises:
            TrimError: if the residual stays significant (point outside the envelope).
        """

        def build(z: Vec) -> tuple[Vec, Vec]:
            alpha, beta, throttle, de, da, dr = z
            a = math.cos(alpha) * math.cos(beta)
            b = math.sin(alpha) * math.cos(beta)  # φ = 0
            sg = math.sin(gamma)
            theta = math.atan2(
                a * b + sg * math.sqrt(max(a * a - sg * sg + b * b, 0.0)), a * a - sg * sg
            )
            x = self.make_state(
                altitude=altitude,
                airspeed=airspeed,
                alpha=alpha,
                beta=beta,
                pitch=theta,
                yaw=yaw,
                power=self.engine.commanded_power(throttle),
                surfaces=(de, da, dr),
            )
            return x, np.array([throttle, de, da, dr])

        def residual(z: Vec) -> Vec:
            x, u = build(z)
            dx = self.derivatives(0.0, x, u)
            return np.concatenate([dx[VEL], dx[RATES] * 10.0])

        lim = self._surf_limits
        lo = np.array([-10 * math.pi / 180, -0.3, 0.0, lim[0][0], lim[1][0], lim[2][0]])
        hi = np.array([45 * math.pi / 180, 0.3, 1.0, lim[0][1], lim[1][1], lim[2][1]])
        z0 = np.array([0.05, 0.0, 0.3, 0.0, 0.0, 0.0])
        sol = least_squares(residual, z0, bounds=(lo, hi), xtol=1e-14, ftol=1e-14, gtol=1e-14)
        if np.max(np.abs(sol.fun)) > 1e-6:
            raise TrimError(
                f"No trim found at V = {airspeed:.0f} m/s, h = {altitude:.0f} m, "
                f"γ = {math.degrees(gamma):.1f}° (residual {np.max(np.abs(sol.fun)):.2e})."
            )
        return build(sol.x)
