"""Instrument panel shared by the 3-DOF and 6-DOF models.

``read_instruments(model, x)`` returns the same quantities whatever the model, which
lets the RL environments (phase 7) build their observations without knowing
which model is running underneath.

Conventions:

* attitude (roll φ, pitch θ, heading ψ): 3-2-1 Euler angles of the **body frame**;
* flight path (flight-path angle γ, course χ): direction of the velocity vector;
* bank μ: rotation of the lift plane about the velocity vector.
  It sets the equilibrium of a turn (n·cos μ = cos γ in level flight). It differs
  from the fuselage roll φ as soon as the angle of attack is large;
* load factors n = f/g, with f the specific force (what an accelerometer
  measures: aero + thrust forces divided by mass, without gravity), in body
  axes. ``nz`` is counted positive upwar (1 in level flight, 9 in a 9 g pull-up);
* specific energy E = h + V²/2g and specific excess power Ps = Ė.

For the 3-DOF model, the body attitude and angular rates are reconstructed exactly
from the wind frame and the angle of attack (model assumption: zero sideslip).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields

import numpy as np
import numpy.typing as npt

from jetsim.aircraft import dynamics_3dof as d3
from jetsim.aircraft import dynamics_6dof as d6
from jetsim.core.atmosphere import calibrated_airspeed, equivalent_airspeed
from jetsim.core.frames import (
    aero_angles,
    dcm_from_quat,
    dcm_wind_to_body,
    euler_from_dcm,
    quat_conjugate,
    quat_multiply,
)

Vec = npt.NDArray[np.float64]


@dataclass(frozen=True)
class Instruments:
    """Observable aircraft quantities (SI, radians)."""

    # Position
    north: float
    east: float
    altitude: float
    # Air data
    tas: float  # true airspeed [m/s]
    cas: float  # calibrated airspeed [m/s]
    eas: float  # equivalent airspeed [m/s]
    mach: float
    dynamic_pressure: float  # [Pa]
    # Vertical speed and flight path
    vertical_speed: float  # ḣ [m/s]
    gamma: float  # flight-path angle
    course: float  # course χ
    bank: float  # bank μ of the lift vector about the velocity
    # Attitude (body frame)
    roll: float  # φ
    pitch: float  # θ
    heading: float  # ψ
    # Aerodynamics
    alpha: float
    beta: float
    # Rate gyros (body frame)
    p: float
    q: float
    r: float
    # Accelerometers (load factors, body frame)
    nx: float
    ny: float
    nz: float  # positive upward
    # Energy
    specific_energy: float  # E [m]
    specific_excess_power: float  # Ps [m/s]
    airspeed_rate: float  # V̇ [m/s²]
    # Engine
    thrust: float  # [N]
    power: float  # [0-1]

    def to_dict(self) -> dict[str, float]:
        return asdict(self)

    def vector(self, names: tuple[str, ...]) -> Vec:
        """Subset of the quantities, in the requested order (for RL observations)."""
        return np.array([getattr(self, n) for n in names], dtype=np.float64)


INSTRUMENT_NAMES: tuple[str, ...] = tuple(f.name for f in fields(Instruments))


def _airspeeds(tas: float, h: float) -> tuple[float, float]:
    return calibrated_airspeed(tas, h), float(equivalent_airspeed(tas, h))


def read_instruments(model: d3.PointMassAircraft | d6.F16SixDof, x: Vec) -> Instruments:
    """Read the instrument panel for state ``x`` of the given model."""
    if isinstance(model, d6.F16SixDof):
        return _read_6dof(model, x)
    if isinstance(model, d3.PointMassAircraft):
        return _read_3dof(model, x)
    raise TypeError(f"Unsupported model: {type(model).__name__}")


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
# 3-DOF: reconstruct the body attitude from the wind frame and α
# --------------------------------------------------------------------------
def _read_3dof(model: d3.PointMassAircraft, x: Vec) -> Instruments:
    fd = model.flight_data(x)
    alpha = fd.alpha
    u = np.array([x[d3.POWER], alpha, x[d3.ROLL_RATE]])
    dx = model.derivatives(0.0, x, u)
    # Wind-frame rotation: ω_w = 2·(q_w* ⊗ q̇_w), vector part
    q_w = x[d3.QUAT]
    omega_w = 2.0 * quat_multiply(quat_conjugate(q_w), dx[d3.QUAT])[1:]
    c_bw = dcm_wind_to_body(alpha, 0.0)
    c_nb = dcm_from_quat(q_w) @ c_bw.T
    roll, pitch, heading = euler_from_dcm(c_nb)
    # ω_body = ω_wind + [0, α̇, 0]. α̇ depends on the command (unknown to the instrument panel):
    # it is omitted; the error is zero in steady state and bounded by the AoA dynamics.
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


# --------------------------------------------------------------------------
# Singularity-free geometry for navigation and aerobatics tasks
# --------------------------------------------------------------------------
def wind_axes(model: d3.PointMassAircraft | d6.F16SixDof, x: Vec) -> Vec:
    """C_nw matrix (wind frame -> NED), valid at any attitude (including vertical).

    Columns: velocity direction ``x_w``, "right wing" ``y_w``, and ``z_w``; lift acts
    along −``z_w``. Unlike the angles (μ, γ, χ), it has no singularity when
    climbing or diving vertically (loop, Split-S).
    """
    if isinstance(model, d6.F16SixDof):
        _, alpha, beta = aero_angles(x[d6.VEL])
        return dcm_from_quat(x[d6.QUAT]) @ dcm_wind_to_body(alpha, beta)
    if isinstance(model, d3.PointMassAircraft):
        return dcm_from_quat(x[d3.QUAT])
    raise TypeError(f"Unsupported model: {type(model).__name__}")


def position_ned(model: d3.PointMassAircraft | d6.F16SixDof, x: Vec) -> Vec:
    """Position [north, east, down] [m]."""
    idx = (d6.PN, d6.PE, d6.H) if isinstance(model, d6.F16SixDof) else (d3.PN, d3.PE, d3.H)
    return np.array([x[idx[0]], x[idx[1]], -x[idx[2]]])
