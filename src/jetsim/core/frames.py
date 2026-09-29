"""Frames, rotations and attitude kinematics.

Conventions (see also the README):

* NED: local inertial frame, x North, y East, z Down (flat Earth). Altitude h = −z.
* Body (FRD): x toward the nose, y toward the right wing, z toward the belly.
* Wind: x aligned with the airspeed vector, obtained from the body frame via α and β.
* Quaternion ``q = [q0, q1, q2, q3]``: scalar first, Hamilton product, unit norm.
  It represents the aircraft attitude: ``v_ned = C_nb(q) @ v_body``.
* Euler 3-2-1: heading ψ, then pitch θ, then bank φ. ``C_nb = Rz(ψ)·Ry(θ)·Rx(φ)``.
* Notation ``C_ab``: matrix that transforms a vector from frame b to frame a.

Euler angles are only used for display, observations and initial conditions:
attitude is always integrated on the quaternion (no gimbal lock
at θ = ±90°, essential for a loop or an Immelmann).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

Vec = npt.NDArray[np.float64]

_TWO_PI = 2.0 * math.pi


# --------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------
def wrap_angle(angle: float) -> float:
    """Wrap an angle into [−π, π) (useful for heading errors)."""
    return (angle + math.pi) % _TWO_PI - math.pi


def skew(v: Vec) -> Vec:
    """Skew-symmetric matrix [v×] such that ``skew(a) @ b == np.cross(a, b)``."""
    return np.array(
        [
            [0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0],
        ]
    )


# --------------------------------------------------------------------------
# Quaternions
# --------------------------------------------------------------------------
def quat_identity() -> Vec:
    """Identity quaternion: aircraft level, nose North."""
    return np.array([1.0, 0.0, 0.0, 0.0])


def quat_normalize(q: Vec) -> Vec:
    """Return the unit quaternion with the same direction (the sign is preserved).

    This is the function to call after each integration step: it does not change the sign,
    so it introduces no discontinuity in the state trajectory.
    """
    n = math.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3])
    if n == 0.0:
        raise ValueError("Zero quaternion: cannot normalize.")
    return np.asarray(q, dtype=np.float64) / n


def quat_canonical(q: Vec) -> Vec:
    """Unique representative: unit norm with non-negative scalar part (q and −q = same attitude)."""
    qn = quat_normalize(q)
    return qn if qn[0] >= 0.0 else -qn


def quat_conjugate(q: Vec) -> Vec:
    """Conjugate q* (= inverse for a unit quaternion)."""
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_multiply(a: Vec, b: Vec) -> Vec:
    """Hamilton product a ⊗ b (composition: apply b then a)."""
    a0, a1, a2, a3 = a
    b0, b1, b2, b3 = b
    return np.array(
        [
            a0 * b0 - a1 * b1 - a2 * b2 - a3 * b3,
            a0 * b1 + a1 * b0 + a2 * b3 - a3 * b2,
            a0 * b2 - a1 * b3 + a2 * b0 + a3 * b1,
            a0 * b3 + a1 * b2 - a2 * b1 + a3 * b0,
        ]
    )


def quat_from_axis_angle(axis: Vec, angle: float) -> Vec:
    """Quaternion of a rotation by ``angle`` [rad] about ``axis`` (normalized here)."""
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    s = math.sin(0.5 * angle)
    return np.array([math.cos(0.5 * angle), axis[0] * s, axis[1] * s, axis[2] * s])


def quat_derivative(q: Vec, omega_body: Vec | Sequence[float], k_norm: float = 0.0) -> Vec:
    """Attitude quaternion derivative: q̇ = ½ · q ⊗ [0, ω].

    Args:
        q: body -> NED quaternion.
        omega_body: angular velocity of the body relative to the NED frame, expressed in the
            body frame: ``[p, q, r]`` [rad/s] (roll, pitch, yaw).
        k_norm: optional gain pulling back toward |q| = 1 (term ``k·(1 − |q|²)·q``). Leave at 0
            when renormalizing after each step (the project default).
    """
    q0, q1, q2, q3 = q
    p, qr, r = omega_body  # qr: pitch rate (avoids clashing with the quaternion q)
    dq = 0.5 * np.array(
        [
            -q1 * p - q2 * qr - q3 * r,
            q0 * p + q2 * r - q3 * qr,
            q0 * qr + q3 * p - q1 * r,
            q0 * r + q1 * qr - q2 * p,
        ]
    )
    if k_norm:
        dq += k_norm * (1.0 - (q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3)) * np.asarray(q)
    return dq


# --------------------------------------------------------------------------
# Rotation matrices (DCM)
# --------------------------------------------------------------------------
def dcm_from_quat(q: Vec) -> Vec:
    """C_nb matrix (body -> NED) from the attitude quaternion."""
    q0, q1, q2, q3 = q
    q00, q11, q22, q33 = q0 * q0, q1 * q1, q2 * q2, q3 * q3
    return np.array(
        [
            [q00 + q11 - q22 - q33, 2.0 * (q1 * q2 - q0 * q3), 2.0 * (q1 * q3 + q0 * q2)],
            [2.0 * (q1 * q2 + q0 * q3), q00 - q11 + q22 - q33, 2.0 * (q2 * q3 - q0 * q1)],
            [2.0 * (q1 * q3 - q0 * q2), 2.0 * (q2 * q3 + q0 * q1), q00 - q11 - q22 + q33],
        ]
    )


def quat_from_dcm(C_nb: Vec) -> Vec:
    """Quaternion from C_nb (Shepperd's method, numerically robust)."""
    C = np.asarray(C_nb, dtype=np.float64)
    tr = C[0, 0] + C[1, 1] + C[2, 2]
    # Pick the largest of the 4 diagonal terms to avoid dividing by ~0
    candidates = (tr, C[0, 0], C[1, 1], C[2, 2])
    k = int(np.argmax(candidates))
    if k == 0:
        s = 2.0 * math.sqrt(1.0 + tr)
        q = [0.25 * s, (C[2, 1] - C[1, 2]) / s, (C[0, 2] - C[2, 0]) / s, (C[1, 0] - C[0, 1]) / s]
    elif k == 1:
        s = 2.0 * math.sqrt(1.0 + C[0, 0] - C[1, 1] - C[2, 2])
        q = [(C[2, 1] - C[1, 2]) / s, 0.25 * s, (C[0, 1] + C[1, 0]) / s, (C[0, 2] + C[2, 0]) / s]
    elif k == 2:
        s = 2.0 * math.sqrt(1.0 + C[1, 1] - C[0, 0] - C[2, 2])
        q = [(C[0, 2] - C[2, 0]) / s, (C[0, 1] + C[1, 0]) / s, 0.25 * s, (C[1, 2] + C[2, 1]) / s]
    else:
        s = 2.0 * math.sqrt(1.0 + C[2, 2] - C[0, 0] - C[1, 1])
        q = [(C[1, 0] - C[0, 1]) / s, (C[0, 2] + C[2, 0]) / s, (C[1, 2] + C[2, 1]) / s, 0.25 * s]
    return quat_canonical(np.array(q))


def dcm_from_euler(phi: float, theta: float, psi: float) -> Vec:
    """C_nb matrix (body -> NED) from 3-2-1 Euler angles [rad]."""
    cf, sf = math.cos(phi), math.sin(phi)
    ct, st = math.cos(theta), math.sin(theta)
    cp, sp = math.cos(psi), math.sin(psi)
    return np.array(
        [
            [ct * cp, sf * st * cp - cf * sp, cf * st * cp + sf * sp],
            [ct * sp, sf * st * sp + cf * cp, cf * st * sp - sf * cp],
            [-st, sf * ct, cf * ct],
        ]
    )


def euler_from_dcm(C_nb: Vec) -> tuple[float, float, float]:
    """3-2-1 Euler angles ``(φ, θ, ψ)`` [rad] from C_nb. ψ ∈ [−π, π]."""
    s_theta = -min(max(C_nb[2, 0], -1.0), 1.0)
    phi = math.atan2(C_nb[2, 1], C_nb[2, 2])
    theta = math.asin(s_theta)
    psi = math.atan2(C_nb[1, 0], C_nb[0, 0])
    return phi, theta, psi


# --------------------------------------------------------------------------
# Euler <-> quaternion
# --------------------------------------------------------------------------
def quat_from_euler(phi: float, theta: float, psi: float) -> Vec:
    """Body -> NED quaternion from 3-2-1 Euler angles [rad]."""
    cf, sf = math.cos(0.5 * phi), math.sin(0.5 * phi)
    ct, st = math.cos(0.5 * theta), math.sin(0.5 * theta)
    cp, sp = math.cos(0.5 * psi), math.sin(0.5 * psi)
    q = np.array(
        [
            cf * ct * cp + sf * st * sp,
            sf * ct * cp - cf * st * sp,
            cf * st * cp + sf * ct * sp,
            cf * ct * sp - sf * st * cp,
        ]
    )
    return quat_canonical(q)


def euler_from_quat(q: Vec) -> tuple[float, float, float]:
    """3-2-1 Euler angles ``(φ, θ, ψ)`` [rad] from the quaternion.

    Near θ = ±90° (nose vertical), φ and ψ are no longer defined separately:
    the result stays finite but how it is split between them is arbitrary. This is a limitation
    of Euler angles, not of the model (which integrates the quaternion).
    """
    q0, q1, q2, q3 = q
    phi = math.atan2(2.0 * (q0 * q1 + q2 * q3), q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3)
    s_theta = min(max(2.0 * (q0 * q2 - q1 * q3), -1.0), 1.0)
    theta = math.asin(s_theta)
    psi = math.atan2(2.0 * (q1 * q2 + q0 * q3), q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3)
    return phi, theta, psi


def euler_rates(phi: float, theta: float, omega_body: Vec) -> tuple[float, float, float]:
    """Euler angle rates ``(φ̇, θ̇, ψ̇)`` from ``[p, q, r]``.

    Singular at θ = ±90°: use only for display or observations.
    """
    p, q, r = omega_body
    cf, sf = math.cos(phi), math.sin(phi)
    ct = math.cos(theta)
    if abs(ct) < 1e-9:
        raise ZeroDivisionError("Euler rates undefined at θ = ±90° (gimbal lock).")
    qs_rc = q * sf + r * cf
    return p + qs_rc * math.tan(theta), q * cf - r * sf, qs_rc / ct


# --------------------------------------------------------------------------
# Vector frame changes
# --------------------------------------------------------------------------
def body_to_ned(q: Vec, v_body: Vec) -> Vec:
    """Express in the NED frame a vector given in the body frame."""
    return dcm_from_quat(q) @ v_body


def ned_to_body(q: Vec, v_ned: Vec) -> Vec:
    """Express in the body frame a vector given in the NED frame."""
    return dcm_from_quat(q).T @ v_ned


# --------------------------------------------------------------------------
# Aerodynamic (wind) frame
# --------------------------------------------------------------------------
def aero_angles(v_body: Vec) -> tuple[float, float, float]:
    """Airspeed, angle of attack and sideslip from the body-frame air-relative velocity.

    Args:
        v_body: ``[u, v, w]`` aircraft velocity relative to the air, in the body frame [m/s].

    Returns:
        ``(V, α, β)``: magnitude [m/s], α = atan2(w, u) [rad], β = asin(v / V) [rad].
        If V = 0, returns ``(0, 0, 0)``.
    """
    u, v, w = v_body
    V = math.sqrt(u * u + v * v + w * w)
    if V == 0.0:
        return 0.0, 0.0, 0.0
    alpha = math.atan2(w, u)
    beta = math.asin(min(max(v / V, -1.0), 1.0))
    return V, alpha, beta


def body_velocity_from_aero(V: float, alpha: float, beta: float) -> Vec:
    """Body-frame velocity ``[u, v, w]`` from (V, α, β). Inverse of ``aero_angles``."""
    cb = math.cos(beta)
    return np.array([V * math.cos(alpha) * cb, V * math.sin(beta), V * math.sin(alpha) * cb])


def dcm_wind_to_body(alpha: float, beta: float) -> Vec:
    """C_bw matrix (wind -> body).

    Used to project the aerodynamic forces, naturally expressed in wind axes
    (drag D along −x_w, side force Y along y_w, lift L along −z_w):
    ``F_body = C_bw @ [-D, Y, -L]``.
    """
    ca, sa = math.cos(alpha), math.sin(alpha)
    cb, sb = math.cos(beta), math.sin(beta)
    return np.array(
        [
            [ca * cb, -ca * sb, -sa],
            [sb, cb, 0.0],
            [sa * cb, -sa * sb, ca],
        ]
    )


def flight_path_angles(v_ned: Vec) -> tuple[float, float, float]:
    """Ground speed, flight-path angle and course from the NED velocity.

    Returns:
        ``(V, γ, χ)``: magnitude [m/s], flight-path angle γ (positive climbing) [rad], course χ from
        North toward East [rad] (χ ∈ [−π, π]).
    """
    vn, ve, vd = v_ned
    v_horiz = math.hypot(vn, ve)
    V = math.hypot(v_horiz, vd)
    gamma = math.atan2(-vd, v_horiz)
    chi = math.atan2(ve, vn)
    return V, gamma, chi
