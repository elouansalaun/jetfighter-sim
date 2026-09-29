"""Validation des conventions de repères et de la cinématique d'attitude."""

import math

import numpy as np
import pytest

from jetsim.core import frames as fr
from jetsim.core.integrators import rk4_step

RNG = np.random.default_rng(42)
TOL = 1e-9


def random_euler(n: int, theta_margin: float = 1e-2):
    phi = RNG.uniform(-math.pi, math.pi, n)
    theta = RNG.uniform(-math.pi / 2 + theta_margin, math.pi / 2 - theta_margin, n)
    psi = RNG.uniform(-math.pi, math.pi, n)
    return zip(phi, theta, psi, strict=True)


def random_quats(n: int):
    for _ in range(n):
        yield fr.quat_normalize(RNG.normal(size=4))


def same_rotation(q1, q2, tol=TOL) -> bool:
    """q et −q représentent la même attitude."""
    return np.allclose(q1, q2, atol=tol) or np.allclose(q1, -q2, atol=tol)


# --------------------------------------------------------------------------
# Allers-retours Euler <-> quaternion <-> DCM
# --------------------------------------------------------------------------
@pytest.mark.parametrize(("phi", "theta", "psi"), list(random_euler(200)))
def test_euler_quat_roundtrip(phi, theta, psi):
    q = fr.quat_from_euler(phi, theta, psi)
    assert np.linalg.norm(q) == pytest.approx(1.0, abs=TOL)
    np.testing.assert_allclose(fr.euler_from_quat(q), (phi, theta, psi), atol=TOL)


@pytest.mark.parametrize(("phi", "theta", "psi"), list(random_euler(200)))
def test_euler_dcm_roundtrip_and_consistency(phi, theta, psi):
    C = fr.dcm_from_euler(phi, theta, psi)
    np.testing.assert_allclose(fr.euler_from_dcm(C), (phi, theta, psi), atol=TOL)
    np.testing.assert_allclose(fr.dcm_from_quat(fr.quat_from_euler(phi, theta, psi)), C, atol=TOL)


def test_quat_dcm_roundtrip_all_branches():
    """Couvre les 4 branches de Shepperd, y compris les rotations proches de 180°."""
    quats = [*random_quats(500)]
    quats += [fr.quat_from_axis_angle(ax, math.pi - 1e-3) for ax in np.eye(3)]
    quats += [fr.quat_from_axis_angle(ax, math.pi) for ax in np.eye(3)]
    for q in quats:
        assert same_rotation(fr.quat_from_dcm(fr.dcm_from_quat(q)), q)


def test_dcm_is_proper_rotation():
    for q in random_quats(200):
        C = fr.dcm_from_quat(q)
        np.testing.assert_allclose(C @ C.T, np.eye(3), atol=TOL)
        assert np.linalg.det(C) == pytest.approx(1.0, abs=TOL)


def test_quat_multiply_matches_dcm_product():
    for a, b in zip(random_quats(100), random_quats(100), strict=True):
        np.testing.assert_allclose(
            fr.dcm_from_quat(fr.quat_multiply(a, b)),
            fr.dcm_from_quat(a) @ fr.dcm_from_quat(b),
            atol=TOL,
        )
        np.testing.assert_allclose(
            fr.quat_multiply(a, fr.quat_conjugate(a)), fr.quat_identity(), atol=TOL
        )


def test_normalize_keeps_sign_canonical_flips():
    q = np.array([-2.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(fr.quat_normalize(q), [-1, 0, 0, 0])
    np.testing.assert_allclose(fr.quat_canonical(q), [1, 0, 0, 0])
    with pytest.raises(ValueError):
        fr.quat_normalize(np.zeros(4))


# --------------------------------------------------------------------------
# Sens physique des conventions (le test anti « bug de signe »)
# --------------------------------------------------------------------------
def test_heading_90_nose_points_east():
    q = fr.quat_from_euler(0.0, 0.0, math.radians(90))
    np.testing.assert_allclose(fr.body_to_ned(q, np.array([1.0, 0, 0])), [0, 1, 0], atol=TOL)


def test_positive_pitch_is_nose_up():
    th = math.radians(30)
    q = fr.quat_from_euler(0.0, th, 0.0)
    # z NED vers le bas -> nez au-dessus de l'horizon = composante z négative
    np.testing.assert_allclose(
        fr.body_to_ned(q, np.array([1.0, 0, 0])), [math.cos(th), 0, -math.sin(th)], atol=TOL
    )


def test_positive_roll_is_right_wing_down():
    ph = math.radians(30)
    q = fr.quat_from_euler(ph, 0.0, 0.0)
    np.testing.assert_allclose(
        fr.body_to_ned(q, np.array([0.0, 1, 0])), [0, math.cos(ph), math.sin(ph)], atol=TOL
    )


def test_gravity_in_body_frame_for_level_flight():
    q = fr.quat_from_euler(0.0, 0.0, math.radians(123))
    np.testing.assert_allclose(fr.ned_to_body(q, np.array([0, 0, 9.81])), [0, 0, 9.81], atol=TOL)


# --------------------------------------------------------------------------
# Cinématique : intégration du quaternion
# --------------------------------------------------------------------------
def _integrate_attitude(q0, omega, t_end, dt=0.01):
    def f(_t, q, u):
        return fr.quat_derivative(q, u)

    q = q0.copy()
    for k in range(round(t_end / dt)):
        q = fr.quat_normalize(rk4_step(f, k * dt, q, omega, dt))
    return q


def test_constant_roll_rate_gives_expected_bank():
    q = _integrate_attitude(fr.quat_identity(), np.array([0.5, 0.0, 0.0]), t_end=2.0)
    np.testing.assert_allclose(fr.euler_from_quat(q), (1.0, 0.0, 0.0), atol=1e-10)


def test_constant_yaw_rate_gives_expected_heading():
    q = _integrate_attitude(fr.quat_identity(), np.array([0.0, 0.0, 0.2]), t_end=5.0)
    np.testing.assert_allclose(fr.euler_from_quat(q), (0.0, 0.0, 1.0), atol=1e-10)


def test_constant_body_rate_matches_analytic_rotation():
    """ω constant en repère corps : q(t) = q0 ⊗ rot(ω̂, |ω|·t). Traverse aussi θ = ±90°."""
    q0 = fr.quat_from_euler(0.3, -0.2, 1.0)
    omega = np.array([0.7, -1.1, 0.4])
    t_end = 10.0
    q_num = _integrate_attitude(q0, omega, t_end)
    q_ref = fr.quat_multiply(q0, fr.quat_from_axis_angle(omega, np.linalg.norm(omega) * t_end))
    assert same_rotation(q_num, q_ref, tol=1e-8)


def test_looping_through_vertical_without_gimbal_lock():
    """Looping complet (360° en 6 s) : passe par θ = ±90° et revient à l'attitude initiale."""
    q_rate = 2 * math.pi / 6.0
    q_mid = _integrate_attitude(fr.quat_identity(), np.array([0.0, q_rate, 0.0]), t_end=1.5)
    _, theta_mid, _ = fr.euler_from_quat(q_mid)
    assert theta_mid == pytest.approx(math.pi / 2, abs=1e-6)  # nez à la verticale
    q = _integrate_attitude(fr.quat_identity(), np.array([0.0, q_rate, 0.0]), t_end=6.0)
    assert same_rotation(q, fr.quat_identity(), tol=1e-8)


def test_euler_rates_match_finite_difference():
    omega = np.array([0.3, 0.2, -0.1])
    q0 = fr.quat_from_euler(0.2, 0.1, 0.5)
    dt = 1e-5
    q1 = _integrate_attitude(q0, omega, t_end=dt, dt=dt)
    e0 = np.array(fr.euler_from_quat(q0))
    e1 = np.array(fr.euler_from_quat(q1))
    np.testing.assert_allclose((e1 - e0) / dt, fr.euler_rates(e0[0], e0[1], omega), atol=1e-5)


def test_k_norm_pulls_toward_unit_norm():
    q = np.array([1.1, 0.0, 0.0, 0.0])
    dq = fr.quat_derivative(q, np.zeros(3), k_norm=1.0)
    assert dq[0] < 0  # |q| > 1 -> ramené vers 1


# --------------------------------------------------------------------------
# Repère aérodynamique et trajectoire
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("V", "alpha", "beta"), [(200.0, 0.1, 0.0), (150.0, -0.2, 0.05), (300.0, 0.6, -0.1)]
)
def test_aero_angles_roundtrip(V, alpha, beta):
    v_body = fr.body_velocity_from_aero(V, alpha, beta)
    np.testing.assert_allclose(fr.aero_angles(v_body), (V, alpha, beta), atol=TOL)
    # l'axe x vent est bien porté par la vitesse
    np.testing.assert_allclose(fr.dcm_wind_to_body(alpha, beta) @ [V, 0, 0], v_body, atol=TOL)


def test_wind_dcm_is_proper_rotation():
    C = fr.dcm_wind_to_body(0.3, -0.2)
    np.testing.assert_allclose(C @ C.T, np.eye(3), atol=TOL)
    assert np.linalg.det(C) == pytest.approx(1.0)


def test_lift_points_up_in_level_flight():
    """Portance (−z vent) : vers le haut en NED pour un avion à plat avec α > 0."""
    alpha = math.radians(5)
    theta = alpha  # vol en palier : θ = α
    q = fr.quat_from_euler(0.0, theta, 0.0)
    lift_body = fr.dcm_wind_to_body(alpha, 0.0) @ np.array([0.0, 0.0, -1.0])
    np.testing.assert_allclose(fr.body_to_ned(q, lift_body), [0, 0, -1], atol=TOL)


def test_aero_angles_zero_velocity():
    assert fr.aero_angles(np.zeros(3)) == (0.0, 0.0, 0.0)


def test_flight_path_angles():
    V, gamma, chi = fr.flight_path_angles(np.array([100.0, 100.0, -50.0]))
    assert V == pytest.approx(math.sqrt(100**2 + 100**2 + 50**2))
    assert gamma == pytest.approx(math.atan2(50, math.hypot(100, 100)))
    assert chi == pytest.approx(math.radians(45))


@pytest.mark.parametrize(
    ("angle", "expected"),
    [
        (0.0, 0.0),
        (math.pi + 0.1, -math.pi + 0.1),
        (-math.pi - 0.1, math.pi - 0.1),
        (7.0, 7.0 - 2 * math.pi),
    ],
)
def test_wrap_angle(angle, expected):
    assert fr.wrap_angle(angle) == pytest.approx(expected)


def test_skew_matches_cross():
    a, b = RNG.normal(size=3), RNG.normal(size=3)
    np.testing.assert_allclose(fr.skew(a) @ b, np.cross(a, b), atol=TOL)
