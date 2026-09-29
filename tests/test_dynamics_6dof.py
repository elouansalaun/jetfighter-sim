"""Validation du modèle 6-DOF du F-16 (phase 3).

Références :
* cas de vérification publié par Stevens & Lewis (dérivées d'état pour un état donné) ;
* valeurs calculées avec l'implémentation de référence AeroBenchVVPython du même modèle
  (5 états aléatoires) ;
* point d'équilibre publié (502 ft/s, niveau de la mer, x_cg = 0.35) ;
* lois physiques (conservation, sens des gouvernes, modes propres).

Pour comparer au modèle d'origine, on utilise ici son atmosphère simplifiée et
g = 32.17 ft/s² (le modèle du projet utilise l'ISA et g0 par défaut).
"""

import math

import numpy as np
import pytest

from jetsim.aircraft import analysis as an
from jetsim.aircraft import dynamics_6dof as d6
from jetsim.aircraft.dynamics_3dof import PointMassAircraft
from jetsim.aircraft.params import load_aircraft
from jetsim.core.constants import FT_TO_M
from jetsim.core.frames import (
    body_to_ned,
    body_velocity_from_aero,
    euler_rates,
    quat_from_euler,
)
from jetsim.core.integrators import simulate

DEG = math.pi / 180
SLUGFT3_TO_KGM3 = 515.378818


# --------------------------------------------------------------------------
# Pont vers les conventions du modèle d'origine (unités impériales, état en V/α/β/Euler)
# --------------------------------------------------------------------------
def sl_atmosphere(h_m: float) -> tuple[float, float, float, float]:
    """Atmosphère simplifiée du modèle Stevens & Lewis (sous-programme ADC), en SI."""
    alt = h_m / FT_TO_M
    tfac = 1 - 0.703e-5 * alt
    t_rankine = 390.0 if alt >= 35_000 else 519.0 * tfac
    rho = 2.377e-3 * tfac**4.14 * SLUGFT3_TO_KGM3
    a = math.sqrt(1.4 * 1716.3 * t_rankine) * FT_TO_M
    return t_rankine, 0.0, rho, a


def sl_model(xcg: float) -> d6.F16SixDof:
    return d6.F16SixDof(
        load_aircraft("f16"), xcg=xcg, atmosphere=sl_atmosphere, gravity=32.17 * FT_TO_M
    )


def from_sl(x, u):
    """État S&L [vt ft/s, α, β, φ, θ, ψ, p, q, r, pn ft, pe ft, h ft, pow %] et commande
    [manette, δe°, δa°, δr°] -> état et commande du projet (gouvernes à leur consigne)."""
    vt, al, be, ph, th, ps, p, q, r, pn, pe, h, pw = x
    s = np.zeros(d6.N_STATE)
    s[d6.PN], s[d6.PE], s[d6.H] = pn * FT_TO_M, pe * FT_TO_M, h * FT_TO_M
    s[d6.VEL] = body_velocity_from_aero(vt * FT_TO_M, al, be)
    s[d6.QUAT] = quat_from_euler(ph, th, ps)
    s[d6.RATES] = (p, q, r)
    s[d6.POWER] = pw / 100
    s[d6.SURF] = np.radians(u[1:4])
    return s, np.array([u[0], *np.radians(u[1:4])])


def to_sl_derivative(s, ds, x_sl):
    """Dérivées du projet -> dérivées de l'état S&L."""
    be, ph, th = x_sl[2], x_sl[3], x_sl[4]
    u, v, w = s[d6.VEL]
    du, dv, dw = ds[d6.VEL]
    vt = math.sqrt(u * u + v * v + w * w)
    vt_dot = (u * du + v * dv + w * dw) / vt
    alpha_dot = (u * dw - w * du) / (u * u + w * w)
    beta_dot = (vt * dv - v * vt_dot) * math.cos(be) / (u * u + w * w)
    er = euler_rates(ph, th, s[d6.RATES])
    return np.array(
        [
            vt_dot / FT_TO_M,
            alpha_dot,
            beta_dot,
            *er,
            *ds[d6.RATES],
            ds[d6.PN] / FT_TO_M,
            ds[d6.PE] / FT_TO_M,
            ds[d6.H] / FT_TO_M,
            ds[d6.POWER] * 100,
        ]
    )


def sl_derivative(x_sl, u_sl, xcg):
    model = sl_model(xcg)
    s, u = from_sl(np.asarray(x_sl, float), np.asarray(u_sl, float))
    return to_sl_derivative(s, model.derivatives(0.0, s, u), x_sl)


# --------------------------------------------------------------------------
# Cas de vérification
# --------------------------------------------------------------------------
SL_CHECK_STATE = [500, 0.5, -0.2, -1, 1, -1, 0.7, -0.8, 0.9, 1000, 900, 10000, 90]
SL_CHECK_CONTROL = [0.9, 20, -15, -20]
# Dérivées publiées par Stevens & Lewis (x_cg = 0.4)
SL_CHECK_PUBLISHED = [
    -75.23724, -0.8813491, -0.4759990, 2.505734, 0.3250820, 2.145926,
    12.62679, 0.9649671, 0.5809759, 342.4439, -266.7707, 248.1241, -58.68999,
]  # fmt: skip
# Même cas avec les tables des implémentations de référence : ṗ et ṙ diffèrent des valeurs
# publiées (écart ≈ 0.001 sur Cl, vraisemblablement une valeur de table différente de
# l'édition utilisée pour la publication). Les 11 autres dérivées sont identiques.
SL_CHECK_REFERENCE_P_R = (12.8289672, 0.5841226)

# (état S&L, commande, x_cg, dérivées de référence AeroBenchVVPython)
REFERENCE_CASES = [
    (
        [
            439.4674,
            0.4119,
            -0.0196,
            -0.518,
            -0.3482,
            1.7431,
            0.8103,
            -0.6453,
            0.3056,
            0.0,
            0.0,
            12633.8079,
            92.0266,
        ],
        [0.92, 5.435, 9.098, 0.758],
        0.383,
        [
            7.5580272,
            -0.821057766,
            0.0190083432,
            0.597941851,
            -0.409328014,
            0.622374494,
            -2.31391487,
            0.662303235,
            0.265918845,
            -134.137043,
            306.408313,
            -285.048759,
            -47.085,
        ],
    ),
    (
        [
            574.1903,
            0.171,
            -0.1333,
            -1.0947,
            0.062,
            -0.4145,
            0.3264,
            -0.9743,
            -0.1046,
            0.0,
            0.0,
            15242.0512,
            22.5858,
        ],
        [0.595, -2.587, -7.2, -14.529],
        0.387,
        [
            1.68468139,
            -0.986624826,
            0.129770966,
            0.377181663,
            -0.539501997,
            0.819584054,
            7.15797995,
            1.33318537,
            0.0823724864,
            539.262749,
            -181.446948,
            -77.247608,
            16.0535,
        ],
    ),
    (
        [
            748.7312,
            0.3854,
            -0.0929,
            1.7873,
            0.1521,
            -0.4034,
            0.8009,
            -0.3613,
            0.392,
            0.0,
            0.0,
            13238.9928,
            28.5398,
        ],
        [0.701, -10.884, -0.248, 4.001],
        0.319,
        [
            -83.7157299,
            -0.640958299,
            0.0052136161,
            0.733903723,
            -0.305235451,
            -0.442178144,
            9.53590838,
            2.67055197,
            -0.985671046,
            508.601109,
            -498.453651,
            231.229927,
            16.98314,
        ],
    ),
    (
        [
            715.6203,
            0.3388,
            0.0729,
            -0.5114,
            -0.1915,
            -0.031,
            -0.0601,
            0.3513,
            0.1544,
            0.0,
            0.0,
            17234.5551,
            5.1622,
        ],
        [0.794, 0.775, -6.244, -0.002],
        0.309,
        [
            -44.4233073,
            0.0651380388,
            -0.203381727,
            -0.0528723819,
            0.381917895,
            -0.0379738034,
            -1.24117607,
            -1.60651421,
            0.575813865,
            631.031543,
            142.052977,
            -306.157732,
            5.48378,
        ],
    ),
    (
        [
            802.352,
            0.6918,
            -0.2648,
            -0.5671,
            0.5522,
            -1.1146,
            0.1341,
            -0.1669,
            0.5484,
            0.0,
            0.0,
            38380.1116,
            84.9556,
        ],
        [0.621, -13.589, 16.094, -23.818],
        0.33,
        [
            -97.103683,
            -0.218426776,
            -0.357064339,
            0.474337531,
            0.153820179,
            0.648613213,
            -0.300263894,
            0.930033169,
            2.56412294,
            425.252314,
            -666.256432,
            -137.95495,
            -224.778,
        ],
    ),
]

# Tolérance : les constantes d'inertie du modèle d'origine sont arrondies à 4 chiffres
# (ex. 1/Iyy = 1.792e-5 au lieu de 1.7917e-5), d'où des écarts relatifs ≈ 2e-4.
REL_TOL = 2e-3


def assert_close(actual, expected, rel=REL_TOL):
    actual, expected = np.asarray(actual), np.asarray(expected)
    err = np.abs(actual - expected) / np.maximum(1.0, np.abs(expected))
    assert err.max() < rel, f"écart relatif max {err.max():.2e} (composante {err.argmax()})"


def test_published_check_case():
    xd = sl_derivative(SL_CHECK_STATE, SL_CHECK_CONTROL, xcg=0.4)
    published = np.array(SL_CHECK_PUBLISHED)
    same = [0, 1, 2, 3, 4, 5, 7, 9, 10, 11, 12]  # toutes sauf ṗ (6) et ṙ (8)
    assert_close(xd[same], published[same])
    assert_close(xd[[6, 8]], SL_CHECK_REFERENCE_P_R)


@pytest.mark.parametrize(("x", "u", "xcg", "expected"), REFERENCE_CASES)
def test_reference_implementation_cases(x, u, xcg, expected):
    assert_close(sl_derivative(x, u, xcg), expected)


def test_published_trim_point():
    """Stevens & Lewis : 502 ft/s, niveau de la mer, x_cg = 0.35 ->
    manette 0.1385, profondeur −0.7588°."""
    model = sl_model(0.35)
    x, u = model.trim(0.0, 502 * FT_TO_M)
    assert u[d6.THROTTLE] == pytest.approx(0.1385, abs=5e-4)
    assert math.degrees(u[d6.ELEVATOR]) == pytest.approx(-0.7588, abs=5e-3)
    assert abs(u[d6.AILERON]) < 1e-6 and abs(u[d6.RUDDER]) < 1e-6
    fd = model.flight_data(x)
    assert math.degrees(fd.alpha) == pytest.approx(2.12, abs=0.02)
    assert fd.gamma == pytest.approx(0.0, abs=1e-9)


# --------------------------------------------------------------------------
# Modèle du projet (ISA, x_cg = 0.30 par défaut)
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def f16() -> d6.F16SixDof:
    return d6.F16SixDof(load_aircraft("f16"))


def run(model, x0, u, t_end, dt=0.01):
    kwargs = {"controller": u} if callable(u) else {"u_const": np.asarray(u, float)}
    return simulate(model.derivatives, x0, t_end, dt=dt, post_step=model.post_step, **kwargs)


def test_default_cg_is_stable_configuration(f16):
    assert f16.xcg == pytest.approx(0.30)


@pytest.mark.parametrize(("h", "v"), [(0, 153), (3000, 200), (6000, 250), (9000, 250)])
def test_trim_holds_steady(f16, h, v):
    x0, u = f16.trim(h, v)
    fd0 = f16.flight_data(x0)
    assert fd0.load_factor == pytest.approx(1.0, abs=2e-3)
    res = run(f16, x0, u, 20.0)
    fd = f16.flight_data(res.x[-1])
    assert fd.altitude == pytest.approx(h, abs=0.5)
    assert fd.airspeed == pytest.approx(v, abs=0.05)
    assert abs(fd.roll) < 1e-6 and abs(fd.beta) < 1e-6


def test_climb_trim(f16):
    x0, u = f16.trim(2000, 200, gamma=10 * DEG)
    fd = f16.flight_data(x0)
    assert fd.gamma == pytest.approx(10 * DEG, abs=1e-9)
    assert fd.climb_rate == pytest.approx(200 * math.sin(10 * DEG), rel=1e-9)
    dx = f16.derivatives(0.0, x0, u)
    assert np.abs(dx[d6.VEL]).max() < 1e-6 and np.abs(dx[d6.RATES]).max() < 1e-6


def test_trim_error_outside_envelope(f16):
    with pytest.raises(d6.TrimError):
        f16.trim(12_000, 80)


def test_consistent_with_point_mass_model(f16):
    """Trim 6-DOF vs polaire du 3-DOF : incidence d'équilibre à moins de 1°."""
    pm = PointMassAircraft(load_aircraft("f16"))
    for h, v in [(0, 153), (3000, 200), (6000, 250)]:
        x6, _ = f16.trim(h, v)
        alpha_3, _ = pm.trim(h, v)
        assert math.degrees(f16.flight_data(x6).alpha - alpha_3) == pytest.approx(0, abs=1.0)


# --------------------------------------------------------------------------
# Sens physique des gouvernes (conventions standard)
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("index", "rate", "name"),
    [
        (d6.ELEVATOR, d6.Q, "profondeur > 0 -> piqué"),
        (d6.AILERON, d6.P, "ailerons > 0 -> roulis à gauche"),
        (d6.RUDDER, d6.R, "direction > 0 -> lacet à gauche"),
    ],
)
def test_control_surface_signs(f16, index, rate, name):
    x0, u0 = f16.trim(3000, 200)
    u = u0.copy()
    u[index] += 5 * DEG
    res = run(f16, x0, u, 0.5)
    assert res.x[-1, rate] < -1e-3, name


def test_actuator_rate_and_position_limits(f16):
    x0, u0 = f16.trim(3000, 200)
    u = u0.copy()
    u[d6.ELEVATOR] = 60 * DEG  # au-delà de la butée de 25°
    res = run(f16, x0, u, 0.1)
    rate_max = f16.params.control_surfaces["elevator"].rate_max
    assert res.x[-1, d6.DE] - x0[d6.DE] == pytest.approx(rate_max * 0.1, rel=1e-3)
    res = run(f16, x0, u, 1.5)
    assert res.x[-1, d6.DE] == pytest.approx(25 * DEG, abs=1e-4)


# --------------------------------------------------------------------------
# Moteur
# --------------------------------------------------------------------------
def test_engine_tables(f16):
    eng = f16.engine
    lbf = 4.4482216152605
    assert eng.thrust(0.5, 0.0, 0.0) == pytest.approx(12_680 * lbf)  # plein gaz sec
    assert eng.thrust(1.0, 0.0, 0.0) == pytest.approx(20_000 * lbf)  # pleine PC
    assert eng.thrust(0.0, 0.0, 0.0) == pytest.approx(1_060 * lbf)  # ralenti
    assert eng.thrust(0.5, 10_000 * FT_TO_M, 0.0) == pytest.approx(9_150 * lbf)
    assert eng.commanded_power(0.77) == pytest.approx(0.5, abs=1e-3)
    assert eng.commanded_power(1.0) == pytest.approx(1.0)
    for p in (0.1, 0.45, 0.8):
        assert eng.commanded_power(eng.throttle_for_power(p)) == pytest.approx(p)


def test_engine_spool_up_is_slow_then_afterburner_lights(f16):
    """Du ralenti à pleine PC : montée lente en régime sec, puis allumage rapide de la PC."""
    x0, u0 = f16.trim(3000, 200)
    u = u0.copy()
    u[d6.THROTTLE] = 1.0
    res = run(f16, x0, u, 12.0)
    power = res.x[:, d6.POWER]
    t_ab = res.t[np.argmax(power >= 0.5)]
    assert 1.0 < t_ab < 6.0
    assert power[-1] > 0.99


# --------------------------------------------------------------------------
# Mécanique du corps rigide
# --------------------------------------------------------------------------
def test_torque_free_rigid_body_conserves_momentum(monkeypatch):
    """Sans aéro, poussée ni gravité : énergie cinétique de rotation et moment cinétique
    (repère inertiel, moteur inclus) conservés — valide les équations d'Euler et Ixz."""
    model = d6.F16SixDof(load_aircraft("f16"), gravity=0.0)
    zero = type("Z", (), {"CX": 0.0, "CY": 0.0, "CZ": 0.0, "Cl": 0.0, "Cm": 0.0, "Cn": 0.0})
    monkeypatch.setattr(model.aero, "coefficients", lambda *a, **k: zero)
    monkeypatch.setattr(model.engine, "thrust", lambda *a, **k: 0.0)
    x0 = model.make_state(altitude=5000, airspeed=200, rates=(1.2, -0.7, 0.4), power=0.0)
    res = run(model, x0, [0.0, 0.0, 0.0, 0.0], 10.0, dt=0.005)

    def invariants(x):
        w = x[d6.RATES]
        h_body = model.inertia @ w + model.h_engine
        return 0.5 * w @ model.inertia @ w, body_to_ned(x[d6.QUAT], h_body)

    e0, h0 = invariants(res.x[0])
    for x in res.x[::200]:
        e, h = invariants(x)
        assert e == pytest.approx(e0, rel=1e-8)
        np.testing.assert_allclose(h, h0, rtol=1e-8, atol=1e-6)


def test_quaternion_stays_normalized_in_aggressive_maneuver(f16):
    x0, _ = f16.trim(5000, 250)

    def pilot(t, _x):
        return np.array([1.0, -15 * DEG * math.sin(2 * t), 20 * DEG * math.sin(3 * t), 0.0])

    res = run(f16, x0, pilot, 10.0)
    assert np.all(np.isfinite(res.x))
    np.testing.assert_allclose(np.linalg.norm(res.x[:, d6.QUAT], axis=1), 1.0, atol=1e-12)


# --------------------------------------------------------------------------
# Modes propres
# --------------------------------------------------------------------------
def test_flight_modes_at_default_cg(f16):
    x, u = f16.trim(0.0, 153.0)
    modes = an.flight_modes(an.linearize(f16, x, u)[0])
    sp, ph = modes["short_period"], modes["phugoid"]
    assert 1.0 < sp.natural_frequency < 4.0 and 0.4 < sp.damping < 0.9
    assert 50 < ph.period < 120 and 0.0 < ph.damping < 0.3
    dr, roll, spiral = modes["dutch_roll"], modes["roll"], modes["spiral"]
    assert 1.5 < dr.period < 3.0 and dr.damping > 0.05
    assert 0.1 < roll.time_constant < 1.0
    assert all(m.stable for m in modes.values())
    assert spiral.eigenvalue.real < 0


def test_nominal_sl_cg_is_statically_unstable():
    """x_cg = 0.35 (nominal Stevens & Lewis) : divergence en tangage, comme le vrai F-16
    sans ses commandes de vol électriques. Plus le centrage recule, plus elle est rapide."""
    rates = []
    for xcg in (0.35, 0.40):
        model = d6.F16SixDof(load_aircraft("f16"), xcg=xcg)
        x, u = model.trim(0.0, 153.0)
        A, _ = an.linearize(model, x, u)
        lon = np.linalg.eigvals(A[np.ix_(an.LONGITUDINAL, an.LONGITUDINAL)])
        rates.append(lon.real.max())
    assert 0 < rates[0] < rates[1]
