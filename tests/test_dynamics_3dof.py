"""Validation physique du modèle point-masse (phase 2)."""

import math

import numpy as np
import pytest
from scipy.integrate import trapezoid

from jetfighter.aircraft import performance as perf
from jetfighter.aircraft.dynamics_3dof import (
    ALPHA,
    N_STATE,
    POWER,
    QUAT,
    H,
    PointMassAircraft,
    TrimError,
    V,
)
from jetfighter.aircraft.params import load_aircraft
from jetfighter.core.constants import G0
from jetfighter.core.frames import wrap_angle
from jetfighter.core.integrators import simulate

DEG = math.pi / 180


@pytest.fixture(scope="module")
def ac() -> PointMassAircraft:
    return PointMassAircraft(load_aircraft("f16"))


def run(ac, x0, u, t_end, dt=0.01):
    """Simule avec une commande constante ou une loi u(t, x)."""
    kwargs = {"controller": u} if callable(u) else {"u_const": np.asarray(u, float)}
    return simulate(ac.derivatives, x0, t_end, dt=dt, post_step=ac.post_step, **kwargs)


# --------------------------------------------------------------------------
# Construction et conventions
# --------------------------------------------------------------------------
def test_make_state_roundtrip(ac):
    x = ac.make_state(altitude=5000, airspeed=200, gamma=0.1, heading=1.2, bank=0.3)
    assert x.shape == (N_STATE,)
    fd = ac.flight_data(x)
    assert (fd.gamma, fd.heading, fd.bank) == pytest.approx((0.1, 1.2, 0.3))
    assert fd.altitude == 5000 and fd.airspeed == 200


def test_velocity_direction_matches_heading_and_gamma(ac):
    gamma, chi = 10 * DEG, 90 * DEG  # montée cap Est
    x = ac.make_state(altitude=3000, airspeed=200, gamma=gamma, heading=chi, power=0.8)
    dx = ac.derivatives(0.0, x, np.array([0.8, 0.0, 0.0]))
    assert dx[0] == pytest.approx(0.0, abs=1e-9)  # pas de déplacement Nord
    assert dx[1] == pytest.approx(200 * math.cos(gamma))  # vers l'Est
    assert dx[2] == pytest.approx(200 * math.sin(gamma))  # altitude qui augmente


# --------------------------------------------------------------------------
# Équilibres
# --------------------------------------------------------------------------
@pytest.mark.parametrize(("h", "v"), [(0, 170), (3000, 263), (6000, 250), (11000, 472)])
def test_level_flight_trim_is_steady(ac, h, v):
    x0, u = ac.trimmed_state(h, v)
    res = run(ac, x0, u, 60.0)
    assert res.x[-1, H] == pytest.approx(h, abs=1.0)
    assert res.x[-1, V] == pytest.approx(v, abs=0.1)
    assert 0.0 < x0[POWER] <= 1.0
    assert 0.0 < x0[ALPHA] < 15 * DEG or h == 11000


def test_steady_climb_trim(ac):
    """Montée rectiligne à pente constante : équilibre exact au point de départ.
    (Sur une longue durée la vitesse dérive, car l'air se raréfie pendant la montée.)"""
    gamma = 15 * DEG
    x0, u = ac.trimmed_state(2000, 250, gamma=gamma)
    dx = ac.derivatives(0.0, x0, u)
    assert dx[V] == pytest.approx(0.0, abs=1e-9)
    assert np.linalg.norm(dx[QUAT]) == pytest.approx(0.0, abs=1e-9)  # trajectoire rectiligne
    assert dx[H] == pytest.approx(250 * math.sin(gamma))
    res = run(ac, x0, u, 3.0)
    assert res.x[-1, V] == pytest.approx(250, abs=0.2)


@pytest.mark.parametrize("n", [2.0, 4.0, 6.0])
def test_level_turn_rate_matches_theory(ac, n):
    """Virage stabilisé : ω = g·√(n² − 1)/V, altitude et vitesse constantes."""
    v = 263.0
    x0, u = ac.trimmed_state(3000, v, load_factor=n)
    fd = ac.flight_data(x0)
    omega_theory = G0 * math.sqrt(n * n - 1) / v
    assert fd.load_factor == pytest.approx(n, rel=1e-9)
    assert fd.turn_rate == pytest.approx(omega_theory, rel=1e-9)
    t_end = 10.0
    res = run(ac, x0, u, t_end)
    fd_end = ac.flight_data(res.x[-1])
    assert fd_end.altitude == pytest.approx(3000, abs=0.5)
    assert fd_end.airspeed == pytest.approx(v, abs=0.05)
    assert wrap_angle(fd_end.heading - fd.heading) == pytest.approx(
        wrap_angle(omega_theory * t_end), abs=1e-4
    )


def test_trim_errors(ac):
    with pytest.raises(TrimError, match="insuffisante"):
        ac.trim(11_000, 700)  # Mach 2.4 : au-delà de la poussée disponible
    with pytest.raises(TrimError, match="incidence"):
        ac.trim(0, 40)  # trop lent : α > α_max
    with pytest.raises(TrimError, match="limites"):
        ac.trim(3000, 250, load_factor=12)


# --------------------------------------------------------------------------
# Conservation de l'énergie : Ė = Ps pour n'importe quelle manœuvre
# --------------------------------------------------------------------------
def test_energy_equation_holds_during_maneuver(ac):
    """E = h + V²/2g varie exactement de ∫ V·(T·cos α − D)/(m·g) dt (portance ⊥ vitesse)."""
    x0, _ = ac.trimmed_state(4000, 230)

    def pilot(t, _x):
        return np.array([1.0 if t < 8 else 0.3, 12 * DEG * math.sin(0.8 * t), 1.5 * math.sin(t)])

    res = run(ac, x0, pilot, 20.0)
    data = [ac.flight_data(x) for x in res.x]
    energy = np.array([d.specific_energy for d in data])
    ps = np.array([d.specific_excess_power for d in data])
    delta_e_integral = trapezoid(ps, res.t)
    assert energy[-1] - energy[0] == pytest.approx(delta_e_integral, rel=1e-4, abs=0.05)


def test_no_drag_no_thrust_conserves_energy(ac, monkeypatch):
    """Sans traînée ni poussée, l'énergie mécanique est conservée (loop libre)."""
    monkeypatch.setattr(ac.aero, "cd", lambda cl, mach: 0.0)
    monkeypatch.setattr(ac.engine, "thrust", lambda power, rho, mach: 0.0)
    x0 = ac.make_state(altitude=5000, airspeed=250, alpha=5 * DEG, power=0.0)
    res = run(ac, x0, [0.0, 8 * DEG, 0.5], 30.0)
    e = res.x[:, H] + res.x[:, V] ** 2 / (2 * G0)
    assert np.max(np.abs(e - e[0])) < 1e-6


# --------------------------------------------------------------------------
# Manœuvres et limites
# --------------------------------------------------------------------------
def test_full_loop_through_vertical(ac):
    """Looping plein gaz : passe à la verticale puis sur le dos, sans singularité."""
    x0, _ = ac.trimmed_state(3000, 280)
    res = run(ac, x0, [1.0, 25 * DEG, 0.0], 25.0)
    data = [ac.flight_data(x) for x in res.x]
    gammas = np.array([d.gamma for d in data])
    assert np.all(np.isfinite(res.x))
    assert gammas.max() > 85 * DEG  # nez à la verticale
    np.testing.assert_allclose(np.linalg.norm(res.x[:, QUAT], axis=1), 1.0, atol=1e-12)
    # au sommet (altitude max), l'avion est sur le dos, pente ≈ 0, cap inversé
    top = int(np.argmax(res.x[:, H]))
    assert abs(data[top].bank) > 170 * DEG
    assert abs(data[top].gamma) < 5 * DEG
    assert abs(wrap_angle(data[top].heading - math.pi)) < 1 * DEG
    # puis redescente nez vers le bas : le looping se referme
    assert gammas.min() < -80 * DEG


def test_g_limiter(ac):
    """Commande d'incidence max à grande vitesse : le facteur de charge reste ≤ 9 g."""
    x0, _ = ac.trimmed_state(2000, 320)
    res = run(ac, x0, [1.0, 25 * DEG, 0.0], 6.0)
    n = np.array([ac.flight_data(x).load_factor for x in res.x])
    assert n.max() == pytest.approx(9.0, abs=0.3)
    assert n.max() <= 9.3


def test_negative_g_limiter(ac):
    x0, _ = ac.trimmed_state(2000, 320)
    res = run(ac, x0, [1.0, -25 * DEG, 0.0], 3.0)
    n = np.array([ac.flight_data(x).load_factor for x in res.x])
    assert n.min() >= -3.2


def test_alpha_limit_at_low_speed(ac):
    x0, _ = ac.trimmed_state(3000, 130)
    res = run(ac, x0, [1.0, 60 * DEG, 0.0], 3.0)
    assert res.x[:, ALPHA].max() <= 25 * DEG + 1e-9


def test_roll_response_and_rate_limit(ac):
    """Taux de roulis commandé au-delà de la limite : saturé à 240°/s après la montée en
    régime (τ = 0.2 s). En 1.5 s de palier incliné, μ tourne d'environ 300°."""
    x0, _ = ac.trimmed_state(5000, 250)
    res = run(ac, x0, [0.8, 2 * DEG, 10.0], 1.5)
    assert res.x[-1, 9] == pytest.approx(240 * DEG, rel=1e-2)
    banks = np.unwrap([ac.flight_data(x).bank for x in res.x])
    expected = 240 * DEG * (1.5 - 0.2 * (1 - math.exp(-1.5 / 0.2)))
    assert banks[-1] - banks[0] == pytest.approx(expected, rel=0.03)


def test_engine_lag_in_flight(ac):
    x0, _ = ac.trimmed_state(3000, 200)
    res = run(ac, x0, [1.0, x0[ALPHA], 0.0], 1.0)
    p_end = res.x[-1, POWER]
    p0 = x0[POWER]
    assert p_end == pytest.approx(1.0 - (1.0 - p0) * math.exp(-1.0), abs=1e-4)


def test_ceiling_is_plausible(ac):
    """Plafond (taux de montée max = 0.5 m/s) entre 15 et 18.5 km.
    Le plafond publié du F-16 (≈ 15.2 km, 50 000 ft) est une limite opérationnelle ; le
    plafond aérodynamique du modèle est un peu plus haut (≈ 17–18 km)."""
    assert 15_000 < perf.ceiling(ac) < 18_500


def test_flight_envelope_is_plausible(ac):
    """Ordres de grandeur publics : ≈ Mach 2 en altitude, ≈ Mach 1.2–1.4 au niveau de la mer."""
    v_min_sl, v_max_sl = perf.level_speed_range(ac, 0.0)
    assert 1.1 < v_max_sl / perf.speed_of_sound(0.0) < 1.5
    assert v_min_sl < 70  # la vitesse mini est bornée par α_max, pas par la poussée
    _, v_max_11 = perf.level_speed_range(ac, 11_000.0)
    assert 1.8 < v_max_11 / perf.speed_of_sound(11_000.0) < 2.2
    assert 200 < perf.max_rate_of_climb(ac, 0.0) < 400  # ≈ 250–300 m/s pleine PC


def test_turn_performance_is_plausible(ac):
    """Virage soutenu F-16 : de l'ordre de 15–20 °/s à basse altitude vers Mach 0.8."""
    v = 0.8 * perf.speed_of_sound(3000)
    sus = perf.sustained_turn(ac, 3000, v)
    inst = perf.instantaneous_turn(ac, 3000, v)
    assert 13 < math.degrees(sus.turn_rate) < 22
    assert inst.load_factor == pytest.approx(9.0)  # limité par n_max à cette vitesse
    assert inst.turn_rate > sus.turn_rate


def test_robust_at_very_low_speed(ac):
    """Chandelle jusqu'à vitesse quasi nulle : pas de NaN (la fin d'épisode gérera ce cas)."""
    x0 = ac.make_state(altitude=5000, airspeed=40, gamma=89 * DEG, power=0.0)
    res = run(ac, x0, [0.0, 0.0, 0.0], 15.0)
    assert np.all(np.isfinite(res.x))


def test_step_matches_simulate(ac):
    x0, u = ac.trimmed_state(3000, 263, load_factor=3)
    x = x0.copy()
    for _ in range(100):
        x = ac.step(x, u)
    res = run(ac, x0, u, 1.0)
    np.testing.assert_allclose(x, res.x[-1], atol=1e-12)
