"""Validation des intégrateurs contre des solutions analytiques."""

import math

import numpy as np
import pytest

from jetsim.core.constants import G0
from jetsim.core.integrators import (
    euler_step,
    rk4_step,
    simulate,
    substeps_per_control,
)


def free_fall(_t, x, _u):
    """x = [h, vz] (h vers le haut) : ḣ = vz, v̇z = −g."""
    return np.array([x[1], -G0])


def harmonic(_t, x, u):
    """Oscillateur harmonique x = [pos, vit], pulsation ω = u[0]."""
    w = u[0]
    return np.array([x[1], -(w**2) * x[0]])


def decay(_t, x, _u):
    return -x


def test_free_fall_exact_with_rk4():
    h0, v0, t_end = 5_000.0, 20.0, 10.0
    res = simulate(free_fall, np.array([h0, v0]), t_end, dt=0.01)
    t = res.t
    np.testing.assert_allclose(res.x[:, 0], h0 + v0 * t - 0.5 * G0 * t**2, atol=1e-8)
    np.testing.assert_allclose(res.x[:, 1], v0 - G0 * t, atol=1e-10)


def test_harmonic_oscillator_rk4_accuracy_and_energy():
    w = 2 * math.pi  # période 1 s
    res = simulate(harmonic, np.array([1.0, 0.0]), 10.0, dt=0.01, u_const=np.array([w]))
    np.testing.assert_allclose(res.x[:, 0], np.cos(w * res.t), atol=1e-4)
    energy = 0.5 * res.x[:, 1] ** 2 + 0.5 * w**2 * res.x[:, 0] ** 2
    assert abs(energy[-1] / energy[0] - 1.0) < 1e-5


def test_euler_is_much_worse_than_rk4():
    """Euler explicite fait diverger l'énergie de l'oscillateur : justifie le choix de RK4."""
    w = 2 * math.pi
    x0, u = np.array([1.0, 0.0]), np.array([w])
    e = simulate(harmonic, x0, 10.0, dt=0.01, u_const=u, method="euler")
    energy_ratio = (e.x[-1, 1] ** 2 + w**2 * e.x[-1, 0] ** 2) / w**2
    assert energy_ratio > 1.5


@pytest.mark.parametrize(("step", "order"), [(euler_step, 1), (rk4_step, 4)])
def test_convergence_order(step, order):
    """Diviser dt par 2 divise l'erreur globale par 2^ordre."""

    def global_error(dt):
        x = np.array([1.0])
        n = round(1.0 / dt)
        for k in range(n):
            x = step(decay, k * dt, x, np.zeros(0), dt)
        return abs(x[0] - math.exp(-1.0))

    ratio = global_error(0.02) / global_error(0.01)
    assert ratio == pytest.approx(2**order, rel=0.1)


def test_controller_zero_order_hold_and_frame_skip():
    calls = []

    def controller(t, _x):
        calls.append(t)
        return np.array([t])

    res = simulate(
        lambda _t, x, u: np.zeros_like(x),
        np.zeros(1),
        1.0,
        dt=0.01,
        controller=controller,
        control_dt=0.05,
    )
    assert len(calls) == 20  # une décision toutes les 5 étapes physiques
    np.testing.assert_allclose(calls, np.arange(20) * 0.05, atol=1e-12)
    # la commande est maintenue entre deux décisions
    np.testing.assert_allclose(res.u[:5, 0], 0.0)
    np.testing.assert_allclose(res.u[5:10, 0], 0.05)
    assert res.x.shape == (101, 1)
    assert res.t[-1] == pytest.approx(1.0)


def test_post_step_is_applied():
    res = simulate(
        lambda _t, x, _u: np.ones_like(x),
        np.zeros(2),
        0.1,
        dt=0.01,
        post_step=lambda x: x / max(1.0, np.linalg.norm(x) / 0.05),
    )
    assert np.linalg.norm(res.x[-1]) <= 0.05 + 1e-12


def test_substeps_per_control():
    assert substeps_per_control(0.01, 0.1) == 10
    assert substeps_per_control(0.01, 0.01) == 1
    with pytest.raises(ValueError):
        substeps_per_control(0.01, 0.033)
    with pytest.raises(ValueError):
        substeps_per_control(0.01, 0.001)
