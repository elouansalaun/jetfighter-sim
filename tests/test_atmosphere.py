"""Validation du modèle ISA contre les valeurs tabulées de l'US Standard Atmosphere 1976."""

import math

import numpy as np
import pytest

from jetsim.core import constants as c
from jetsim.core.atmosphere import (
    H_MAX,
    dynamic_pressure,
    equivalent_airspeed,
    geometric_altitude,
    geopotential_altitude,
    isa,
    mach_number,
)

# (altitude géopotentielle [m], T [K], P [Pa], rho [kg/m³]) — US Std Atm 1976
GEOPOTENTIAL_TABLE = [
    (0.0, 288.15, 101_325.0, 1.2250),
    (11_000.0, 216.65, 22_632.06, 0.36392),
    (20_000.0, 216.65, 5_474.89, 0.088035),
    (32_000.0, 228.65, 868.02, 0.013225),
]

# (altitude géométrique [m], T [K], P [Pa], rho [kg/m³], a [m/s]) — US Std Atm 1976
GEOMETRIC_TABLE = [
    (1_000.0, 281.651, 89_874.6, 1.11164, 336.435),
    (5_000.0, 255.676, 54_048.3, 0.736429, 320.545),
    (10_000.0, 223.252, 26_499.9, 0.413510, 299.532),
    (15_000.0, 216.650, 12_111.8, 0.194755, 295.070),
]


def test_sea_level_constants():
    assert c.RHO0 == pytest.approx(1.225, rel=1e-4)
    assert c.A0 == pytest.approx(340.294, rel=1e-5)
    atm = isa(0.0)
    assert atm.temperature == pytest.approx(c.T0)
    assert atm.pressure == pytest.approx(c.P0)
    assert atm.density == pytest.approx(c.RHO0)
    assert atm.speed_of_sound == pytest.approx(c.A0)


@pytest.mark.parametrize(("H", "T", "P", "rho"), GEOPOTENTIAL_TABLE)
def test_layer_boundaries_geopotential(H, T, P, rho):
    atm = isa(H, geometric=False)
    assert atm.temperature == pytest.approx(T, rel=1e-6)
    assert atm.pressure == pytest.approx(P, rel=2e-5)
    assert atm.density == pytest.approx(rho, rel=1e-4)


@pytest.mark.parametrize(("h", "T", "P", "rho", "a"), GEOMETRIC_TABLE)
def test_geometric_table(h, T, P, rho, a):
    atm = isa(h)
    assert atm.temperature == pytest.approx(T, rel=1e-5)
    assert atm.pressure == pytest.approx(P, rel=1e-4)
    assert atm.density == pytest.approx(rho, rel=1e-4)
    assert atm.speed_of_sound == pytest.approx(a, rel=1e-5)


def test_continuity_at_layer_boundaries():
    """Pas de saut de T, P, rho aux changements de couche."""
    for H in (11_000.0, 20_000.0):
        below = isa(H - 1e-6, geometric=False)
        above = isa(H + 1e-6, geometric=False)
        for b, a in zip(below, above, strict=True):
            assert b == pytest.approx(a, rel=1e-8)


def test_monotonic_pressure_and_density():
    h = np.linspace(0.0, 30_000.0, 3001)
    atm = isa(h)
    assert np.all(np.diff(atm.pressure) < 0)
    assert np.all(np.diff(atm.density) < 0)


def test_vectorized_matches_scalar():
    h = np.array([-500.0, 0.0, 3_000.0, 11_000.0, 15_000.0, 25_000.0])
    vec = isa(h)
    for i, hi in enumerate(h):
        sca = isa(float(hi))
        for field_vec, field_sca in zip(vec, sca, strict=True):
            assert field_vec[i] == pytest.approx(field_sca, rel=1e-12)


def test_clipping_outside_domain():
    top = isa(H_MAX, geometric=False)
    assert isa(100_000.0, geometric=False).pressure == pytest.approx(top.pressure)
    assert math.isfinite(isa(-5_000.0).density)


def test_geopotential_roundtrip():
    h = np.linspace(0.0, 30_000.0, 7)
    np.testing.assert_allclose(geometric_altitude(geopotential_altitude(h)), h, rtol=1e-12)
    # ≈ 63 m d'écart à 20 km
    assert 20_000.0 - geopotential_altitude(20_000.0) == pytest.approx(62.7, abs=0.5)


def test_derived_quantities():
    V = 250.0
    h = 10_000.0
    atm = isa(h)
    assert mach_number(V, h) == pytest.approx(V / atm.speed_of_sound)
    assert dynamic_pressure(V, h) == pytest.approx(0.5 * atm.density * V**2)
    assert equivalent_airspeed(V, 0.0) == pytest.approx(V)
    # même pression dynamique qu'au niveau de la mer à la vitesse EAS
    eas = equivalent_airspeed(V, h)
    assert 0.5 * c.RHO0 * eas**2 == pytest.approx(dynamic_pressure(V, h))
