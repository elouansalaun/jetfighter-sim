"""Loading of aircraft parameters, drag-polar aerodynamics and propulsion."""

import itertools
import math

import pytest

from jetsim.aircraft.aero_polar import PolarAero, interp1
from jetsim.aircraft.params import AircraftParams, load_aircraft
from jetsim.aircraft.propulsion import SimpleTurbofan
from jetsim.core.atmosphere import isa_scalar
from jetsim.core.constants import RHO0


@pytest.fixture(scope="module")
def f16() -> AircraftParams:
    return load_aircraft("f16")


def test_load_f16_units(f16):
    assert f16.geometry.wing_area == pytest.approx(300 * 0.3048**2, rel=1e-3)
    assert f16.geometry.aspect_ratio == pytest.approx(3.0, rel=0.01)
    assert f16.mass.mass == pytest.approx(1 / 1.57e-3 * 14.5939029, rel=1e-5)  # 1/m S&L
    assert f16.limits.alpha_max == pytest.approx(math.radians(25))
    assert f16.limits.roll_rate_max == pytest.approx(math.radians(240))
    assert f16.control_surfaces["elevator"].max == pytest.approx(math.radians(25))
    assert f16.mass.inertia_tensor[0, 2] == pytest.approx(-f16.mass.Ixz)


def test_load_unknown_aircraft():
    with pytest.raises(FileNotFoundError):
        load_aircraft("nonexistent_aircraft")


def test_polar_tables_must_match_mach(f16):
    import dataclasses

    with pytest.raises(ValueError):
        dataclasses.replace(f16.aero_polar, cd0=(0.02, 0.03))


def test_interp1_saturates_and_interpolates():
    xs, ys = (0.0, 1.0, 2.0), (0.0, 10.0, 0.0)
    assert interp1(-1.0, xs, ys) == 0.0
    assert interp1(0.5, xs, ys) == pytest.approx(5.0)
    assert interp1(1.5, xs, ys) == pytest.approx(5.0)
    assert interp1(9.0, xs, ys) == 0.0


def test_polar_lift_and_drag(f16):
    aero = PolarAero(f16.aero_polar)
    a = math.radians(10)
    # low-speed slope ≈ 0.061/deg, consistent with the F-16 tables (CL(10°) ≈ 0.7)
    assert aero.cl(a, 0.0) == pytest.approx(0.1 + 3.5 * a)
    assert aero.cl(a, 0.9) > aero.cl(a, 0.0)  # lift slope rises in the transonic regime
    assert aero.alpha_for_cl(aero.cl(a, 0.85), 0.85) == pytest.approx(a)
    # transonic drag rise
    assert aero.cd0(1.1) > 2 * aero.cd0(0.5)
    # plausible max lift-to-drag ratio for a fighter (≈ 8–12 subsonic)
    assert 8.0 < aero.max_lift_to_drag(0.5) < 12.0


def test_thrust_levels_and_lapse(f16):
    eng = SimpleTurbofan(f16.propulsion)
    p = f16.propulsion
    assert eng.thrust(0.0, RHO0, 0.0) == pytest.approx(p.thrust_idle_sl)
    assert eng.thrust(p.mil_power, RHO0, 0.0) == pytest.approx(p.thrust_mil_sl)
    assert eng.thrust(1.0, RHO0, 0.0) == pytest.approx(p.thrust_max_sl)
    assert eng.thrust(2.0, RHO0, 0.0) == pytest.approx(p.thrust_max_sl)  # saturation
    # monotonic in power
    powers = [i / 50 for i in range(51)]
    thrusts = [eng.thrust(pw, RHO0, 0.8) for pw in powers]
    assert all(b >= a for a, b in itertools.pairwise(thrusts))
    # decreases with altitude
    assert eng.thrust(1.0, isa_scalar(10_000.0)[2], 0.0) < 0.5 * p.thrust_max_sl
    # ram effect bounded by the engine limit
    assert eng.thrust(1.0, RHO0, 1.5) == pytest.approx(p.ram_limit * p.thrust_max_sl)


@pytest.mark.parametrize(
    ("alt_ft", "mil_ratio", "max_ratio"),
    # Thrust(alt)/thrust(sea level) ratios at Mach 0 from the Stevens & Lewis tables (F100 engine,
    # src/jetsim/data/aircraft/f16_sl_tables.yaml); 12 % tolerance.
    [
        (10_000, 9_150 / 12_680, 15_000 / 20_000),
        (30_000, 3_950 / 12_680, 7_000 / 20_000),
        (40_000, 2_450 / 12_680, 4_000 / 20_000),
        (50_000, 1_400 / 12_680, 2_500 / 20_000),
    ],
)
def test_static_thrust_lapse_matches_reference(f16, alt_ft, mil_ratio, max_ratio):
    eng = SimpleTurbofan(f16.propulsion)
    factor = eng.altitude_factor(isa_scalar(alt_ft * 0.3048)[2])
    assert factor == pytest.approx(0.5 * (mil_ratio + max_ratio), rel=0.12)


def test_engine_lag(f16):
    eng = SimpleTurbofan(f16.propulsion)
    assert eng.power_rate(0.2, 1.0) == pytest.approx(0.8 / f16.propulsion.engine_time_constant)
    assert eng.power_rate(0.5, 0.5) == 0.0
    assert eng.power_rate(0.5, -3.0) < 0.0
