"""Phase 4 : vitesses anémométriques, tableau de bord, capteurs, enveloppe de vol."""

import dataclasses
import math

import numpy as np
import pytest

from jetfighter.aircraft import dynamics_3dof as d3
from jetfighter.aircraft import dynamics_6dof as d6
from jetfighter.aircraft.envelope import EnvelopeMonitor, Violation
from jetfighter.aircraft.instruments import INSTRUMENT_NAMES, Instruments, read_instruments
from jetfighter.aircraft.params import CONFIG_DIR, load_aircraft, load_envelope
from jetfighter.aircraft.sensors import ChannelNoise, SensorSuite
from jetfighter.core.atmosphere import (
    A0,
    P0,
    calibrated_airspeed,
    equivalent_airspeed,
    impact_pressure,
    true_airspeed_from_calibrated,
)
from jetfighter.core.constants import G0
from jetfighter.core.integrators import simulate

DEG = math.pi / 180


@pytest.fixture(scope="module")
def pm() -> d3.PointMassAircraft:
    return d3.PointMassAircraft(load_aircraft("f16"))


@pytest.fixture(scope="module")
def f16() -> d6.F16SixDof:
    return d6.F16SixDof(load_aircraft("f16"))


# --------------------------------------------------------------------------
# Vitesses anémométriques
# --------------------------------------------------------------------------
@pytest.mark.parametrize("v", [50.0, 200.0, 340.0, 400.0, 600.0])
def test_cas_equals_tas_at_sea_level(v):
    assert calibrated_airspeed(v, 0.0) == pytest.approx(v, rel=1e-12)


def test_impact_pressure_continuous_at_mach_one():
    below = impact_pressure(1.0 - 1e-9, P0)
    above = impact_pressure(1.0 + 1e-9, P0)
    assert below == pytest.approx(above, rel=1e-6)
    # formule isentropique à Mach 1 : qc/P = 1.2^3.5 − 1
    assert impact_pressure(1.0, 1.0) == pytest.approx(1.2**3.5 - 1)


@pytest.mark.parametrize(("v", "h"), [(240, 10_000), (150, 3000), (500, 11_000), (700, 15_000)])
def test_cas_roundtrip_and_ordering(v, h):
    cas = calibrated_airspeed(v, h)
    assert true_airspeed_from_calibrated(cas, h) == pytest.approx(v, rel=1e-10)
    eas = float(equivalent_airspeed(v, h))
    assert eas < cas < v  # en altitude : EAS < CAS < TAS


def test_cas_known_value():
    """À 10 000 m, 240 m/s vrais (M 0.80) : EAS ≈ 139 m/s, CAS ≈ 147 m/s
    (correction de compressibilité ≈ +8 m/s)."""
    assert float(equivalent_airspeed(240.0, 10_000.0)) == pytest.approx(139.4, abs=0.1)
    assert calibrated_airspeed(240.0, 10_000.0) == pytest.approx(147.4, abs=0.1)
    assert calibrated_airspeed(A0, 0.0) == pytest.approx(A0)


# --------------------------------------------------------------------------
# Tableau de bord
# --------------------------------------------------------------------------
def test_instrument_vector_and_names(f16):
    x, _ = f16.trim(3000, 200)
    ins = read_instruments(f16, x)
    assert set(ins.to_dict()) == set(INSTRUMENT_NAMES)
    np.testing.assert_allclose(ins.vector(("tas", "altitude")), [200.0, 3000.0])


def test_unknown_model_rejected():
    with pytest.raises(TypeError):
        read_instruments(object(), np.zeros(3))  # type: ignore[arg-type]


@pytest.mark.parametrize("which", ["pm", "f16"])
def test_level_flight_dashboard(request, which):
    """Palier stabilisé : n_z ≈ cos θ, n_x ≈ sin θ (l'accéléromètre voit −g), Ps ≈ 0,
    assiette = incidence, variomètre nul."""
    model = request.getfixturevalue(which)
    x, _ = model.trim(3000.0, 200.0) if which == "f16" else model.trimmed_state(3000.0, 200.0)
    ins = read_instruments(model, x)
    assert ins.tas == pytest.approx(200.0)
    assert ins.pitch == pytest.approx(ins.alpha, abs=1e-9)
    assert ins.gamma == pytest.approx(0.0, abs=1e-9)
    assert ins.vertical_speed == pytest.approx(0.0, abs=1e-6)
    assert ins.nz == pytest.approx(math.cos(ins.pitch), abs=1e-6)
    assert ins.nx == pytest.approx(math.sin(ins.pitch), abs=1e-6)
    assert ins.specific_excess_power == pytest.approx(0.0, abs=1e-6)
    assert ins.specific_energy == pytest.approx(3000 + 200**2 / (2 * G0), rel=1e-4)
    assert ins.cas < ins.tas


def test_point_mass_body_rates_in_level_turn(pm):
    """Virage stabilisé à n = 4 : en axes corps, p = −ω·sin θ, q = ω·sin φ·cos θ,
    r = ω·cos φ·cos θ, avec ω le taux de virage."""
    x, _ = pm.trimmed_state(3000.0, 250.0, load_factor=4.0)
    ins = read_instruments(pm, x)
    omega = G0 * math.sqrt(15) / 250.0
    th, ph = ins.pitch, ins.roll
    # la gîte du fuselage diffère légèrement de l'inclinaison du vecteur vitesse (μ),
    # car le fuselage est cabré de α par rapport à la vitesse
    assert abs(ph - math.acos(1 / 4)) < 0.2 * DEG
    np.testing.assert_allclose(
        [ins.p, ins.q, ins.r],
        [-omega * math.sin(th), omega * math.sin(ph) * math.cos(th),
         omega * math.cos(ph) * math.cos(th)],
        atol=1e-9,
    )  # fmt: skip
    # f_corps = C_bw·f_vent avec T·cos α = D à l'équilibre : n_z = 4·cos α
    assert ins.nz == pytest.approx(4.0 * math.cos(ins.alpha), rel=1e-9)


@pytest.mark.parametrize("which", ["pm", "f16"])
def test_velocity_bank_angle(request, which):
    """Inclinaison μ du vecteur portance : nulle en palier ; en virage stabilisé 3-DOF,
    égale à acos(1/n) ; proche de la gîte φ du fuselage quand l'incidence est faible."""
    model = request.getfixturevalue(which)
    if which == "pm":
        x, _ = model.trimmed_state(3000.0, 250.0, load_factor=3.0)
        ins = read_instruments(model, x)
        assert ins.bank == pytest.approx(math.acos(1 / 3), abs=1e-9)
    else:
        x, _ = model.trim(3000.0, 250.0)
        x = model.make_state(altitude=3000.0, airspeed=250.0, alpha=2 * DEG, roll=40 * DEG,
                             pitch=2 * DEG * math.cos(40 * DEG))  # fmt: skip
        ins = read_instruments(model, x)
        assert ins.bank == pytest.approx(ins.roll, abs=1.5 * DEG)
    assert abs(ins.bank - ins.roll) < 2 * DEG


def test_point_mass_attitude_in_climb(pm):
    x, _ = pm.trimmed_state(2000.0, 220.0, gamma=12 * DEG)
    ins = read_instruments(pm, x)
    assert ins.pitch == pytest.approx(12 * DEG + ins.alpha, abs=1e-9)
    assert ins.roll == pytest.approx(0.0, abs=1e-9)


def test_models_agree_on_level_flight(pm, f16):
    """Même point de vol : les deux modèles donnent des tableaux de bord proches."""
    a = read_instruments(pm, pm.trimmed_state(6000.0, 250.0)[0])
    b = read_instruments(f16, f16.trim(6000.0, 250.0)[0])
    for name in ("tas", "cas", "mach", "altitude", "nz", "specific_energy"):
        assert getattr(a, name) == pytest.approx(getattr(b, name), rel=2e-3), name
    assert abs(a.alpha - b.alpha) < 1 * DEG


def test_6dof_pull_up_load_factor_and_energy(f16):
    """Ressource : n_z monte au-dessus de 1 ; E + ∫(−Ps) reste cohérent (Ė = Ps)."""
    x0, u0 = f16.trim(3000.0, 220.0)
    u = u0.copy()
    u[d6.ELEVATOR] -= 4 * DEG
    res = simulate(f16.derivatives, x0, 3.0, u_const=u, post_step=f16.post_step)
    data = [read_instruments(f16, x) for x in res.x]
    assert max(d.nz for d in data) > 3.0
    energy = np.array([d.specific_energy for d in data])
    ps = np.array([d.specific_excess_power for d in data])
    integral = float(np.sum(0.5 * (ps[1:] + ps[:-1]) * np.diff(res.t)))
    assert energy[-1] - energy[0] == pytest.approx(integral, rel=1e-3, abs=0.05)


# --------------------------------------------------------------------------
# Capteurs
# --------------------------------------------------------------------------
@pytest.fixture
def truth(f16) -> Instruments:
    return read_instruments(f16, f16.trim(3000.0, 200.0)[0])


def test_perfect_sensors_return_truth(truth):
    sensors = SensorSuite.perfect()
    assert sensors.is_perfect
    assert sensors.measure(truth) is truth


def test_noise_statistics_and_bias(truth):
    rng = np.random.default_rng(0)
    sensors = SensorSuite({"alpha": ChannelNoise(std=0.01, bias_std=0.02)}, rng)
    samples = np.array([sensors.measure(truth).alpha for _ in range(20_000)]) - truth.alpha
    assert samples.mean() == pytest.approx(sensors.bias["alpha"], abs=5e-4)
    assert samples.std() == pytest.approx(0.01, rel=0.03)
    # le biais est fixe pendant l'épisode et retiré au reset
    b0 = sensors.bias["alpha"]
    sensors.reset()
    assert sensors.bias["alpha"] != b0
    # les autres voies ne sont pas touchées
    m = sensors.measure(truth)
    assert m.altitude == truth.altitude and m.tas == truth.tas


def test_sensor_reproducibility(truth):
    def run(seed):
        s = SensorSuite({"nz": ChannelNoise(0.01, 0.01)}, np.random.default_rng(seed))
        return [s.measure(truth).nz for _ in range(5)]

    assert run(7) == run(7)
    assert run(7) != run(8)


def test_sensor_yaml_config(truth):
    sensors = SensorSuite.from_yaml(
        CONFIG_DIR.parent / "sensors" / "realistic.yaml", np.random.default_rng(1)
    )
    assert sensors.noise["alpha"].std == pytest.approx(0.2 * DEG)
    assert sensors.noise["altitude"].bias_std == 10.0
    m = sensors.measure(truth)
    assert abs(m.altitude - truth.altitude) < 60.0
    assert abs(m.alpha - truth.alpha) < 3 * DEG


def test_unknown_sensor_channel():
    with pytest.raises(ValueError, match="inconnues"):
        SensorSuite({"vitesse_lumiere": ChannelNoise(1.0)})


# --------------------------------------------------------------------------
# Enveloppe
# --------------------------------------------------------------------------
@pytest.fixture
def monitor() -> EnvelopeMonitor:
    return EnvelopeMonitor(load_envelope("f16"))


def test_envelope_config():
    common, six = load_envelope("f16"), load_envelope("f16", six_dof=True)
    assert common.max_mach == 2.0 and six.max_mach == 0.95
    assert common.alpha_stall == pytest.approx(30 * DEG)


def test_nominal_flight_passes(monitor, truth):
    for _ in range(100):
        assert monitor.check(truth, dt=0.1) is None
    assert monitor.message == ""


@pytest.mark.parametrize(
    ("changes", "expected"),
    [
        ({"altitude": -1.0}, Violation.GROUND),
        ({"nz": 10.5}, Violation.OVERLOAD),
        ({"nz": -4.5}, Violation.OVERLOAD),
        ({"altitude": 17_500.0}, Violation.CEILING),
        ({"tas": 40.0}, Violation.LOW_SPEED),
        ({"mach": 2.1}, Violation.OVERSPEED),
        ({"beta": 35 * DEG}, Violation.SIDESLIP),
        ({"alpha": math.nan}, Violation.NUMERICAL),
        ({"altitude": -5.0, "nz": 12.0}, Violation.GROUND),  # priorité au sol
    ],
)
def test_each_violation(monitor, truth, changes, expected):
    ins = dataclasses.replace(truth, **changes)
    assert monitor.check(ins, dt=0.1) == expected
    assert expected.value in monitor.message


def test_nan_in_state_is_detected(monitor, truth):
    state = np.zeros(17)
    state[3] = np.inf
    assert monitor.check(truth, dt=0.1, state=state) == Violation.NUMERICAL


def test_stall_needs_duration_and_resets(monitor, truth):
    stalled = dataclasses.replace(truth, alpha=35 * DEG)
    for _ in range(20):  # 2.0 s tolérées
        assert monitor.check(stalled, dt=0.1) is None
    assert monitor.check(stalled, dt=0.1) == Violation.STALL
    monitor.reset()
    for _ in range(15):
        monitor.check(stalled, dt=0.1)
    monitor.check(truth, dt=0.1)  # sortie du décrochage : chrono remis à zéro
    for _ in range(15):
        assert monitor.check(stalled, dt=0.1) is None


def test_six_dof_overspeed_limit(truth):
    monitor = EnvelopeMonitor(load_envelope("f16", six_dof=True))
    assert monitor.check(dataclasses.replace(truth, mach=0.97), dt=0.1) == Violation.OVERSPEED


def test_crash_detected_in_simulation(f16):
    """Piqué plein gaz depuis 1000 m : l'enveloppe arrête le vol au sol ou en survitesse."""
    monitor = EnvelopeMonitor(load_envelope("f16", six_dof=True))
    x, u0 = f16.trim(1000.0, 200.0)
    u = u0.copy()
    u[d6.ELEVATOR] += 6 * DEG
    u[d6.THROTTLE] = 1.0
    violation, t = None, 0.0
    while violation is None and t < 60.0:
        for _ in range(10):
            x = f16.step(x, u)
        t += 0.1
        violation = monitor.check(read_instruments(f16, x), dt=0.1, state=x)
    assert violation in (Violation.GROUND, Violation.OVERSPEED, Violation.OVERLOAD)
    assert t < 30.0
