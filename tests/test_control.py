"""Phase 6: PID, fly-by-wire flight controls, autopilot, LQR, maneuvers."""

import math

import numpy as np
import pytest

from jetsim.aircraft import dynamics_3dof as d3
from jetsim.aircraft import dynamics_6dof as d6
from jetsim.aircraft.instruments import read_instruments
from jetsim.aircraft.params import load_aircraft
from jetsim.control.autopilot import Autopilot, AutopilotTargets
from jetsim.control.fbw import (
    FlyByWire,
    HighLevelCommand,
    PointMassInnerLoop,
    make_inner_loop,
)
from jetsim.control.lqr import LongitudinalLQR, lqr
from jetsim.control.maneuvers import (
    beam_heading,
    bearing_to,
    break_turn,
    drag_heading,
    relative_bearing,
)
from jetsim.control.pid import PID, Washout, wrap_angle
from jetsim.core.constants import G0

DEG = math.pi / 180
DT = 0.02  # control loop period (50 Hz), 2 physics steps


def model(kind: str, xcg: float | None = None):
    params = load_aircraft("f16")
    return d6.F16SixDof(params, xcg=xcg) if kind == "6dof" else d3.PointMassAircraft(params)


def trim(m, h, v):
    return m.trim(h, v) if isinstance(m, d6.F16SixDof) else m.trimmed_state(h, v)


def fly_inner(m, h, v, command, t_end):
    """Fly under the inner loop; ``command(t)`` -> HighLevelCommand. Returns the log."""
    x, u0 = trim(m, h, v)
    loop = make_inner_loop(m)
    loop.reset(x)
    log, t = [], 0.0
    while t < t_end:
        ins = read_instruments(m, x)
        u = loop(ins, command(t, u0), DT)
        for _ in range(2):
            x = m.step(x, u)
        t = round(t + DT, 9)
        log.append((t, ins.nz, ins.p, ins.beta, ins.alpha, ins.altitude))
    return np.array(log)


def fly_autopilot(m, h, v, targets, t_end):
    x, _ = trim(m, h, v)
    ap = Autopilot(m)
    ins = read_instruments(m, x)
    ap.reset(x, ins)
    log, t = [], 0.0
    while t < t_end:
        ap.targets = targets(t)
        u = ap.controls(ins, DT)
        for _ in range(2):
            x = m.step(x, u)
        t = round(t + DT, 9)
        ins = read_instruments(m, x)
        log.append((t, ins.altitude, ins.tas, ins.course, ins.roll, ins.nz, ins.beta))
    return np.array(log)


def response(log, col, target, t0):
    """(90 % rise time, overshoot [%], final value)."""
    t, y = log[:, 0], log[:, col]
    y0 = y[0]
    after = t > t0
    reached = np.nonzero(after & ((y - y0) >= 0.9 * (target - y0)))[0]
    rise = t[reached[0]] - t0 if reached.size else math.inf
    overshoot = (y[after].max() - target) / (target - y0) * 100
    return rise, overshoot, y[-1]


# --------------------------------------------------------------------------
# PID and filters
# --------------------------------------------------------------------------
def test_pid_proportional_and_integral():
    pid = PID(kp=2.0, ki=0.5)
    assert pid(1.0, dt=0.1) == pytest.approx(2.0 + 0.5 * 0.1)
    for _ in range(9):
        out = pid(1.0, dt=0.1)
    assert pid.integral == pytest.approx(1.0)
    assert out == pytest.approx(2.0 + 0.5 * 1.0)


def test_pid_anti_windup_and_bumpless_reset():
    pid = PID(kp=1.0, ki=1.0, out_min=-1.0, out_max=1.0)
    for _ in range(100):
        out = pid(5.0, dt=0.1)
    assert out == 1.0
    assert pid.integral < 1.0  # frozen during saturation
    pid.reset(output=0.3)
    assert pid(0.0, dt=0.01) == pytest.approx(0.3)


def test_pid_derivative_on_measurement():
    pid = PID(kp=0.0, kd=1.0, derivative_tau=0.0)
    pid(0.0, dt=0.1, measurement=0.0)
    assert pid(0.0, dt=0.1, measurement=1.0) == pytest.approx(-10.0)  # opposes the increase


def test_washout_and_wrap():
    w = Washout(tau=1.0)
    first = w(1.0, 0.01)
    for _ in range(1000):
        last = w(1.0, 0.01)
    assert first > 0.98 and abs(last) < 1e-3
    assert wrap_angle(3 * math.pi / 2) == pytest.approx(-math.pi / 2)


# --------------------------------------------------------------------------
# 6-DOF fly-by-wire flight controls
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def f16() -> d6.F16SixDof:
    return model("6dof")


@pytest.mark.parametrize(("h", "v"), [(0, 150), (3000, 200), (6000, 250)])
def test_fbw_load_factor_step(f16, h, v):
    log = fly_inner(f16, h, v, lambda t, u0: HighLevelCommand(4.0 if t > 0.5 else 1.0, 0.0,
                                                              u0[0]), 3.0)  # fmt: skip
    rise, overshoot, final = response(log, 1, 4.0, 0.5)
    assert rise < 1.0
    assert overshoot < 15.0
    assert final == pytest.approx(4.0, abs=0.25)


@pytest.mark.parametrize(("h", "v"), [(0, 150), (3000, 200), (6000, 250)])
def test_fbw_roll_rate_step_keeps_sideslip_small(f16, h, v):
    p_cmd = 90 * DEG
    log = fly_inner(f16, h, v, lambda t, u0: HighLevelCommand(1.0, p_cmd if t > 0.5 else 0.0,
                                                              u0[0]), 2.0)  # fmt: skip
    rise, overshoot, final = response(log, 2, p_cmd, 0.5)
    assert rise < 0.4
    assert overshoot < 15.0
    assert final == pytest.approx(p_cmd, rel=0.05)
    assert np.abs(log[:, 3]).max() < 1.5 * DEG


def test_fbw_alpha_limiter_prevents_stall(f16):
    """9 g requested at low speed: the angle of attack stays below 25°."""
    log = fly_inner(f16, 3000, 130, lambda t, u0: HighLevelCommand(9.0, 0.0, 1.0), 5.0)
    assert log[:, 4].max() < 25.5 * DEG
    assert log[-1, 1] < 3.0  # the available lift does not allow more


def test_fbw_negative_g(f16):
    log = fly_inner(f16, 3000, 200, lambda t, u0: HighLevelCommand(-2.0 if t > 0.5 else 1.0,
                                                                   0.0, u0[0]), 3.0)  # fmt: skip
    assert log[-1, 1] == pytest.approx(-2.0, abs=0.15)


@pytest.mark.parametrize("xcg", [0.35, 0.40])
def test_fbw_stabilizes_unstable_airframe(xcg):
    """Unstable CG position: the fly-by-wire flight controls hold 1 g without diverging."""
    m = model("6dof", xcg=xcg)
    log = fly_inner(m, 3000, 200, lambda t, u0: HighLevelCommand(1.0, 0.0, u0[0]), 15.0)
    assert np.abs(log[:, 1] - 1.0).max() < 0.05
    assert abs(log[-1, 5] - 3000) < 20


def test_point_mass_inner_loop_tracks_load_factor():
    m = model("3dof")
    log = fly_inner(m, 3000, 250, lambda t, u0: HighLevelCommand(4.0 if t > 0.5 else 1.0, 0.0,
                                                                 0.9), 4.0)  # fmt: skip
    assert log[-1, 1] == pytest.approx(4.0, abs=0.08)


def test_make_inner_loop_dispatch(f16):
    assert isinstance(make_inner_loop(f16), FlyByWire)
    assert isinstance(make_inner_loop(model("3dof")), PointMassInnerLoop)
    with pytest.raises(TypeError):
        make_inner_loop(object())  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# Autopilot
# --------------------------------------------------------------------------
@pytest.mark.parametrize("kind", ["3dof", "6dof"])
def test_autopilot_altitude_heading_speed(kind):
    def targets(t):
        if t < 2:
            return AutopilotTargets(altitude=3000, heading=0.0, airspeed=200)
        return AutopilotTargets(altitude=3500, heading=90 * DEG, airspeed=230)

    log = fly_autopilot(model(kind), 3000, 200, targets, 60.0)
    assert log[-1, 1] == pytest.approx(3500, abs=15)
    assert log[:, 1].max() < 3520  # no significant overshoot
    assert wrap_angle(log[-1, 3] - 90 * DEG) == pytest.approx(0, abs=1 * DEG)
    assert log[-1, 2] == pytest.approx(230, abs=2)
    assert log[:, 5].max() < 4.5  # gentle maneuvers


@pytest.mark.parametrize("n", [4.0, 5.0])
def test_level_turn_consistent_between_models(n):
    """3-DOF / 6-DOF consistency in a steady turn at imposed bank: altitude held,
    turn rate matching ω = g·√(n²−1)/V (measured n), and close between the two models."""
    bank = math.acos(1 / n)
    rates = {}
    for kind in ("3dof", "6dof"):

        def targets(t):
            return AutopilotTargets(altitude=3000, airspeed=250, bank=bank if t > 2 else 0.0)

        log = fly_autopilot(model(kind), 3000, 250, targets, 35.0)
        steady = log[:, 0] > 20
        rate = np.gradient(np.unwrap(log[:, 3]), log[:, 0])[steady].mean()
        nz, v = log[steady, 5].mean(), log[steady, 2].mean()
        assert nz == pytest.approx(n, rel=0.06), kind
        assert rate == pytest.approx(G0 * math.sqrt(nz * nz - 1) / v, rel=0.02), kind
        assert np.abs(log[steady, 1] - 3000).max() < 40, kind
        assert v == pytest.approx(250, abs=2), kind
        rates[kind] = rate
    assert rates["3dof"] == pytest.approx(rates["6dof"], rel=0.05)


def test_pid_integral_zone():
    pid = PID(kp=0.0, ki=1.0, integral_zone=0.5)
    pid(2.0, dt=1.0)
    assert pid.integral == 0.0  # error too large: no integration
    pid(0.4, dt=1.0)
    assert pid.integral == pytest.approx(0.4)


# --------------------------------------------------------------------------
# LQR
# --------------------------------------------------------------------------
def test_lqr_textbook_double_integrator():
    A = np.array([[0.0, 1.0], [0.0, 0.0]])
    B = np.array([[0.0], [1.0]])
    K, _, eig = lqr(A, B, np.eye(2), np.eye(1))
    np.testing.assert_allclose(K, [[1.0, math.sqrt(3)]], atol=1e-9)  # analytical solution
    assert np.all(eig.real < 0)


@pytest.mark.parametrize("xcg", [0.35, 0.40])
def test_lqr_stabilizes_unstable_airframe(xcg):
    m = model("6dof", xcg=xcg)
    ctrl = LongitudinalLQR.design(m, 3000, 200)
    assert ctrl.open_loop.real.max() > 0  # unstable aircraft
    assert ctrl.closed_loop.real.max() < 0  # regulated: stable

    def run(controlled: bool) -> float:
        x, u = m.trim(3000, 200)
        x[d6.Q] += 5 * DEG  # gust: pitch disturbance
        for _ in range(750):  # 15 s
            ins = read_instruments(m, x)
            uu = ctrl(ins) if controlled else u
            for _ in range(2):
                x = m.step(x, uu)
            if abs(ins.alpha) > 30 * DEG:
                break
        return read_instruments(m, x).alpha

    assert abs(run(True) - ctrl.s_eq[1]) < 0.3 * DEG
    assert abs(run(False) - ctrl.s_eq[1]) > 3 * DEG  # without the regulator: divergence


# --------------------------------------------------------------------------
# Evasive maneuvers (geometry)
# --------------------------------------------------------------------------
def test_bearings():
    assert bearing_to(0, 0, 1000, 0) == pytest.approx(0.0)
    assert bearing_to(0, 0, 0, 1000) == pytest.approx(math.pi / 2)
    assert relative_bearing(math.pi / 2, 0.0) > 0  # threat on the right
    assert relative_bearing(-math.pi / 2, 0.0) < 0


def test_beam_and_drag_headings():
    # threat to the East (090); flying North: beam = heading 000 (already), otherwise 180
    assert beam_heading(math.pi / 2, 0.1) == pytest.approx(0.0, abs=1e-12)
    assert abs(beam_heading(math.pi / 2, math.pi - 0.1)) == pytest.approx(math.pi)
    assert drag_heading(math.pi / 2) == pytest.approx(-math.pi / 2)


def test_break_turn_rolls_toward_threat(f16):
    x, _ = f16.trim(3000, 250)
    ins = read_instruments(f16, x)
    right = break_turn(ins, threat_bearing=math.pi / 2)
    left = break_turn(ins, threat_bearing=-math.pi / 2)
    assert right.roll_rate > 0 > left.roll_rate
    assert right.nz == 9.0 and right.throttle == 1.0
