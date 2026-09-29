"""Inner loop ("fly-by-wire flight controls"): high-level commands -> control surfaces.

All high-level controllers in the project (autopilot, evasion
heuristics, and the RL agent in hierarchical mode) speak the same language ::

    HighLevelCommand(nz=…, roll_rate=…, throttle=…)

* ``nz``        : requested load factor [g] (1 = level flight; 9 = 9 g pull-up);
* ``roll_rate`` : requested roll rate [rad/s] (> 0: to the right);
* ``throttle``  : throttle [0, 1].

``make_inner_loop(model)`` returns the inner loop suited to the model, which converts these
setpoints into model controls.

6-DOF — control laws (gains in ``FbwGains``, multiplied by q̄_ref/q̄ to keep the
same dynamics as dynamic pressure varies):

* pitch: PI on the n_z error + q damping; δe < 0 pitches nose up;
* angle-of-attack limiter: as soon as α approaches α_max, the n_z setpoint is reduced to what
  keeps α ≤ α_max (same on the negative side) — the aircraft cannot be stalled;
* load-factor limiter: n_z ∈ [n_min, n_max];
* roll: PI on the roll-rate error;
* yaw: sideslip driven to zero (β → 0) + yaw damper (high-pass filtered r, so as
  not to fight the normal yaw rate of a turn).

Tuning (by simulation over 0–9 km, 150–300 m/s): 1 -> 4 g step in 0.4–0.9 s,
overshoot ≤ 5 % (17 % at 9 km); 90°/s roll step in 0.2–0.3 s, |β| < 1°.
These laws **also stabilize the aircraft at the unstable CG position** (x_cg = 0.35 or 0.40).

3-DOF: n_z -> commanded α by inverting the drag polar + integral correction; the roll
rate and throttle are passed through directly (the point-mass model already has its own
response laws and limiters).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from jetsim.aircraft import dynamics_3dof as d3
from jetsim.aircraft import dynamics_6dof as d6
from jetsim.aircraft.instruments import Instruments
from jetsim.control.pid import PID, Washout

Vec = npt.NDArray[np.float64]
DEG = math.pi / 180


@dataclass(frozen=True)
class HighLevelCommand:
    nz: float = 1.0  # [g]
    roll_rate: float = 0.0  # [rad/s]
    throttle: float = 0.5  # [0-1]


@dataclass(frozen=True)
class FbwGains:
    """6-DOF flight control gains (at the reference point q̄_ref)."""

    q_ref: float = 18_000.0  # [Pa] ≈ 200 m/s at 3000 m
    schedule_min: float = 0.3
    schedule_max: float = 3.0
    k_nz: float = 3.0 * DEG  # [rad/g]
    ki_nz: float = 6.0 * DEG  # [rad/(g·s)]
    k_q: float = 0.4  # [rad/(rad/s)]
    k_p: float = 0.35  # [rad/(rad/s)]
    ki_p: float = 1.0  # [rad/rad]
    k_beta: float = 3.0  # [rad/rad]
    k_r: float = 0.5  # [rad/(rad/s)]
    yaw_washout: float = 1.0  # [s]
    alpha_max: float = 25.0 * DEG
    alpha_min: float = -8.0 * DEG
    k_alpha: float = 0.5  # [g/°] angle-of-attack limiter
    k_alpha_q: float = 0.05  # [g/(°/s)] limiter damping
    nz_max: float = 9.0
    nz_min: float = -3.0
    k_nz_q: float = 0.05  # [g/(°/s)] n_z limiter anticipation (overshoot)
    nz_slew_rate: float = 20.0  # [g/s] max rate of change of the n_z setpoint
    damping_exponent: float = 0.5  # q damping scheduled as (q̄_ref/q̄)^0.5
    roll_rate_max: float = 240.0 * DEG


class FlyByWire:
    """Inner loop of the 6-DOF model (see the module docstring)."""

    def __init__(self, model: d6.F16SixDof, gains: FbwGains | None = None) -> None:
        self.model = model
        self.g = gains or FbwGains()
        cs = model.params.control_surfaces
        self._de = (cs["elevator"].min, cs["elevator"].max)
        self._da = (cs["aileron"].min, cs["aileron"].max)
        self._dr = (cs["rudder"].min, cs["rudder"].max)
        self.pitch = PID(self.g.k_nz, self.g.ki_nz, out_min=-self._de[1], out_max=-self._de[0])
        self.roll = PID(self.g.k_p, self.g.ki_p, out_min=-self._da[1], out_max=-self._da[0])
        self.yaw_filter = Washout(self.g.yaw_washout)
        self.nz_command = 1.0  # setpoint actually tracked (after limiters)
        self._nz_ref = 1.0  # setpoint after rate limiting

    def reset(self, x: Vec) -> None:
        """Engage the loop bumplessly: the integrators pick up the current surface positions."""
        self.pitch.reset(output=-x[d6.DE])
        self.roll.reset(output=-x[d6.DA])
        self.yaw_filter.reset(x[d6.R])
        self.nz_command = 1.0
        self._nz_ref = 1.0

    def limited_nz(self, ins: Instruments, nz_cmd: float) -> float:
        """n_z setpoint after the load-factor and angle-of-attack limiters."""
        g = self.g
        q_deg = ins.q / DEG
        # Load-factor limiter with pitch-rate anticipation: without
        # it, a 1 -> −3 g step at high speed overshoots −4 g (structural limit)
        # (the anticipation only tightens the limits, never widens them)
        nz = min(max(nz_cmd, g.nz_min - g.k_nz_q * min(q_deg, 0.0)),
                 g.nz_max - g.k_nz_q * max(q_deg, 0.0))  # fmt: skip
        upper = ins.nz + g.k_alpha * (g.alpha_max - ins.alpha) / DEG - g.k_alpha_q * q_deg
        lower = ins.nz + g.k_alpha * (g.alpha_min - ins.alpha) / DEG - g.k_alpha_q * q_deg
        return min(max(nz, lower), upper)

    def __call__(self, ins: Instruments, cmd: HighLevelCommand, dt: float) -> Vec:
        g = self.g
        sched = min(max(g.q_ref / max(ins.dynamic_pressure, 1.0), g.schedule_min),
                    g.schedule_max)  # fmt: skip
        # rate-limited setpoint: an abrupt reversal (+9 -> −3 g) must
        # not wind up the loop (overshoot measured down to −5 g without this filter)
        step = g.nz_slew_rate * dt
        self._nz_ref = min(max(cmd.nz, self._nz_ref - step), self._nz_ref + step)
        self.nz_command = self.limited_nz(ins, self._nz_ref)
        damping = g.k_q * sched**g.damping_exponent
        nose_up = self.pitch(self.nz_command - ins.nz, dt, gain=sched) - damping * ins.q
        p_cmd = min(max(cmd.roll_rate, -g.roll_rate_max), g.roll_rate_max)
        roll_right = self.roll(p_cmd - ins.p, dt, gain=sched)
        rudder = sched * (g.k_r * self.yaw_filter(ins.r, dt) - g.k_beta * ins.beta)
        u = np.empty(4)
        u[d6.THROTTLE] = min(max(cmd.throttle, 0.0), 1.0)
        u[d6.ELEVATOR] = min(max(-nose_up, self._de[0]), self._de[1])
        u[d6.AILERON] = min(max(-roll_right, self._da[0]), self._da[1])
        u[d6.RUDDER] = min(max(rudder, self._dr[0]), self._dr[1])
        return u


class PointMassInnerLoop:
    """Inner loop of the 3-DOF model: n_z -> commanded α (polar inversion + PI)."""

    def __init__(self, model: d3.PointMassAircraft, ki: float = 1.0 * DEG) -> None:
        self.model = model
        lim = model.params.limits
        self.correction = PID(0.0, ki, out_min=-5 * DEG, out_max=5 * DEG)
        self._alpha_bounds = (lim.alpha_min, lim.alpha_max)
        self.nz_command = 1.0

    def reset(self, x: Vec) -> None:
        self.correction.reset(0.0)
        self.nz_command = 1.0

    def __call__(self, ins: Instruments, cmd: HighLevelCommand, dt: float) -> Vec:
        m = self.model
        qs = max(ins.dynamic_pressure, 1.0) * m.S
        lim = m.params.limits
        self.nz_command = min(max(cmd.nz, lim.n_min), lim.n_max)
        alpha_ff = m.aero.alpha_for_cl(self.nz_command * m.weight / qs, ins.mach)
        alpha = alpha_ff + self.correction(self.nz_command - ins.nz, dt)
        u = np.empty(3)
        u[d3.THROTTLE] = min(max(cmd.throttle, 0.0), 1.0)
        u[d3.ALPHA_CMD] = min(max(alpha, self._alpha_bounds[0]), self._alpha_bounds[1])
        u[d3.ROLL_RATE_CMD] = cmd.roll_rate
        return u


InnerLoop = FlyByWire | PointMassInnerLoop


def make_inner_loop(model: d3.PointMassAircraft | d6.F16SixDof) -> InnerLoop:
    if isinstance(model, d6.F16SixDof):
        return FlyByWire(model)
    if isinstance(model, d3.PointMassAircraft):
        return PointMassInnerLoop(model)
    raise TypeError(f"Unsupported model: {type(model).__name__}")
