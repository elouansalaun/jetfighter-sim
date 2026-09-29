"""Autopilot: altitude hold, heading hold, airspeed hold, turn at a given bank angle.

Outer loops producing a ``HighLevelCommand`` (n_z, roll rate, throttle),
then executed by the inner loop (``fbw.py``). Same tuning for 3-DOF and 6-DOF ::

    altitude --(P)--> vertical speed --> flight-path angle γ --(PI)--> n_z  (+ 1/cos φ in a turn)
    heading  --(P)--> bank μ --(PI)--> roll rate
    airspeed --(PI)--> throttle

Setpoints (``AutopilotTargets``): each hold can be disabled (``None``);
``bank`` (imposed bank angle) takes priority over ``heading``.

Usage ::

    ap = Autopilot(model)
    ap.reset(x, instruments)
    ap.targets = AutopilotTargets(altitude=5000, heading=math.radians(90), airspeed=250)
    u = ap.controls(instruments, dt)      # model control
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from jetsim.aircraft import dynamics_3dof as d3
from jetsim.aircraft import dynamics_6dof as d6
from jetsim.aircraft.instruments import Instruments
from jetsim.control.fbw import HighLevelCommand, make_inner_loop
from jetsim.control.pid import PID, wrap_angle
from jetsim.core.constants import G0

Vec = npt.NDArray[np.float64]
DEG = math.pi / 180


@dataclass
class AutopilotTargets:
    altitude: float | None = None  # [m]
    heading: float | None = None  # course χ [rad]
    airspeed: float | None = None  # true airspeed [m/s]
    bank: float | None = None  # imposed bank angle [rad] (takes priority over heading)
    throttle: float | None = None  # fixed throttle if airspeed is None


@dataclass(frozen=True)
class AutopilotGains:
    k_altitude: float = 0.12  # [1/s] altitude error -> vertical speed
    max_vertical_speed: float = 40.0  # [m/s]
    k_gamma: float = 0.6  # [1/s] flight-path angle error -> flight-path angle rate γ̇
    ki_gamma: float = 0.1  # [1/s²]
    max_gamma_rate: float = 0.1  # [rad/s] ≈ ±2 g of correction at 200 m/s
    nz_min: float = -0.5  # comfort: the autopilot does not push beyond this
    nz_max: float = 7.5
    k_heading: float = 0.4  # [1/s] heading error -> turn rate
    max_bank: float = 60.0 * DEG  # in heading hold
    max_bank_explicit: float = 85.0 * DEG  # imposed bank angle (tight turns)
    k_bank: float = 2.0  # [1/s] bank error -> roll rate
    ki_bank: float = 0.2  # [1/s²] removes the steady-state bank error in tight turns
    max_roll_rate: float = 90.0 * DEG
    k_speed: float = 0.03  # [1/(m/s)]
    ki_speed: float = 0.01  # [1/(m·s)]


class Autopilot:
    """Full autopilot: outer loops + inner loop (see the module docstring)."""

    def __init__(
        self,
        model: d3.PointMassAircraft | d6.F16SixDof,
        gains: AutopilotGains | None = None,
    ) -> None:
        self.model = model
        self.g = gains or AutopilotGains()
        self.inner = make_inner_loop(model)
        self.targets = AutopilotTargets()
        self.gamma_pid = PID(self.g.k_gamma, self.g.ki_gamma, out_min=-self.g.max_gamma_rate,
                             out_max=self.g.max_gamma_rate)  # fmt: skip
        self.speed_pid = PID(self.g.k_speed, self.g.ki_speed, out_min=0.0, out_max=1.0)
        self.bank_pid = PID(self.g.k_bank, self.g.ki_bank, out_min=-self.g.max_roll_rate,
                            out_max=self.g.max_roll_rate, integral_zone=5 * DEG)  # fmt: skip
        self.command = HighLevelCommand()

    def reset(self, x: Vec, ins: Instruments) -> None:
        """Engage the autopilot on the current state, bumplessly."""
        self.inner.reset(x)
        self.gamma_pid.reset(0.0)
        self.bank_pid.reset(0.0)
        if isinstance(self.model, d6.F16SixDof):
            thr = self.model.engine.throttle_for_power(float(x[d6.POWER]))
        else:
            thr = float(x[d3.POWER])
        self.speed_pid.reset(output=min(max(thr, 0.0), 1.0))
        self.command = HighLevelCommand(nz=ins.nz, roll_rate=0.0, throttle=0.5)

    # ------------------------------------------------------------------
    def high_level(self, ins: Instruments, dt: float) -> HighLevelCommand:
        """Outer loops: setpoints -> (n_z, roll rate, throttle)."""
        g, tg = self.g, self.targets
        v = max(ins.tas, 1.0)
        cos_bank = max(math.cos(ins.bank), 0.12)  # turn compensation up to ≈ 83°

        # Pitch: altitude -> vertical speed -> flight-path angle -> n_z
        if tg.altitude is not None:
            vs = min(max(g.k_altitude * (tg.altitude - ins.altitude), -g.max_vertical_speed),
                     g.max_vertical_speed)  # fmt: skip
            gamma_cmd = math.asin(min(max(vs / v, -0.8), 0.8))
            gamma_rate = self.gamma_pid(gamma_cmd - ins.gamma, dt)
            correction = gamma_rate * v / G0  # extra n_z to curve the flight path
            nz = math.cos(ins.gamma) / cos_bank + correction
        else:
            nz = 1.0 / cos_bank
        nz = min(max(nz, g.nz_min), g.nz_max)

        # Roll: heading -> bank -> roll rate
        if tg.bank is not None:
            bank_cmd = min(max(tg.bank, -g.max_bank_explicit), g.max_bank_explicit)
        elif tg.heading is not None:
            turn_rate = g.k_heading * wrap_angle(tg.heading - ins.course)
            bank_cmd = min(max(math.atan(turn_rate * v / G0), -g.max_bank), g.max_bank)
        else:
            bank_cmd = 0.0
        roll_rate = self.bank_pid(wrap_angle(bank_cmd - ins.bank), dt)

        # Engine
        if tg.airspeed is not None:
            throttle = self.speed_pid(tg.airspeed - ins.tas, dt)
        elif tg.throttle is not None:
            throttle = tg.throttle
        else:
            throttle = self.command.throttle
        self.command = HighLevelCommand(nz=nz, roll_rate=roll_rate, throttle=throttle)
        return self.command

    def controls(self, ins: Instruments, dt: float) -> Vec:
        """Model control (setpoints -> outer loops -> inner loop)."""
        return self.inner(ins, self.high_level(ins, dt), dt)
