"""Geometric primitives of classical evasive maneuvers (used in phase 10).

These are the "scripted" baselines the RL agent will be compared against. Facing a threat
whose bearing is known (direction of the threat as seen from the aircraft, from North):

* beam: fly perpendicular to the line of sight. The closing velocity
  becomes small, which hampers Doppler seekers and forces the
  missile to turn;
* drag: turn your back on the threat, to lengthen the chase and drain
  the missile's energy;
* break turn: a turn at maximum load factor, at the
  last moment, to create an aiming error that the missile, less maneuverable at
  short range, can no longer correct.

The first two provide a heading to the autopilot; the break turn directly
produces a high-level command (maximum n_z, bank toward the threat).
"""

from __future__ import annotations

import math

from jetsim.aircraft.instruments import Instruments
from jetsim.control.fbw import HighLevelCommand
from jetsim.control.pid import wrap_angle


def bearing_to(own_north: float, own_east: float, north: float, east: float) -> float:
    """True bearing [rad] of a point seen from the aircraft (0 = North, π/2 = East)."""
    return math.atan2(east - own_east, north - own_north)


def relative_bearing(threat_bearing: float, heading: float) -> float:
    """Relative bearing ∈ [−π, π): > 0 if the threat is on the right."""
    return wrap_angle(threat_bearing - heading)


def beam_heading(threat_bearing: float, heading: float) -> float:
    """Beam heading: perpendicular to the threat, on the side closest to the current
    heading (shortest turn)."""
    options = (threat_bearing + math.pi / 2, threat_bearing - math.pi / 2)
    best = min(options, key=lambda h: abs(wrap_angle(h - heading)))
    return wrap_angle(best)


def drag_heading(threat_bearing: float) -> float:
    """Drag heading: back to the threat."""
    return wrap_angle(threat_bearing + math.pi)


def break_turn(
    ins: Instruments,
    threat_bearing: float,
    *,
    nz: float = 9.0,
    bank: float = math.radians(80),
    roll_gain: float = 3.0,
    max_roll_rate: float = math.radians(240),
    throttle: float = 1.0,
) -> HighLevelCommand:
    """Break turn **toward** the threat: bank ``bank`` on the threat side, load factor
    ``nz`` (the inner loop limits it to what α_max and n_max allow)."""
    side = 1.0 if relative_bearing(threat_bearing, ins.heading) >= 0 else -1.0
    bank_error = wrap_angle(side * bank - ins.bank)
    p_cmd = min(max(roll_gain * bank_error, -max_roll_rate), max_roll_rate)
    return HighLevelCommand(nz=nz, roll_rate=p_cmd, throttle=throttle)
