"""Manual flying: simulation core (no graphics dependency) and pygame HUD.

The core (``ManualFlight``) turns logical inputs ("pitch up", "roll
right"…) or joystick axes into controls for the 3-DOF or 6-DOF model, advances
the simulation in real time, monitors the envelope and records the flight. It is testable
without a screen. The renderer (``HudRenderer``) draws an artificial horizon and the instruments
with pygame.

Control laws (stick ∈ [−1, 1], positive = pitch up / roll right / yaw right):

* 6-DOF (direct control-surface command): δe = δe_trim + trim − 12°·pitch_stick,
  δa = −15°·roll_stick, δr = −15°·rudder_pedals;
* 3-DOF: α_cmd = α_trim + trim + 15°·pitch_stick, roll rate = 180°/s·stick.

With the keyboard, the stick moves progressively while the key is held and returns
to neutral when it is released (like a real spring-loaded stick).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from jetsim.aircraft import dynamics_3dof as d3
from jetsim.aircraft import dynamics_6dof as d6
from jetsim.aircraft.envelope import EnvelopeMonitor
from jetsim.aircraft.instruments import Instruments, read_instruments
from jetsim.aircraft.params import load_aircraft, load_envelope
from jetsim.viz.recorder import FlightRecorder, FlightRecording

DEG = math.pi / 180

INPUTS = (
    "pitch_up", "pitch_down", "roll_left", "roll_right", "yaw_left", "yaw_right",
    "throttle_up", "throttle_down", "trim_up", "trim_down",
)  # fmt: skip


@dataclass
class Stick:
    """Pilot control positions."""

    pitch: float = 0.0  # > 0: pitch up
    roll: float = 0.0  # > 0: roll right
    yaw: float = 0.0  # > 0: yaw right
    throttle: float = 0.5
    trim: float = 0.0  # [rad] > 0: pitch up

    MOVE_RATE = 2.5  # stick travel per second, key held
    RETURN_RATE = 4.0  # return to neutral per second, key released
    THROTTLE_RATE = 0.4
    TRIM_RATE = 1.0 * DEG

    @staticmethod
    def _axis(value: float, plus: bool, minus: bool, dt: float) -> float:
        target = float(plus) - float(minus)
        rate = Stick.MOVE_RATE if target else Stick.RETURN_RATE
        step = rate * dt
        if abs(target - value) <= step:
            return target
        return value + math.copysign(step, target - value)

    def update(
        self,
        pressed: set[str],
        dt: float,
        axes: tuple[float, float, float, float | None] | None = None,
    ) -> None:
        """Update the stick (keyboard) or copy it from a joystick.

        Args:
            pressed: active logical inputs (see ``INPUTS``).
            dt: frame duration [s].
            axes: (roll, pitch, yaw, throttle or None) from a joystick, in [−1, 1] and
                [0, 1]; if provided, they replace the keyboard for those axes.
        """
        if axes is not None:
            self.roll, self.pitch, self.yaw = (float(np.clip(a, -1, 1)) for a in axes[:3])
            if axes[3] is not None:
                self.throttle = float(np.clip(axes[3], 0.0, 1.0))
        else:
            self.pitch = self._axis(self.pitch, "pitch_up" in pressed, "pitch_down" in pressed, dt)
            self.roll = self._axis(self.roll, "roll_right" in pressed, "roll_left" in pressed, dt)
            self.yaw = self._axis(self.yaw, "yaw_right" in pressed, "yaw_left" in pressed, dt)
        dthr = ("throttle_up" in pressed) - ("throttle_down" in pressed)
        self.throttle = float(np.clip(self.throttle + dthr * self.THROTTLE_RATE * dt, 0.0, 1.0))
        dtrim = ("trim_up" in pressed) - ("trim_down" in pressed)
        self.trim = float(np.clip(self.trim + dtrim * self.TRIM_RATE * dt, -10 * DEG, 10 * DEG))


class ManualFlight:
    """Real-time simulation flown by a human (see the module docstring)."""

    def __init__(
        self,
        model_kind: str = "6dof",
        *,
        altitude: float = 3000.0,
        airspeed: float = 200.0,
        physics_dt: float = 0.01,
        frame_dt: float = 0.02,
    ) -> None:
        if model_kind not in ("3dof", "6dof"):
            raise ValueError("model_kind must be '3dof' or '6dof'.")
        self.kind = model_kind
        self.altitude0, self.airspeed0 = altitude, airspeed
        self.physics_dt = physics_dt
        self.substeps = max(1, round(frame_dt / physics_dt))
        self.frame_dt = self.substeps * physics_dt
        params = load_aircraft("f16")
        self.model: d3.PointMassAircraft | d6.F16SixDof = (
            d6.F16SixDof(params) if model_kind == "6dof" else d3.PointMassAircraft(params)
        )
        self.monitor = EnvelopeMonitor(load_envelope("f16", six_dof=model_kind == "6dof"))
        self.reset()

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Restart in trimmed flight; the current recording is discarded."""
        if isinstance(self.model, d6.F16SixDof):
            self.x, self.u_trim = self.model.trim(self.altitude0, self.airspeed0)
        else:
            self.x, self.u_trim = self.model.trimmed_state(self.altitude0, self.airspeed0)
        self.stick = Stick(throttle=float(self.u_trim[0]))
        self.t = 0.0
        self.paused = False
        self.ended = False
        self.message = ""
        self.monitor.reset()
        self.recorder = FlightRecorder(
            self.model, physics_dt=self.physics_dt, substeps=self.substeps,
            metadata={"source": "manual flight"},
        )  # fmt: skip
        self.instruments: Instruments = read_instruments(self.model, self.x)

    def controls(self) -> np.ndarray:
        """Model control from the stick position."""
        s, u = self.stick, self.u_trim.copy()
        u[0] = s.throttle
        if self.kind == "6dof":
            u[d6.ELEVATOR] = self.u_trim[d6.ELEVATOR] - s.trim - 12 * DEG * s.pitch
            u[d6.AILERON] = -15 * DEG * s.roll
            u[d6.RUDDER] = -15 * DEG * s.yaw
        else:
            u[d3.ALPHA_CMD] = self.u_trim[d3.ALPHA_CMD] + s.trim + 15 * DEG * s.pitch
            u[d3.ROLL_RATE_CMD] = 180 * DEG * s.roll
        return u

    def update(
        self,
        pressed: set[str],
        axes: tuple[float, float, float, float | None] | None = None,
    ) -> None:
        """Advance one frame: stick, simulation, recording, envelope."""
        if self.paused or self.ended:
            return
        self.stick.update(pressed, self.frame_dt, axes)
        u = self.controls()
        self.recorder.record(self.t, self.x, u)
        for _ in range(self.substeps):
            self.x = self.model.step(self.x, u, self.physics_dt)
        self.t = round(self.t + self.frame_dt, 9)
        self.instruments = read_instruments(self.model, self.x)
        violation = self.monitor.check(self.instruments, self.frame_dt, state=self.x)
        if violation is not None:
            self.ended = True
            self.message = self.monitor.message
            self.recorder.record(self.t, self.x, u)
            self.recorder.event(self.t, self.message)

    def recording(self) -> FlightRecording | None:
        return self.recorder.finish() if len(self.recorder) >= 2 else None

    def save(self, directory: str | Path = "outputs/flights") -> dict[str, Path]:
        """Save the flight (.npz, Tacview .acmi, .png of the curves). Returns the paths."""
        from jetsim.viz.plots import plot_time_series
        from jetsim.viz.tacview import export_recording

        rec = self.recording()
        if rec is None:
            return {}
        stem = Path(directory) / f"manual_{self.kind}_{datetime.now():%Y%m%d_%H%M%S}"
        paths = {
            "npz": rec.save(stem.with_suffix(".npz")),
            "acmi": export_recording(rec, stem.with_suffix(".acmi"), pilot="Human pilot"),
        }
        fig = plot_time_series(rec, title=f"Manual flight {self.kind.upper()}")
        paths["png"] = stem.with_suffix(".png")
        fig.savefig(paths["png"], dpi=120)
        import matplotlib.pyplot as plt

        plt.close(fig)
        return paths


# ==========================================================================
# pygame rendering
# ==========================================================================
SKY = (70, 130, 200)
GROUND = (140, 95, 55)
WHITE = (245, 245, 240)
YELLOW = (255, 210, 60)
RED = (230, 70, 60)
PANEL = (24, 26, 30)
DIM = (150, 150, 145)

HELP = (
    "↑/↓ pitch down/up   ←/→ roll   W/X rudder   A/Q throttle +/−   T/G trim",
    "Space pause   R restart   S save   Esc quit (saves)",
)


class HudRenderer:
    """Draw the artificial horizon and the instruments on a pygame surface."""

    def __init__(self, size: tuple[int, int] = (1100, 700)) -> None:
        import pygame

        self.pg = pygame
        self.size = size
        pygame.font.init()
        # system font with accents, arrows and Greek letters (fallback: pygame font)
        names = "helveticaneue,helvetica,arial,dejavusans,liberationsans"
        self.font = pygame.font.SysFont(names, 19)
        self.small = pygame.font.SysFont(names, 15)
        self.big = pygame.font.SysFont(names, 30, bold=True)

    # --- utilities ------------------------------------------------------
    def _text(self, surf: Any, text: str, pos: tuple[int, int], font: Any = None,
              color: tuple[int, int, int] = WHITE, center: bool = False) -> None:  # fmt: skip
        img = (font or self.font).render(text, True, color)
        rect = img.get_rect(center=pos) if center else img.get_rect(topleft=pos)
        surf.blit(img, rect)

    # --- artificial horizon --------------------------------------------
    def _horizon(self, surf: Any, rect: Any, ins: Instruments) -> None:
        pg = self.pg
        view = surf.subsurface(rect)
        w, h = rect.size
        cx, cy = w / 2, h / 2
        ppd = h / 60.0  # pixels per degree of pitch (±30° visible)
        roll, pitch = ins.roll, math.degrees(ins.pitch)
        # screen vectors: along the horizon and up toward the sky
        along = (math.cos(-roll), math.sin(-roll))
        up = (math.sin(-roll), -math.cos(-roll))

        def point(a: float, u: float) -> tuple[float, float]:
            # a along the horizon, u upward (in pixels), from the horizon line
            return (cx + along[0] * a + up[0] * (u - pitch * ppd),
                    cy + along[1] * a + up[1] * (u - pitch * ppd))  # fmt: skip

        view.fill(SKY)
        far = 3 * max(w, h)
        pg.draw.polygon(view, GROUND, [point(-far, 0), point(far, 0), point(far, -far),
                                       point(-far, -far)])  # fmt: skip
        pg.draw.line(view, WHITE, point(-far, 0), point(far, 0), 2)
        for deg in range(-90, 91, 10):
            if deg == 0:
                continue
            half = 60 if deg % 20 == 0 else 35
            u = deg * ppd
            p1, p2 = point(-half, u), point(half, u)
            pg.draw.line(view, WHITE, p1, p2, 2 if deg > 0 else 1)
            self._text(view, f"{deg}", (int(point(half + 18, u)[0]), int(point(half + 18, u)[1])),
                       self.small, WHITE, center=True)  # fmt: skip
        # aircraft symbol (fixed)
        pg.draw.line(view, YELLOW, (cx - 90, cy), (cx - 30, cy), 5)
        pg.draw.line(view, YELLOW, (cx + 30, cy), (cx + 90, cy), 5)
        pg.draw.lines(view, YELLOW, False, [(cx - 30, cy), (cx - 15, cy + 15), (cx, cy),
                                            (cx + 15, cy + 15), (cx + 30, cy)], 4)  # fmt: skip
        pg.draw.circle(view, YELLOW, (int(cx), int(cy)), 3)
        # bank scale
        for deg in (-60, -45, -30, -20, -10, 0, 10, 20, 30, 45, 60):
            ang = math.radians(deg) - math.pi / 2
            r1, r2 = h * 0.42, h * 0.42 + (14 if deg % 30 == 0 else 8)
            pg.draw.line(view, WHITE, (cx + r1 * math.cos(ang), cy + r1 * math.sin(ang)),
                         (cx + r2 * math.cos(ang), cy + r2 * math.sin(ang)), 2)  # fmt: skip
        ang = -roll - math.pi / 2
        tip = (cx + h * 0.41 * math.cos(ang), cy + h * 0.41 * math.sin(ang))
        pg.draw.circle(view, YELLOW, (int(tip[0]), int(tip[1])), 6)
        pg.draw.rect(view, DIM, view.get_rect(), 1)

    # --- instrument panel ----------------------------------------------
    def _panel(self, surf: Any, rect: Any, flight: ManualFlight) -> None:
        pg = self.pg
        ins, s = flight.instruments, flight.stick
        pg.draw.rect(surf, PANEL, rect)
        x, y = rect.x + 20, rect.y + 16
        self._text(
            surf, f"F-16 {flight.kind.upper()}   t = {flight.t:6.1f} s", (x, y), self.font, DIM
        )
        y += 38
        rows = [
            ("True airspeed", f"{ins.tas:6.0f} m/s  {ins.tas * 1.94384:4.0f} kt"),
            ("Calibrated airspeed", f"{ins.cas:6.0f} m/s"),
            ("Mach", f"{ins.mach:6.2f}"),
            ("Altitude", f"{ins.altitude:6.0f} m"),
            ("Vertical speed", f"{ins.vertical_speed:+6.0f} m/s"),
            ("Heading", f"{math.degrees(ins.heading) % 360:6.0f}°"),
            ("Angle of attack α", f"{math.degrees(ins.alpha):6.1f}°"),
            ("Sideslip β", f"{math.degrees(ins.beta):6.1f}°"),
            ("Load factor", f"{ins.nz:6.1f} g"),
            ("Ps", f"{ins.specific_excess_power:+6.0f} m/s"),
        ]
        for label, value in rows:
            self._text(surf, label, (x, y), self.font, DIM)
            color = RED if (label == "Load factor" and not -3 <= ins.nz <= 9) else WHITE
            self._text(surf, value, (x + 170, y), self.font, color)
            y += 29
        # throttle and power
        y += 8
        self._text(surf, "Throttle / power", (x, y), self.font, DIM)
        y += 26
        bar = pg.Rect(x, y, rect.width - 40, 14)
        pg.draw.rect(surf, DIM, bar, 1)
        pg.draw.rect(surf, WHITE, (bar.x, bar.y, int(bar.width * s.throttle), 6))
        power_color = RED if ins.power > 0.5 and flight.kind == "6dof" else YELLOW
        pg.draw.rect(surf, power_color, (bar.x, bar.y + 8, int(bar.width * ins.power), 6))
        y += 30
        # stick position
        self._text(surf, f"Stick (trim {math.degrees(s.trim):+.1f}°)", (x, y), self.font, DIM)
        box = pg.Rect(x, y + 26, 110, 110)
        pg.draw.rect(surf, DIM, box, 1)
        pg.draw.line(surf, DIM, box.midtop, box.midbottom)
        pg.draw.line(surf, DIM, box.midleft, box.midright)
        pos = (box.centerx + int(s.roll * 52), box.centery + int(s.pitch * 52))
        pg.draw.circle(surf, YELLOW, pos, 7)
        yaw_bar = pg.Rect(x + 130, y + 76, 120, 10)
        pg.draw.rect(surf, DIM, yaw_bar, 1)
        pg.draw.circle(surf, YELLOW, (yaw_bar.centerx + int(s.yaw * 58), yaw_bar.centery), 6)
        self._text(surf, "rudder", (x + 130, y + 92), self.small, DIM)

    def draw(self, surf: Any, flight: ManualFlight, joystick_name: str | None = None) -> None:
        pg = self.pg
        w, h = surf.get_size()
        surf.fill((12, 13, 15))
        panel_w = 360
        self._horizon(surf, pg.Rect(10, 10, w - panel_w - 30, h - 80), flight.instruments)
        self._panel(surf, pg.Rect(w - panel_w - 10, 10, panel_w, h - 80), flight)
        for i, line in enumerate(HELP):
            self._text(surf, line, (14, h - 60 + 24 * i), self.small, DIM)
        if joystick_name:
            self._text(surf, f"Joystick: {joystick_name}", (w - panel_w, h - 60), self.small, DIM)
        cx, cy = (w - panel_w - 20) // 2, 60
        if flight.ended:
            self._text(surf, f"Flight ended: {flight.message}", (cx, cy), self.big, RED, True)
            self._text(surf, "R to restart", (cx, cy + 34), self.font, WHITE, True)
        elif flight.paused:
            self._text(surf, "PAUSE", (cx, cy), self.big, YELLOW, True)
