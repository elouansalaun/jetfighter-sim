"""Export to the Tacview ACMI 2.2 format (text), to review flights in 3D.

Tacview (free for playback) shows the aircraft on a globe with its attitude, trajectory
and parameters. The format is text:

    FileType=text/acmi/tacview
    FileVersion=2.2
    0,ReferenceTime=2026-01-01T12:00:00Z
    0,ReferenceLongitude=-6
    0,ReferenceLatitude=46
    #0.00
    101,T=lon|lat|alt|roll|pitch|yaw,Name=F-16C,Type=Air+FixedWing,...
    #0.10
    101,T=...

* the longitude and latitude in ``T=`` are offsets from the reference values;
* altitude in meters, angles in degrees (roll positive right, pitch positive nose up,
  yaw clockwise from North: the project's conventions);
* one hexadecimal identifier per object: the missile can be added in phase 9.

The project's world is flat (local NED frame): it is "placed" around a
reference point, by default over the Atlantic so as not to hit terrain in Tacview.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from jetsim.viz.recorder import FlightRecording

EARTH_RADIUS = 6_371_000.0
RAD_TO_DEG = 180.0 / math.pi


def _escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace(",", "\\,").replace("\n", " ")


def _num(value: float, digits: int = 3) -> str:
    """Compact number: useless zeros removed, never "-0"."""
    if not math.isfinite(value):
        return "0"
    text = f"{value:.{digits}f}".rstrip("0").rstrip(".")
    return "0" if text in ("-0", "") else text


def _heading_deg(yaw: float) -> float:
    """Heading in degrees in [0, 360) (avoids "360" after rounding)."""
    deg = math.degrees(yaw) % 360.0
    return 0.0 if deg >= 359.995 else deg


@dataclass
class AcmiWriter:
    """Build an ACMI file object by object and frame by frame."""

    title: str = "jetfighter-sim"
    reference_latitude: float = 46.0
    reference_longitude: float = -6.0
    reference_time: datetime = field(
        default_factory=lambda: datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    )
    author: str = "jetfighter-sim"
    _frames: dict[float, list[str]] = field(default_factory=dict, init=False)
    _declared: set[int] = field(default_factory=set, init=False)
    _objects: dict[int, str] = field(default_factory=dict, init=False)

    def add_object(
        self,
        object_id: int,
        *,
        name: str,
        type_tags: str,
        color: str = "Blue",
        coalition: str = "Allies",
        pilot: str | None = None,
    ) -> None:
        """Declare an object; its fixed properties are written on its first appearance."""
        props = [f"Name={_escape(name)}", f"Type={type_tags}", f"Color={color}",
                 f"Coalition={coalition}"]  # fmt: skip
        if pilot:
            props.append(f"Pilot={_escape(pilot)}")
        self._objects[object_id] = ",".join(props)

    def _line(self, t: float, line: str) -> None:
        self._frames.setdefault(round(t, 3), []).append(line)

    def position(
        self,
        t: float,
        object_id: int,
        *,
        north: float,
        east: float,
        altitude: float,
        roll: float,
        pitch: float,
        yaw: float,
        **properties: float,
    ) -> None:
        """Position (m, local NED frame) and attitude (rad) of an object at time t."""
        lat0 = math.radians(self.reference_latitude)
        dlat = north / EARTH_RADIUS * RAD_TO_DEG
        dlon = east / (EARTH_RADIUS * math.cos(lat0)) * RAD_TO_DEG
        transform = "|".join(
            [_num(dlon, 7), _num(dlat, 7), _num(altitude, 1), _num(roll * RAD_TO_DEG, 2),
             _num(pitch * RAD_TO_DEG, 2), _num(_heading_deg(yaw), 2)]
        )  # fmt: skip
        parts = [f"{object_id:x}", f"T={transform}"]
        if object_id not in self._declared and object_id in self._objects:
            parts.append(self._objects[object_id])
            self._declared.add(object_id)
        parts += [f"{k}={_num(v)}" for k, v in properties.items()]
        self._line(t, ",".join(parts))

    def remove(self, t: float, object_id: int) -> None:
        self._line(t, f"-{object_id:x}")

    def event(self, t: float, kind: str, object_ids: tuple[int, ...] = (), text: str = "") -> None:
        """Tacview event: Message, Bookmark, Destroyed, LeftArea, Timeout…"""
        ids = "|".join(f"{i:x}" for i in object_ids)
        body = "|".join(p for p in (kind, ids, _escape(text)) if p)
        self._line(t, f"0,Event={body}")

    def text(self) -> str:
        header = [
            "FileType=text/acmi/tacview",
            "FileVersion=2.2",
            f"0,ReferenceTime={self.reference_time.strftime('%Y-%m-%dT%H:%M:%SZ')}",
            f"0,ReferenceLongitude={_num(self.reference_longitude, 6)}",
            f"0,ReferenceLatitude={_num(self.reference_latitude, 6)}",
            f"0,Title={_escape(self.title)}",
            f"0,Author={_escape(self.author)}",
            "0,DataSource=jetfighter-sim (simulation)",
        ]
        body: list[str] = []
        for t in sorted(self._frames):
            body.append(f"#{_num(t, 3) if t else '0'}")
            body.extend(self._frames[t])
        return "\n".join(header + body) + "\n"

    def write(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.text(), encoding="utf-8")
        return path


AIRCRAFT_ID = 0x101


def export_recording(
    rec: FlightRecording,
    path: str | Path,
    *,
    name: str = "F-16C",
    pilot: str = "jetfighter-sim",
    title: str | None = None,
    writer: AcmiWriter | None = None,
    object_id: int = 0x101,
    color: str = "Blue",
    write: bool = True,
) -> Path:
    """Export a recorded flight to a ``.acmi`` file readable by Tacview.

    To put several flights in the same file (agent and autopilot side by
    side, for example), pass the same ``writer`` with different ``object_id`` values and
    ``write=False`` except for the last one.
    """
    w = writer or AcmiWriter(title=title or f"Flight {rec.model_type.upper()}")
    w.add_object(object_id, name=name, type_tags="Air+FixedWing", pilot=pilot, color=color)
    ins = rec.instruments
    for k, t in enumerate(rec.t):
        w.position(
            float(t),
            object_id,
            north=float(ins["north"][k]),
            east=float(ins["east"][k]),
            altitude=float(ins["altitude"][k]),
            roll=float(ins["roll"][k]),
            pitch=float(ins["pitch"][k]),
            yaw=float(ins["heading"][k]),
            TAS=float(ins["tas"][k]),
            CAS=float(ins["cas"][k]),
            Mach=float(ins["mach"][k]),
            AOA=float(ins["alpha"][k]) * RAD_TO_DEG,
            AOS=float(ins["beta"][k]) * RAD_TO_DEG,
            AGL=float(ins["altitude"][k]),
            Throttle=float(ins["power"][k]),
        )
    for t, message in rec.events:
        kind = "Destroyed" if "ground" in message else "Message"
        w.event(t, kind, (object_id,), message)
    return w.write(path) if write else Path(path)


def add_waypoints(
    writer: AcmiWriter, waypoints: list[tuple[float, float, float]], first_id: int = 0x201
) -> None:
    """Add waypoints (north, east, altitude) as static objects."""
    for k, (n, e, h) in enumerate(waypoints):
        oid = first_id + k
        writer.add_object(oid, name=f"WP{k + 1}", type_tags="Navaid+Static+Waypoint",
                          color="Green", coalition="Neutrals")  # fmt: skip
        writer.position(0.0, oid, north=n, east=e, altitude=h, roll=0.0, pitch=0.0, yaw=0.0)
