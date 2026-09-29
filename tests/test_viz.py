"""Phase 5 : enregistrement et rejeu, export Tacview, tracés, pilotage manuel."""

import math

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from jetsim.aircraft import dynamics_3dof as d3
from jetsim.aircraft import dynamics_6dof as d6
from jetsim.aircraft.params import load_aircraft
from jetsim.viz.manual import INPUTS, ManualFlight, Stick
from jetsim.viz.plots import plot_time_series, plot_trajectory_3d
from jetsim.viz.recorder import FlightRecorder, FlightRecording, build_model, replay
from jetsim.viz.tacview import EARTH_RADIUS, AcmiWriter, export_recording

DEG = math.pi / 180


def fly(model, x, u0, seconds=6.0, dt=0.01, substeps=10):
    """Petit vol : mise en virage puis ressource. Renvoie l'enregistrement."""
    rec = FlightRecorder(model, physics_dt=dt, substeps=substeps)
    t = 0.0
    for k in range(round(seconds / (dt * substeps))):
        u = u0.copy()
        if isinstance(model, d6.F16SixDof):
            u[d6.AILERON] = -5 * DEG if 5 <= k < 12 else 0.0
            u[d6.ELEVATOR] -= 2 * DEG if k >= 20 else 0.0
        else:
            u[d3.ROLL_RATE_CMD] = 60 * DEG if 5 <= k < 12 else 0.0
            u[d3.ALPHA_CMD] += 3 * DEG if k >= 20 else 0.0
        rec.record(t, x, u)
        for _ in range(substeps):
            x = model.step(x, u, dt)
        t = round(t + dt * substeps, 9)
    rec.record(t, x, u)
    rec.event(t, "fin du test")
    return rec.finish()


@pytest.fixture(scope="module")
def flight6() -> FlightRecording:
    model = d6.F16SixDof(load_aircraft("f16"))
    x, u = model.trim(3000.0, 220.0)
    return fly(model, x, u)


@pytest.fixture(scope="module")
def flight3() -> FlightRecording:
    model = d3.PointMassAircraft(load_aircraft("f16"))
    x, u = model.trimmed_state(3000.0, 220.0)
    return fly(model, x, u)


# --------------------------------------------------------------------------
# Enregistrement et rejeu
# --------------------------------------------------------------------------
@pytest.mark.parametrize("name", ["flight6", "flight3"])
def test_save_load_roundtrip(request, tmp_path, name):
    rec = request.getfixturevalue(name)
    path = rec.save(tmp_path / "vol.npz")
    back = FlightRecording.load(path)
    np.testing.assert_array_equal(back.states, rec.states)
    np.testing.assert_array_equal(back.controls, rec.controls)
    np.testing.assert_array_equal(back.instruments["nz"], rec.instruments["nz"])
    assert back.events == rec.events
    assert back.description == rec.description
    assert back.substeps == 10 and back.physics_dt == 0.01
    assert len(back) == len(rec) == 61
    assert back.duration == pytest.approx(6.0)


@pytest.mark.parametrize("name", ["flight6", "flight3"])
def test_replay_is_deterministic(request, name):
    rec = request.getfixturevalue(name)
    np.testing.assert_array_equal(replay(rec), rec.states)


def test_build_model_from_description(flight6, flight3):
    m6 = build_model(flight6.description)
    assert isinstance(m6, d6.F16SixDof) and m6.xcg == pytest.approx(0.30)
    assert isinstance(build_model(flight3.description), d3.PointMassAircraft)
    with pytest.raises(ValueError):
        build_model({"model": "4dof"})


def test_empty_recorder_raises():
    model = d3.PointMassAircraft(load_aircraft("f16"))
    with pytest.raises(ValueError):
        FlightRecorder(model, physics_dt=0.01, substeps=1).finish()


# --------------------------------------------------------------------------
# Tacview
# --------------------------------------------------------------------------
def parse_acmi(text: str):
    lines = text.splitlines()
    frames, current = {}, None
    for line in lines[2:]:
        if line.startswith("#"):
            current = float(line[1:])
            frames[current] = []
        elif current is not None:
            frames[current].append(line)
    return lines, frames


def test_acmi_structure(flight6, tmp_path):
    path = export_recording(flight6, tmp_path / "vol.acmi")
    lines, frames = parse_acmi(path.read_text(encoding="utf-8"))
    assert lines[0] == "FileType=text/acmi/tacview"
    assert lines[1] == "FileVersion=2.2"
    assert any(line.startswith("0,ReferenceTime=") for line in lines[:8])
    times = list(frames)
    assert times == sorted(times) and len(times) == len(flight6)
    first = frames[times[0]][0]
    assert first.startswith("101,T=") and "Type=Air+FixedWing" in first
    assert "Type=" not in frames[times[1]][0]  # propriétés fixes écrites une seule fois
    transform = first.split(",")[1][2:].split("|")
    assert len(transform) == 6  # lon|lat|alt|roulis|tangage|lacet
    assert float(transform[2]) == pytest.approx(3000.0, abs=0.1)
    assert any("Event=Message|101|fin du test" in line for line in frames[times[-1]])
    assert "-0|" not in path.read_text(encoding="utf-8")


def test_acmi_coordinates_and_units():
    w = AcmiWriter(reference_latitude=45.0, reference_longitude=0.0)
    w.add_object(0x2A, name="Test", type_tags="Air+FixedWing")
    w.position(0.0, 0x2A, north=1000.0, east=1000.0, altitude=500.0,
               roll=30 * DEG, pitch=-10 * DEG, yaw=-90 * DEG, TAS=200.0)  # fmt: skip
    line = next(ln for ln in w.text().splitlines() if ln.startswith("2a,"))
    lon, lat, alt, roll, pitch, yaw = (float(v) for v in line.split(",")[1][2:].split("|"))
    assert lat == pytest.approx(math.degrees(1000 / EARTH_RADIUS), rel=1e-4)
    assert lon == pytest.approx(math.degrees(1000 / (EARTH_RADIUS * math.cos(math.radians(45)))),
                                rel=1e-4)  # fmt: skip
    assert (alt, roll, pitch, yaw) == (500.0, 30.0, -10.0, 270.0)
    assert "TAS=200" in line


def test_acmi_events_escape_commas():
    w = AcmiWriter()
    w.event(1.0, "Message", (0x101,), "surcharge, 10 g")
    w.remove(2.0, 0x101)
    text = w.text()
    assert "0,Event=Message|101|surcharge\\, 10 g" in text
    assert "\n-101" in text


def test_ground_collision_is_destroyed_event(flight6, tmp_path):
    rec = FlightRecording(**{**flight6.__dict__, "events": [(6.0, "collision avec le sol : -3 m")]})
    text = export_recording(rec, tmp_path / "crash.acmi").read_text(encoding="utf-8")
    assert "Event=Destroyed|101|collision avec le sol" in text


# --------------------------------------------------------------------------
# Tracés
# --------------------------------------------------------------------------
@pytest.mark.parametrize("name", ["flight6", "flight3"])
def test_plots_render(request, tmp_path, name):
    rec = request.getfixturevalue(name)
    fig = plot_time_series(rec)
    assert len(fig.axes) == 8
    fig.savefig(tmp_path / "series.png")
    fig3d = plot_trajectory_3d(rec)
    fig3d.savefig(tmp_path / "traj.png")
    assert (tmp_path / "series.png").stat().st_size > 10_000


# --------------------------------------------------------------------------
# Pilotage manuel (cœur, sans écran)
# --------------------------------------------------------------------------
def test_stick_moves_and_springs_back():
    s = Stick()
    for _ in range(10):  # 0.2 s à 2.5/s
        s.update({"pitch_up"}, 0.02)
    assert s.pitch == pytest.approx(0.5)
    for _ in range(50):
        s.update({"pitch_up"}, 0.02)
    assert s.pitch == 1.0  # butée
    for _ in range(10):  # retour à 4/s
        s.update(set(), 0.02)
    assert s.pitch == pytest.approx(0.2)
    s.update({"roll_left", "roll_right"}, 0.02)  # deux touches opposées : neutre
    assert s.roll == 0.0


def test_stick_throttle_trim_and_joystick():
    s = Stick(throttle=0.5)
    for _ in range(50):
        s.update({"throttle_up", "trim_up"}, 0.02)
    assert s.throttle == pytest.approx(0.9)
    assert s.trim == pytest.approx(1.0 * DEG)
    s.update(set(), 0.02, axes=(0.3, -2.0, 0.1, 0.25))
    assert (s.roll, s.pitch, s.yaw, s.throttle) == (0.3, -1.0, 0.1, 0.25)
    assert set(INPUTS) >= {"pitch_up", "roll_right", "throttle_down"}


@pytest.mark.parametrize("kind", ["6dof", "3dof"])
def test_manual_flight_control_signs(kind):
    """Roulis à droite -> p > 0 et gîte > 0 ; tirer -> q > 0 et n_z > 1."""
    f = ManualFlight(kind)
    for _ in range(25):
        f.update({"roll_right"})
    assert f.instruments.p > 0 and f.instruments.roll > 10 * DEG
    f.reset()
    for _ in range(25):
        f.update({"pitch_up"})
    assert f.instruments.q > 0 and f.instruments.nz > 1.5


def test_manual_flight_pause_record_and_crash(tmp_path):
    f = ManualFlight("6dof", altitude=300.0, airspeed=200.0)
    f.paused = True
    f.update({"pitch_down"})
    assert f.t == 0.0 and len(f.recorder) == 0
    f.paused = False
    for _ in range(1500):  # manche en avant à fond : surcharge négative ou sol
        f.update({"pitch_down"})
        if f.ended:
            break
    assert f.ended and f.message
    t_end = f.t
    f.update({"pitch_down"})  # plus rien ne bouge après la fin
    assert f.t == t_end
    paths = f.save(tmp_path)
    assert set(paths) == {"npz", "acmi", "png"} and all(p.exists() for p in paths.values())
    rec = FlightRecording.load(paths["npz"])
    assert rec.events[-1][1] == f.message
    np.testing.assert_allclose(replay(rec), rec.states, atol=1e-9)


def test_manual_flight_rejects_unknown_model():
    with pytest.raises(ValueError):
        ManualFlight("2dof")


def test_hud_renders_headless(monkeypatch):
    pygame = pytest.importorskip("pygame")
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    from jetsim.viz.manual import HudRenderer

    pygame.init()
    surf = pygame.Surface((1100, 700))
    hud = HudRenderer()
    f = ManualFlight("6dof")
    for _ in range(20):
        f.update({"roll_right"})
    hud.draw(surf, f)
    f.paused = True
    hud.draw(surf, f, joystick_name="Test")
    # le ciel et le sol sont tous deux visibles en vol quasi horizontal
    colors = {tuple(surf.get_at((x, 300)))[:3] for x in range(20, 700, 20)}
    assert (70, 130, 200) in colors and (140, 95, 55) in colors
    pygame.quit()
