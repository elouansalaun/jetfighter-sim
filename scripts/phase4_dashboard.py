"""Phase 4 demonstration: instrument panel, noisy sensors and end of flight.

The 6-DOF F-16 starts from steady flight (3000 m, 200 m/s), pulls up, rolls
inverted then dives at full throttle (botched Split-S) until the envelope
monitor stops the flight.
The instrument panel is printed every second, as true and measured values
(sensors from ``src/jetsim/data/sensors/realistic.yaml``).


"""

from __future__ import annotations

import math

import numpy as np

from jetsim.aircraft import dynamics_6dof as d6
from jetsim.aircraft.envelope import EnvelopeMonitor
from jetsim.aircraft.instruments import read_instruments
from jetsim.aircraft.params import SENSORS_DIR, load_aircraft, load_envelope
from jetsim.aircraft.sensors import SensorSuite

DEG = math.pi / 180
DT, CONTROL_DT = 0.01, 0.1


def pilot(t: float, u_trim: np.ndarray) -> np.ndarray:
    """Scenario: pull-up (0-3 s), half roll (3-4 s), stabilize inverted (4-5 s),
    then Split-S: pull while inverted (the nose heads for the ground) and go full throttle,
    pulling too little to recover from the dive."""
    u = u_trim.copy()
    if t < 3.0:
        u[d6.ELEVATOR] -= 3 * DEG
    elif t < 4.0:
        u[d6.AILERON] = -12 * DEG
    elif t >= 5.0:
        u[d6.ELEVATOR] -= 2 * DEG if t < 8.0 else 0.0
        u[d6.THROTTLE] = 1.0
    return u


def main() -> None:
    model = d6.F16SixDof(load_aircraft("f16"))
    monitor = EnvelopeMonitor(load_envelope("f16", six_dof=True))
    sensors = SensorSuite.from_yaml(SENSORS_DIR / "realistic.yaml", np.random.default_rng(0))
    x, u_trim = model.trim(3000.0, 200.0)

    header = (
        f"{'t [s]':>6}{'alt [m]':>9}{'TAS':>6}{'CAS':>6}{'Mach':>6}{'α°':>6}"
        f"{'nz':>6}{'φ°':>6}{'θ°':>6}{'Ps':>7}   measured: {'alt':>6}{'α°':>6}{'nz':>6}"
    )
    print(header)
    print("-" * len(header))
    t, violation = 0.0, None
    steps = round(CONTROL_DT / DT)
    while violation is None and t < 60.0:
        u = pilot(t, u_trim)
        for _ in range(steps):
            x = model.step(x, u, DT)
        t += CONTROL_DT
        truth = read_instruments(model, x)
        meas = sensors.measure(truth)
        violation = monitor.check(truth, CONTROL_DT, state=x)
        if round(t / CONTROL_DT) % 10 == 0 or violation is not None:
            print(
                f"{t:6.1f}{truth.altitude:9.0f}{truth.tas:6.0f}{truth.cas:6.0f}"
                f"{truth.mach:6.2f}{truth.alpha / DEG:6.1f}{truth.nz:6.1f}"
                f"{truth.roll / DEG:6.0f}{truth.pitch / DEG:6.0f}"
                f"{truth.specific_excess_power:7.0f}   "
                f"         {meas.altitude:6.0f}{meas.alpha / DEG:6.1f}{meas.nz:6.2f}"
            )
    print(f"\nEnd of flight at t = {t:.1f} s — {monitor.message or 'maximum duration reached'}")


if __name__ == "__main__":
    main()
