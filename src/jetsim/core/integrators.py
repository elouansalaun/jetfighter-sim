"""Fixed-step integrators and a generic simulation loop.

Every dynamics in the project (3-DOF aircraft, 6-DOF, missile) is written in the form ::

    ẋ = f(t, x, u)

with x the state vector (1-D numpy) and u the control vector. The control is held
constant over one step (zero-order hold), as a flight computer would do.

Two rates coexist:

* the physics advances with a fixed step ``dt`` (0.01 s by default, 100 Hz);
* the controller (PID, RL agent) decides every ``control_dt`` (typically 0.05 to 0.1 s),
  i.e. a frame skip of ``control_dt / dt`` physics steps per decision.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

Vec = npt.NDArray[np.float64]
Dynamics = Callable[[float, Vec, Vec], Vec]
"""Signature of a dynamics function: ``f(t, x, u) -> ẋ``."""

Controller = Callable[[float, Vec], Vec]
"""Signature of a controller: ``u = controller(t, x)``."""

PostStep = Callable[[Vec], Vec]
"""Processing after each step (e.g. quaternion renormalization): ``x = post_step(x)``."""

DEFAULT_DT: float = 0.01
"""Default physics time step [s] (100 Hz)."""


# --------------------------------------------------------------------------
# Integration steps
# --------------------------------------------------------------------------
def euler_step(f: Dynamics, t: float, x: Vec, u: Vec, dt: float) -> Vec:
    """One explicit Euler step (order 1). Reserved for tests and comparisons."""
    return x + dt * f(t, x, u)


def rk4_step(f: Dynamics, t: float, x: Vec, u: Vec, dt: float) -> Vec:
    """One classical 4th-order Runge-Kutta step (the project's default integrator)."""
    half = 0.5 * dt
    k1 = f(t, x, u)
    k2 = f(t + half, x + half * k1, u)
    k3 = f(t + half, x + half * k2, u)
    k4 = f(t + dt, x + dt * k3, u)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


INTEGRATORS: dict[str, Callable[[Dynamics, float, Vec, Vec, float], Vec]] = {
    "euler": euler_step,
    "rk4": rk4_step,
}


# --------------------------------------------------------------------------
# Ratio between physics rate and decision rate
# --------------------------------------------------------------------------
def substeps_per_control(dt: float, control_dt: float) -> int:
    """Number of physics steps per controller decision (*frame skip*).

    Raises:
        ValueError: if ``control_dt`` is not an integer multiple of ``dt``.
    """
    ratio = control_dt / dt
    n = round(ratio)
    if n < 1 or not math.isclose(ratio, n, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(
            f"control_dt ({control_dt}) must be an integer multiple of dt ({dt}), ratio={ratio}"
        )
    return n


# --------------------------------------------------------------------------
# Simulation loop
# --------------------------------------------------------------------------
@dataclass
class SimResult:
    """History of a simulation. ``x[k]`` is the state at ``t[k]``; ``u[k]`` the control
    applied between ``t[k]`` and ``t[k+1]`` (the last row repeats the previous one)."""

    t: Vec
    x: Vec
    u: Vec


def simulate(
    f: Dynamics,
    x0: Vec,
    t_end: float,
    *,
    dt: float = DEFAULT_DT,
    controller: Controller | None = None,
    u_const: Vec | None = None,
    control_dt: float | None = None,
    method: str = "rk4",
    post_step: PostStep | None = None,
    t0: float = 0.0,
) -> SimResult:
    """Simulate ẋ = f(t, x, u) from ``t0`` to ``t_end`` with a fixed step.

    Args:
        f: dynamics ``f(t, x, u)``.
        x0: initial state.
        t_end: final time [s].
        dt: physics step [s].
        controller: control law ``u = controller(t, x)``. If ``None``, ``u_const`` is used.
        u_const: constant control (empty vector by default).
        control_dt: controller decision period [s] (default: ``dt``). Must be a
            multiple of ``dt``; the control is held between two decisions.
        method: ``"rk4"`` (default) or ``"euler"``.
        post_step: function applied to the state after each step (e.g. normalize the quaternion).
        t0: initial time [s].

    Returns:
        ``SimResult`` with ``n_steps + 1`` samples.
    """
    step = INTEGRATORS[method]
    n_sub = substeps_per_control(dt, control_dt if control_dt is not None else dt)
    n_steps = round((t_end - t0) / dt)
    if n_steps < 1:
        raise ValueError("t_end must exceed t0 by at least one step dt.")

    x = np.asarray(x0, dtype=np.float64).copy()
    u = np.zeros(0) if u_const is None else np.asarray(u_const, dtype=np.float64)
    if controller is not None:
        u = np.asarray(controller(t0, x), dtype=np.float64)

    ts = t0 + dt * np.arange(n_steps + 1, dtype=np.float64)
    xs = np.empty((n_steps + 1, x.size))
    us = np.empty((n_steps + 1, u.size))
    xs[0] = x

    for k in range(n_steps):
        t = ts[k]
        if controller is not None and k > 0 and k % n_sub == 0:
            u = np.asarray(controller(t, x), dtype=np.float64)
        us[k] = u
        x = step(f, t, x, u, dt)
        if post_step is not None:
            x = post_step(x)
        xs[k + 1] = x
    us[-1] = u
    return SimResult(t=ts, x=xs, u=us)
