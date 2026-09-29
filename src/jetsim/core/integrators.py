"""Intégrateurs à pas fixe et boucle de simulation générique.

Toute dynamique du projet (avion 3-DOF, 6-DOF, missile) s'écrit sous la forme ::

    ẋ = f(t, x, u)

avec x le vecteur d'état (numpy 1-D) et u le vecteur de commande. La commande est maintenue
**constante pendant un pas** (bloqueur d'ordre zéro), comme le ferait un calculateur de vol.

Deux fréquences coexistent :

* la **physique** avance à pas fixe ``dt`` (0.01 s par défaut, 100 Hz) ;
* le **contrôleur** (PID, agent RL) décide toutes les ``control_dt`` (typiquement 0.05 à 0.1 s),
  soit un *frame skip* de ``control_dt / dt`` pas physiques par décision.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

Vec = npt.NDArray[np.float64]
Dynamics = Callable[[float, Vec, Vec], Vec]
"""Signature d'une dynamique : ``f(t, x, u) -> ẋ``."""

Controller = Callable[[float, Vec], Vec]
"""Signature d'un contrôleur : ``u = controller(t, x)``."""

PostStep = Callable[[Vec], Vec]
"""Traitement après chaque pas (ex. renormalisation du quaternion) : ``x = post_step(x)``."""

DEFAULT_DT: float = 0.01
"""Pas de temps physique par défaut [s] (100 Hz)."""


# --------------------------------------------------------------------------
# Pas d'intégration
# --------------------------------------------------------------------------
def euler_step(f: Dynamics, t: float, x: Vec, u: Vec, dt: float) -> Vec:
    """Un pas d'Euler explicite (ordre 1). Réservé aux tests et comparaisons."""
    return x + dt * f(t, x, u)


def rk4_step(f: Dynamics, t: float, x: Vec, u: Vec, dt: float) -> Vec:
    """Un pas de Runge-Kutta classique d'ordre 4 (intégrateur par défaut du projet)."""
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
# Rapport entre fréquence physique et fréquence de décision
# --------------------------------------------------------------------------
def substeps_per_control(dt: float, control_dt: float) -> int:
    """Nombre de pas physiques par décision du contrôleur (*frame skip*).

    Raises:
        ValueError: si ``control_dt`` n'est pas un multiple entier de ``dt``.
    """
    ratio = control_dt / dt
    n = round(ratio)
    if n < 1 or not math.isclose(ratio, n, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(
            f"control_dt ({control_dt}) doit être un multiple entier de dt ({dt}), ratio={ratio}"
        )
    return n


# --------------------------------------------------------------------------
# Boucle de simulation
# --------------------------------------------------------------------------
@dataclass
class SimResult:
    """Historique d'une simulation. ``x[k]`` est l'état à ``t[k]`` ; ``u[k]`` la commande
    appliquée entre ``t[k]`` et ``t[k+1]`` (la dernière ligne répète la précédente)."""

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
    """Simule ẋ = f(t, x, u) de ``t0`` à ``t_end`` à pas fixe.

    Args:
        f: dynamique ``f(t, x, u)``.
        x0: état initial.
        t_end: instant final [s].
        dt: pas physique [s].
        controller: loi de commande ``u = controller(t, x)``. Si ``None``, ``u_const`` est utilisé.
        u_const: commande constante (vecteur vide par défaut).
        control_dt: période de décision du contrôleur [s] (défaut : ``dt``). Doit être un
            multiple de ``dt`` ; la commande est maintenue entre deux décisions.
        method: ``"rk4"`` (défaut) ou ``"euler"``.
        post_step: fonction appliquée à l'état après chaque pas (ex. normaliser le quaternion).
        t0: instant initial [s].

    Returns:
        ``SimResult`` avec ``n_steps + 1`` échantillons.
    """
    step = INTEGRATORS[method]
    n_sub = substeps_per_control(dt, control_dt if control_dt is not None else dt)
    n_steps = round((t_end - t0) / dt)
    if n_steps < 1:
        raise ValueError("t_end doit être supérieur à t0 d'au moins un pas dt.")

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
