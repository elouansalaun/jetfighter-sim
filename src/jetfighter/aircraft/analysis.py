"""Linéarisation et modes propres du modèle 6-DOF.

On linéarise autour d'un équilibre sur l'état réduit classique de la mécanique du vol :

    s = [V, α, β, φ, θ, p, q, r]

(la position, le cap ψ, le moteur et les servocommandes sont figés). On obtient ṡ ≈ A·δs + B·δu
avec δu = [puissance, δe, δa, δr] (gouvernes appliquées directement).

Modes attendus pour un avion stable :

* longitudinal (V, α, θ, q) : **oscillation d'incidence** (short period, rapide, bien amortie)
  et **phugoïde** (échange lent vitesse/altitude, peu amorti) ;
* latéral (β, φ, p, r) : **roulis pur** (réel, rapide), **roulis hollandais** (Dutch roll,
  oscillation lacet/roulis) et **spirale** (réel, lent, parfois légèrement instable).

Ces grandeurs serviront aussi aux lois de commande classiques (phase 6 : LQR).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from jetfighter.aircraft import dynamics_6dof as d6
from jetfighter.core.frames import (
    aero_angles,
    body_velocity_from_aero,
    euler_from_quat,
    euler_rates,
    quat_from_euler,
)

Vec = npt.NDArray[np.float64]

REDUCED_STATES = ("V", "alpha", "beta", "phi", "theta", "p", "q", "r")
REDUCED_INPUTS = ("power", "elevator", "aileron", "rudder")
LONGITUDINAL = (0, 1, 4, 6)  # V, α, θ, q
LATERAL = (2, 3, 5, 7)  # β, φ, p, r


def reduced_state(x: Vec) -> Vec:
    V, alpha, beta = aero_angles(x[d6.VEL])
    phi, theta, _ = euler_from_quat(x[d6.QUAT])
    return np.array([V, alpha, beta, phi, theta, *x[d6.RATES]])


def full_state(s: Vec, template: Vec) -> Vec:
    """État complet à partir de l'état réduit ``s`` (le reste est pris dans ``template``)."""
    x = template.copy()
    V, alpha, beta, phi, theta, p, q, r = s
    _, _, psi = euler_from_quat(template[d6.QUAT])
    x[d6.VEL] = body_velocity_from_aero(V, alpha, beta)
    x[d6.QUAT] = quat_from_euler(phi, theta, psi)
    x[d6.RATES] = (p, q, r)
    return x


def reduced_derivative(x: Vec, dx: Vec) -> Vec:
    """ṡ à partir de x et ẋ."""
    s = reduced_state(x)
    u, v, w = x[d6.VEL]
    du, dv, dw = dx[d6.VEL]
    V, _, beta = s[0], s[1], s[2]
    V_dot = (u * du + v * dv + w * dw) / V
    uw2 = u * u + w * w
    alpha_dot = (u * dw - w * du) / uw2
    beta_dot = (V * dv - v * V_dot) * math.cos(beta) / uw2
    phi_dot, theta_dot, _ = euler_rates(s[3], s[4], x[d6.RATES])
    return np.array([V_dot, alpha_dot, beta_dot, phi_dot, theta_dot, *dx[d6.RATES]])


def linearize(model: d6.F16SixDof, x_trim: Vec, u_trim: Vec, eps: float = 1e-6) -> tuple[Vec, Vec]:
    """Matrices (A, B) du modèle réduit par différences finies centrées."""
    s0 = reduced_state(x_trim)
    controls = np.array([x_trim[d6.POWER], *x_trim[d6.SURF]])

    def f(s: Vec, c: Vec) -> Vec:
        x = full_state(s, x_trim)
        x[d6.POWER] = c[0]
        x[d6.SURF] = c[1:]
        u = u_trim.copy()
        u[1:] = c[1:]
        return reduced_derivative(x, model.derivatives(0.0, x, u))

    n, m = s0.size, controls.size
    A = np.empty((n, n))
    B = np.empty((n, m))
    for j in range(n):
        h = eps * max(1.0, abs(s0[j]))
        sp, sm = s0.copy(), s0.copy()
        sp[j] += h
        sm[j] -= h
        A[:, j] = (f(sp, controls) - f(sm, controls)) / (2 * h)
    for j in range(m):
        h = eps
        cp, cm = controls.copy(), controls.copy()
        cp[j] += h
        cm[j] -= h
        B[:, j] = (f(s0, cp) - f(s0, cm)) / (2 * h)
    return A, B


@dataclass(frozen=True)
class Mode:
    name: str
    eigenvalue: complex

    @property
    def natural_frequency(self) -> float:
        return abs(self.eigenvalue)

    @property
    def damping(self) -> float:
        wn = self.natural_frequency
        return -self.eigenvalue.real / wn if wn > 0 else 0.0

    @property
    def period(self) -> float:
        im = abs(self.eigenvalue.imag)
        return 2 * math.pi / im if im > 0 else math.inf

    @property
    def time_constant(self) -> float:
        """τ = −1/Re(λ) (négatif si instable)."""
        re = self.eigenvalue.real
        return -1.0 / re if re != 0 else math.inf

    @property
    def stable(self) -> bool:
        return self.eigenvalue.real < 0


def _pairs(eigs: npt.NDArray[Any]) -> tuple[list[complex], list[float]]:
    oscillatory = sorted(
        (complex(e) for e in eigs if e.imag > 1e-9), key=lambda z: abs(z), reverse=True
    )
    real = sorted(float(e.real) for e in eigs if abs(e.imag) <= 1e-9)
    return oscillatory, real


def flight_modes(A: Vec) -> dict[str, Mode]:
    """Identifie les modes propres classiques à partir de la matrice A réduite."""
    modes: dict[str, Mode] = {}
    lon = np.linalg.eigvals(A[np.ix_(LONGITUDINAL, LONGITUDINAL)])
    osc, real = _pairs(lon)
    if len(osc) == 2:
        modes["short_period"] = Mode("oscillation d'incidence", osc[0])
        modes["phugoid"] = Mode("phugoïde", osc[1])
    else:  # oscillation d'incidence dégénérée en deux réels (avion instable par ex.)
        for i, z in enumerate(osc):
            modes[f"longitudinal_osc_{i}"] = Mode("oscillation longitudinale", z)
        for i, re in enumerate(real):
            modes[f"longitudinal_real_{i}"] = Mode("mode longitudinal apériodique", complex(re))

    lat = np.linalg.eigvals(A[np.ix_(LATERAL, LATERAL)])
    osc, real = _pairs(lat)
    if osc:
        modes["dutch_roll"] = Mode("roulis hollandais", osc[0])
    if real:
        modes["roll"] = Mode("roulis pur", complex(real[0]))
        if len(real) > 1:
            modes["spiral"] = Mode("spirale", complex(real[-1]))
    return modes
