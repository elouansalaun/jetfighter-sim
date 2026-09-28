"""Commande optimale LQR sur le modèle linéarisé (option de la phase 6).

Le régulateur linéaire quadratique minimise ∫ (δxᵀ·Q·δx + δuᵀ·R·δu) dt pour ẋ = A·x + B·u ;
la solution est un retour d'état u = −K·δx, avec K = R⁻¹·Bᵀ·S et S solution de l'équation
de Riccati algébrique.

``LongitudinalLQR`` stabilise le mouvement longitudinal (V, α, θ, q) autour d'un vol en
palier avec la seule profondeur, la manette restant à sa valeur d'équilibre. Il rend l'avion
stable même au centrage instable (x_cg = 0.35), mais seulement **autour de son point de
conception** : c'est un bon régulateur local, pas une loi de pilotage (voir ``fbw.py``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.linalg import solve_continuous_are

from jetfighter.aircraft import analysis as an
from jetfighter.aircraft import dynamics_6dof as d6
from jetfighter.aircraft.instruments import Instruments

Vec = npt.NDArray[np.float64]
CVec = npt.NDArray[Any]  # valeurs propres (complexes)
DEG = np.pi / 180


def lqr(A: Vec, B: Vec, Q: Vec, R: Vec) -> tuple[Vec, Vec, CVec]:
    """Gain LQR en temps continu.

    Returns:
        ``(K, S, valeurs propres en boucle fermée)``.
    """
    S = np.asarray(solve_continuous_are(A, B, Q, R), dtype=np.float64)
    K = np.asarray(np.linalg.solve(R, B.T @ S), dtype=np.float64)
    return K, S, np.linalg.eigvals(A - B @ K)


@dataclass
class LongitudinalLQR:
    """Régulateur de profondeur δe = δe_eq − K·(s − s_eq), s = [V, α, θ, q]."""

    K: Vec  # (1, 4)
    s_eq: Vec  # (4,)
    elevator_eq: float
    throttle_eq: float
    open_loop: CVec  # valeurs propres longitudinales sans régulateur
    closed_loop: CVec  # avec régulateur
    elevator_limits: tuple[float, float]

    @classmethod
    def design(
        cls,
        model: d6.F16SixDof,
        altitude: float,
        airspeed: float,
        state_weights: tuple[float, float, float, float] = (1e-2, 100.0, 10.0, 10.0),
        control_weight: float = 100.0,
    ) -> LongitudinalLQR:
        """Conception au point d'équilibre (altitude, vitesse). Poids par défaut :
        V faiblement pénalisée, α et θ fortement, δe coûteuse (commandes douces)."""
        x, u = model.trim(altitude, airspeed)
        A, B = an.linearize(model, x, u)
        lon = list(an.LONGITUDINAL)
        A_lon = A[np.ix_(lon, lon)]
        B_lon = B[lon, 1:2]  # colonne profondeur
        K, _, closed = lqr(A_lon, B_lon, np.diag(state_weights), np.array([[control_weight]]))
        s = an.reduced_state(x)[lon]
        cs = model.params.control_surfaces["elevator"]
        return cls(
            K=K,
            s_eq=s,
            elevator_eq=float(u[d6.ELEVATOR]),
            throttle_eq=float(u[d6.THROTTLE]),
            open_loop=np.linalg.eigvals(A_lon),
            closed_loop=closed,
            elevator_limits=(cs.min, cs.max),
        )

    def __call__(self, ins: Instruments) -> Vec:
        """Commande 6-DOF [manette, δe, δa, δr] (ailerons et direction au neutre)."""
        s = np.array([ins.tas, ins.alpha, ins.pitch, ins.q])
        de = self.elevator_eq - float((self.K @ (s - self.s_eq))[0])
        de = min(max(de, self.elevator_limits[0]), self.elevator_limits[1])
        return np.array([self.throttle_eq, de, 0.0, 0.0])
