"""Repères, rotations et cinématique d'attitude.

Conventions (voir aussi le README) :

* **NED** : repère inertiel local, x Nord, y Est, z Bas (Terre plate). Altitude h = −z.
* **Corps (FRD)** : x vers le nez, y vers l'aile droite, z vers le ventre.
* **Vent** : x aligné sur la vitesse air, obtenu à partir du repère corps par α et β.
* **Quaternion** ``q = [q0, q1, q2, q3]`` : scalaire en premier, produit de Hamilton, unitaire.
  Il représente l'attitude de l'avion : ``v_ned = C_nb(q) @ v_body``.
* **Euler 3-2-1** : cap ψ, puis assiette θ, puis gîte φ. ``C_nb = Rz(ψ)·Ry(θ)·Rx(φ)``.
* **Notation** ``C_ab`` : matrice qui transforme un vecteur du repère b vers le repère a.

Les angles d'Euler ne servent qu'à l'affichage, aux observations et aux conditions initiales :
l'intégration de l'attitude se fait **toujours** sur le quaternion (pas de blocage de cardan
à θ = ±90°, indispensable pour le looping ou l'Immelmann).
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

Vec = npt.NDArray[np.float64]

_TWO_PI = 2.0 * math.pi


# --------------------------------------------------------------------------
# Utilitaires
# --------------------------------------------------------------------------
def wrap_angle(angle: float) -> float:
    """Ramène un angle dans [−π, π[ (utile pour les erreurs de cap)."""
    return (angle + math.pi) % _TWO_PI - math.pi


def skew(v: Vec) -> Vec:
    """Matrice antisymétrique [v×] telle que ``skew(a) @ b == np.cross(a, b)``."""
    return np.array(
        [
            [0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0],
        ]
    )


# --------------------------------------------------------------------------
# Quaternions
# --------------------------------------------------------------------------
def quat_identity() -> Vec:
    """Quaternion identité : avion à plat, nez au Nord."""
    return np.array([1.0, 0.0, 0.0, 0.0])


def quat_normalize(q: Vec) -> Vec:
    """Renvoie le quaternion unitaire de même direction (le signe est conservé).

    C'est la fonction à appeler après chaque pas d'intégration : elle ne change pas le signe,
    donc n'introduit aucune discontinuité dans la trajectoire de l'état.
    """
    n = math.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3])
    if n == 0.0:
        raise ValueError("Quaternion nul : impossible de normaliser.")
    return np.asarray(q, dtype=np.float64) / n


def quat_canonical(q: Vec) -> Vec:
    """Représentant unique : unitaire et de partie scalaire positive (q et −q = même attitude)."""
    qn = quat_normalize(q)
    return qn if qn[0] >= 0.0 else -qn


def quat_conjugate(q: Vec) -> Vec:
    """Conjugué q* (= inverse pour un quaternion unitaire)."""
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_multiply(a: Vec, b: Vec) -> Vec:
    """Produit de Hamilton a ⊗ b (composition : appliquer b puis a)."""
    a0, a1, a2, a3 = a
    b0, b1, b2, b3 = b
    return np.array(
        [
            a0 * b0 - a1 * b1 - a2 * b2 - a3 * b3,
            a0 * b1 + a1 * b0 + a2 * b3 - a3 * b2,
            a0 * b2 - a1 * b3 + a2 * b0 + a3 * b1,
            a0 * b3 + a1 * b2 - a2 * b1 + a3 * b0,
        ]
    )


def quat_from_axis_angle(axis: Vec, angle: float) -> Vec:
    """Quaternion d'une rotation d'angle ``angle`` [rad] autour de ``axis`` (normalisé ici)."""
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    s = math.sin(0.5 * angle)
    return np.array([math.cos(0.5 * angle), axis[0] * s, axis[1] * s, axis[2] * s])


def quat_derivative(q: Vec, omega_body: Vec, k_norm: float = 0.0) -> Vec:
    """Dérivée du quaternion d'attitude : q̇ = ½ · q ⊗ [0, ω].

    Args:
        q: quaternion corps -> NED.
        omega_body: vitesse angulaire du corps par rapport au repère NED, exprimée dans le
            repère corps : ``[p, q, r]`` [rad/s] (roulis, tangage, lacet).
        k_norm: gain optionnel de rappel vers |q| = 1 (terme ``k·(1 − |q|²)·q``). Laisser à 0
            si l'on renormalise après chaque pas (c'est le cas par défaut dans le projet).
    """
    q0, q1, q2, q3 = q
    p, qr, r = omega_body  # qr : vitesse de tangage (évite la collision avec le quaternion q)
    dq = 0.5 * np.array(
        [
            -q1 * p - q2 * qr - q3 * r,
            q0 * p + q2 * r - q3 * qr,
            q0 * qr + q3 * p - q1 * r,
            q0 * r + q1 * qr - q2 * p,
        ]
    )
    if k_norm:
        dq += k_norm * (1.0 - (q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3)) * np.asarray(q)
    return dq


# --------------------------------------------------------------------------
# Matrices de rotation (DCM)
# --------------------------------------------------------------------------
def dcm_from_quat(q: Vec) -> Vec:
    """Matrice C_nb (corps -> NED) à partir du quaternion d'attitude."""
    q0, q1, q2, q3 = q
    q00, q11, q22, q33 = q0 * q0, q1 * q1, q2 * q2, q3 * q3
    return np.array(
        [
            [q00 + q11 - q22 - q33, 2.0 * (q1 * q2 - q0 * q3), 2.0 * (q1 * q3 + q0 * q2)],
            [2.0 * (q1 * q2 + q0 * q3), q00 - q11 + q22 - q33, 2.0 * (q2 * q3 - q0 * q1)],
            [2.0 * (q1 * q3 - q0 * q2), 2.0 * (q2 * q3 + q0 * q1), q00 - q11 - q22 + q33],
        ]
    )


def quat_from_dcm(C_nb: Vec) -> Vec:
    """Quaternion à partir de C_nb (méthode de Shepperd, numériquement robuste)."""
    C = np.asarray(C_nb, dtype=np.float64)
    tr = C[0, 0] + C[1, 1] + C[2, 2]
    # On choisit le plus grand des 4 termes diagonaux pour éviter une division par ~0
    candidates = (tr, C[0, 0], C[1, 1], C[2, 2])
    k = int(np.argmax(candidates))
    if k == 0:
        s = 2.0 * math.sqrt(1.0 + tr)
        q = [0.25 * s, (C[2, 1] - C[1, 2]) / s, (C[0, 2] - C[2, 0]) / s, (C[1, 0] - C[0, 1]) / s]
    elif k == 1:
        s = 2.0 * math.sqrt(1.0 + C[0, 0] - C[1, 1] - C[2, 2])
        q = [(C[2, 1] - C[1, 2]) / s, 0.25 * s, (C[0, 1] + C[1, 0]) / s, (C[0, 2] + C[2, 0]) / s]
    elif k == 2:
        s = 2.0 * math.sqrt(1.0 + C[1, 1] - C[0, 0] - C[2, 2])
        q = [(C[0, 2] - C[2, 0]) / s, (C[0, 1] + C[1, 0]) / s, 0.25 * s, (C[1, 2] + C[2, 1]) / s]
    else:
        s = 2.0 * math.sqrt(1.0 + C[2, 2] - C[0, 0] - C[1, 1])
        q = [(C[1, 0] - C[0, 1]) / s, (C[0, 2] + C[2, 0]) / s, (C[1, 2] + C[2, 1]) / s, 0.25 * s]
    return quat_canonical(np.array(q))


def dcm_from_euler(phi: float, theta: float, psi: float) -> Vec:
    """Matrice C_nb (corps -> NED) à partir des angles d'Euler 3-2-1 [rad]."""
    cf, sf = math.cos(phi), math.sin(phi)
    ct, st = math.cos(theta), math.sin(theta)
    cp, sp = math.cos(psi), math.sin(psi)
    return np.array(
        [
            [ct * cp, sf * st * cp - cf * sp, cf * st * cp + sf * sp],
            [ct * sp, sf * st * sp + cf * cp, cf * st * sp - sf * cp],
            [-st, sf * ct, cf * ct],
        ]
    )


def euler_from_dcm(C_nb: Vec) -> tuple[float, float, float]:
    """Angles d'Euler 3-2-1 ``(φ, θ, ψ)`` [rad] à partir de C_nb. ψ ∈ [−π, π]."""
    s_theta = -min(max(C_nb[2, 0], -1.0), 1.0)
    phi = math.atan2(C_nb[2, 1], C_nb[2, 2])
    theta = math.asin(s_theta)
    psi = math.atan2(C_nb[1, 0], C_nb[0, 0])
    return phi, theta, psi


# --------------------------------------------------------------------------
# Euler <-> quaternion
# --------------------------------------------------------------------------
def quat_from_euler(phi: float, theta: float, psi: float) -> Vec:
    """Quaternion corps -> NED à partir des angles d'Euler 3-2-1 [rad]."""
    cf, sf = math.cos(0.5 * phi), math.sin(0.5 * phi)
    ct, st = math.cos(0.5 * theta), math.sin(0.5 * theta)
    cp, sp = math.cos(0.5 * psi), math.sin(0.5 * psi)
    q = np.array(
        [
            cf * ct * cp + sf * st * sp,
            sf * ct * cp - cf * st * sp,
            cf * st * cp + sf * ct * sp,
            cf * ct * sp - sf * st * cp,
        ]
    )
    return quat_canonical(q)


def euler_from_quat(q: Vec) -> tuple[float, float, float]:
    """Angles d'Euler 3-2-1 ``(φ, θ, ψ)`` [rad] à partir du quaternion.

    Au voisinage de θ = ±90° (nez à la verticale), φ et ψ ne sont plus définis séparément :
    le résultat reste fini mais leur répartition est arbitraire. C'est une limite des angles
    d'Euler, pas du modèle (qui intègre le quaternion).
    """
    q0, q1, q2, q3 = q
    phi = math.atan2(2.0 * (q0 * q1 + q2 * q3), q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3)
    s_theta = min(max(2.0 * (q0 * q2 - q1 * q3), -1.0), 1.0)
    theta = math.asin(s_theta)
    psi = math.atan2(2.0 * (q1 * q2 + q0 * q3), q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3)
    return phi, theta, psi


def euler_rates(phi: float, theta: float, omega_body: Vec) -> tuple[float, float, float]:
    """Dérivées des angles d'Euler ``(φ̇, θ̇, ψ̇)`` à partir de ``[p, q, r]``.

    Singulières à θ = ±90° : à n'utiliser que pour l'affichage ou des observations.
    """
    p, q, r = omega_body
    cf, sf = math.cos(phi), math.sin(phi)
    ct = math.cos(theta)
    if abs(ct) < 1e-9:
        raise ZeroDivisionError("Taux d'Euler non définis à θ = ±90° (blocage de cardan).")
    qs_rc = q * sf + r * cf
    return p + qs_rc * math.tan(theta), q * cf - r * sf, qs_rc / ct


# --------------------------------------------------------------------------
# Changements de repère de vecteurs
# --------------------------------------------------------------------------
def body_to_ned(q: Vec, v_body: Vec) -> Vec:
    """Exprime dans le repère NED un vecteur donné en repère corps."""
    return dcm_from_quat(q) @ v_body


def ned_to_body(q: Vec, v_ned: Vec) -> Vec:
    """Exprime dans le repère corps un vecteur donné en repère NED."""
    return dcm_from_quat(q).T @ v_ned


# --------------------------------------------------------------------------
# Repère aérodynamique (vent)
# --------------------------------------------------------------------------
def aero_angles(v_body: Vec) -> tuple[float, float, float]:
    """Vitesse air, incidence et dérapage à partir de la vitesse air en repère corps.

    Args:
        v_body: ``[u, v, w]`` vitesse de l'avion par rapport à l'air, en repère corps [m/s].

    Returns:
        ``(V, α, β)`` : norme [m/s], α = atan2(w, u) [rad], β = asin(v / V) [rad].
        Si V = 0, renvoie ``(0, 0, 0)``.
    """
    u, v, w = v_body
    V = math.sqrt(u * u + v * v + w * w)
    if V == 0.0:
        return 0.0, 0.0, 0.0
    alpha = math.atan2(w, u)
    beta = math.asin(min(max(v / V, -1.0), 1.0))
    return V, alpha, beta


def body_velocity_from_aero(V: float, alpha: float, beta: float) -> Vec:
    """Vitesse en repère corps ``[u, v, w]`` à partir de (V, α, β). Inverse de ``aero_angles``."""
    cb = math.cos(beta)
    return np.array([V * math.cos(alpha) * cb, V * math.sin(beta), V * math.sin(alpha) * cb])


def dcm_wind_to_body(alpha: float, beta: float) -> Vec:
    """Matrice C_bw (vent -> corps).

    Sert à projeter les efforts aérodynamiques, naturellement exprimés en axes vent
    (traînée D selon −x_w, force latérale Y selon y_w, portance L selon −z_w) :
    ``F_body = C_bw @ [-D, Y, -L]``.
    """
    ca, sa = math.cos(alpha), math.sin(alpha)
    cb, sb = math.cos(beta), math.sin(beta)
    return np.array(
        [
            [ca * cb, -ca * sb, -sa],
            [sb, cb, 0.0],
            [sa * cb, -sa * sb, ca],
        ]
    )


def flight_path_angles(v_ned: Vec) -> tuple[float, float, float]:
    """Vitesse sol, pente et route à partir de la vitesse en repère NED.

    Returns:
        ``(V, γ, χ)`` : norme [m/s], pente γ (positive en montée) [rad], route χ depuis le Nord
        vers l'Est [rad] (χ ∈ [−π, π]).
    """
    vn, ve, vd = v_ned
    v_horiz = math.hypot(vn, ve)
    V = math.hypot(v_horiz, vd)
    gamma = math.atan2(-vd, v_horiz)
    chi = math.atan2(ve, vn)
    return V, gamma, chi
