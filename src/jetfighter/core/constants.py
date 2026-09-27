"""Constantes physiques et facteurs de conversion.

Toutes les grandeurs internes du projet sont en unités SI. Les facteurs de conversion
ci-dessous servent uniquement aux frontières : lecture de données en unités impériales
(le modèle F-16 de Stevens & Lewis est en ft / slug / lbf) et affichage.

Usage : ``longueur_m = 30.0 * FT_TO_M`` ; ``angle_deg = angle_rad * RAD_TO_DEG``.
"""

from __future__ import annotations

import math

# --------------------------------------------------------------------------
# Gravité et air (atmosphère standard internationale, ISO 2533 / US Standard Atmosphere 1976)
# --------------------------------------------------------------------------
G0: float = 9.80665
"""Accélération de la pesanteur standard [m/s²]."""

R_AIR: float = 287.05287
"""Constante spécifique de l'air sec [J/(kg·K)]."""

GAMMA_AIR: float = 1.4
"""Rapport des chaleurs spécifiques de l'air (cp/cv) [-]."""

T0: float = 288.15
"""Température au niveau de la mer [K]."""

P0: float = 101_325.0
"""Pression au niveau de la mer [Pa]."""

RHO0: float = P0 / (R_AIR * T0)
"""Masse volumique au niveau de la mer [kg/m³] (≈ 1.225)."""

A0: float = math.sqrt(GAMMA_AIR * R_AIR * T0)
"""Vitesse du son au niveau de la mer [m/s] (≈ 340.294)."""

# --------------------------------------------------------------------------
# Conversions d'unités
# --------------------------------------------------------------------------
FT_TO_M: float = 0.3048
M_TO_FT: float = 1.0 / FT_TO_M

LBF_TO_N: float = 4.4482216152605
N_TO_LBF: float = 1.0 / LBF_TO_N

LB_TO_KG: float = 0.45359237
KG_TO_LB: float = 1.0 / LB_TO_KG

SLUG_TO_KG: float = LB_TO_KG * G0 / FT_TO_M  # ≈ 14.5939
KG_TO_SLUG: float = 1.0 / SLUG_TO_KG

SLUGFT2_TO_KGM2: float = SLUG_TO_KG * FT_TO_M**2  # moments d'inertie, ≈ 1.35582
KGM2_TO_SLUGFT2: float = 1.0 / SLUGFT2_TO_KGM2

FT2_TO_M2: float = FT_TO_M**2
M2_TO_FT2: float = 1.0 / FT2_TO_M2

KT_TO_MS: float = 1852.0 / 3600.0  # nœud -> m/s
MS_TO_KT: float = 1.0 / KT_TO_MS

DEG_TO_RAD: float = math.pi / 180.0
RAD_TO_DEG: float = 180.0 / math.pi
