"""Physical constants and conversion factors.

All internal quantities in the project are in SI units. The conversion factors
below are only used at the boundaries: reading data in imperial units
(the Stevens & Lewis F-16 model is in ft / slug / lbf) and display.

"""

from __future__ import annotations

import math

# --------------------------------------------------------------------------
# Gravity and air (International Standard Atmosphere, ISO 2533 / US Standard Atmosphere 1976)
# --------------------------------------------------------------------------
G0: float = 9.80665
"""Standard gravitational acceleration [m/s²]."""

R_AIR: float = 287.05287
"""Specific gas constant of dry air [J/(kg·K)]."""

GAMMA_AIR: float = 1.4
"""Ratio of specific heats of air (cp/cv) [-]."""

T0: float = 288.15
"""Sea-level temperature [K]."""

P0: float = 101_325.0
"""Sea-level pressure [Pa]."""

RHO0: float = P0 / (R_AIR * T0)
"""Sea-level density [kg/m³] (≈ 1.225)."""

A0: float = math.sqrt(GAMMA_AIR * R_AIR * T0)
"""Sea-level speed of sound [m/s] (≈ 340.294)."""

# --------------------------------------------------------------------------
# Unit conversions
# --------------------------------------------------------------------------
FT_TO_M: float = 0.3048
M_TO_FT: float = 1.0 / FT_TO_M

LBF_TO_N: float = 4.4482216152605
N_TO_LBF: float = 1.0 / LBF_TO_N

LB_TO_KG: float = 0.45359237
KG_TO_LB: float = 1.0 / LB_TO_KG

SLUG_TO_KG: float = LB_TO_KG * G0 / FT_TO_M  # ≈ 14.5939
KG_TO_SLUG: float = 1.0 / SLUG_TO_KG

SLUGFT2_TO_KGM2: float = SLUG_TO_KG * FT_TO_M**2  # moments of inertia, ≈ 1.35582
KGM2_TO_SLUGFT2: float = 1.0 / SLUGFT2_TO_KGM2

FT2_TO_M2: float = FT_TO_M**2
M2_TO_FT2: float = 1.0 / FT2_TO_M2

KT_TO_MS: float = 1852.0 / 3600.0  # knot -> m/s
MS_TO_KT: float = 1.0 / KT_TO_MS

DEG_TO_RAD: float = math.pi / 180.0
RAD_TO_DEG: float = 180.0 / math.pi
