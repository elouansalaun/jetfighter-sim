"""Propulsion simplifiée : turboréacteur double flux avec post-combustion (PC).

Poussée en fonction de la **position de puissance** P ∈ [0, 1] :

* 0 → ``mil_power`` : du ralenti au plein gaz sec (MIL), interpolation linéaire ;
* ``mil_power`` → 1 : du plein gaz sec à la pleine PC (MAX).

Corrections :

* **altitude** : (ρ/ρ0)^n. Avec n ≈ 1.15, on retrouve à quelques % près les poussées
  statiques (Mach 0) des tables du modèle Stevens & Lewis entre 0 et 50 000 ft ;
* **effet d'admission** (ram) : facteur (1 + k·M), k interpolé entre ``ram_factor_mil``
  (régimes secs) et ``ram_factor_max`` (pleine PC) ;
* **limite moteur** : la poussée ne dépasse pas ``ram_limit`` × la poussée statique au sol
  du même régime. À basse altitude et grande vitesse, un vrai moteur est limité en
  température et en pression ; sans cette borne, l'effet d'admission donnerait un
  Mach max irréaliste au niveau de la mer.

La puissance suit la manette avec un retard du premier ordre (constante de temps
``engine_time_constant``) : l'agent RL ne peut pas obtenir la pleine poussée instantanément.

Ce modèle simple sert au point-masse (phase 2). Le modèle 6-DOF utilise
``TabulatedTurbofan`` (tables de poussée altitude × Mach de Stevens & Lewis), plus bas.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from jetfighter.aircraft.f16_aero import UniformGrid, lerp2
from jetfighter.aircraft.params import PropulsionParams
from jetfighter.core.constants import RHO0


class SimpleTurbofan:
    """Modèle de poussée paramétrique (voir le module)."""

    def __init__(self, params: PropulsionParams) -> None:
        self.p = params

    def altitude_factor(self, rho: float) -> float:
        """Rapport poussée(altitude) / poussée(niveau de la mer) à Mach nul."""
        return float((rho / RHO0) ** self.p.density_exponent)

    def thrust(self, power: float, rho: float, mach: float) -> float:
        """Poussée [N].

        Args:
            power: position de puissance effective, dans [0, 1] (saturée).
            rho: masse volumique de l'air [kg/m³].
            mach: nombre de Mach.
        """
        p = self.p
        power = min(max(power, 0.0), 1.0)
        if power <= p.mil_power:
            frac = power / p.mil_power
            t_sl = p.thrust_idle_sl + (p.thrust_mil_sl - p.thrust_idle_sl) * frac
            ram = p.ram_factor_mil
        else:
            frac = (power - p.mil_power) / (1.0 - p.mil_power)
            t_sl = p.thrust_mil_sl + (p.thrust_max_sl - p.thrust_mil_sl) * frac
            ram = p.ram_factor_mil + (p.ram_factor_max - p.ram_factor_mil) * frac
        factor = self.altitude_factor(rho) * (1.0 + ram * max(mach, 0.0))
        return t_sl * min(factor, p.ram_limit)

    def power_rate(self, power: float, throttle: float) -> float:
        """Dérivée de la puissance effective vers la consigne de manette (1er ordre)."""
        throttle = min(max(throttle, 0.0), 1.0)
        return (throttle - power) / self.p.engine_time_constant


# ==========================================================================
# Moteur tabulé du modèle Stevens & Lewis (utilisé par le modèle 6-DOF)
# ==========================================================================
class TabulatedTurbofan:
    """Moteur F100 du modèle Stevens & Lewis : tables de poussée (altitude × Mach) pour
    le ralenti, le plein gaz sec et la pleine post-combustion, et dynamique de puissance.

    La **puissance** est ici exprimée en fraction [0, 1] (0.5 = plein gaz sec, au-delà :
    post-combustion), soit le « pourcentage » du modèle d'origine divisé par 100.

    * consigne de puissance : loi « throttle gearing » (coude à 77 % de manette) ;
    * dynamique : 1er ordre dont la constante de temps dépend de l'écart (réponse lente
      pour les grands écarts), 5 s⁻¹ en post-combustion, passage sec <-> PC par paliers.
    """

    def __init__(self, engine: dict[str, Any], *, lbf_to_n: float, ft_to_m: float) -> None:
        self.alt = UniformGrid(
            engine["altitude_ft"]["start"] * ft_to_m,
            engine["altitude_ft"]["step"] * ft_to_m,
            engine["altitude_ft"]["count"],
        )
        self.mach = UniformGrid.from_dict(engine["mach"])
        self.idle = np.asarray(engine["idle_lbf"], dtype=float) * lbf_to_n
        self.mil = np.asarray(engine["mil_lbf"], dtype=float) * lbf_to_n
        self.max = np.asarray(engine["max_lbf"], dtype=float) * lbf_to_n
        gear = engine["throttle_gear"]
        self._gear = (
            float(gear["breakpoint"]),
            float(gear["slope_dry"]),
            float(gear["slope_ab"]),
            float(gear["offset_ab"]),
        )
        self._ab = float(engine["afterburner_threshold"]) / 100.0

    def commanded_power(self, throttle: float) -> float:
        """Consigne de puissance [0, 1] en fonction de la manette [0, 1]."""
        throttle = min(max(throttle, 0.0), 1.0)
        bp, s_dry, s_ab, off_ab = self._gear
        pct = s_dry * throttle if throttle <= bp else s_ab * throttle + off_ab
        return pct / 100.0

    def throttle_for_power(self, power: float) -> float:
        """Inverse de ``commanded_power`` (utile pour l'équilibrage)."""
        bp, s_dry, s_ab, off_ab = self._gear
        pct = power * 100.0
        return pct / s_dry if pct <= s_dry * bp else (pct - off_ab) / s_ab

    def thrust(self, power: float, altitude: float, mach: float) -> float:
        """Poussée [N] pour une puissance [0, 1], une altitude [m] et un Mach."""
        h = max(altitude, 0.0)
        t_mil = lerp2(self.mil, self.alt, h, self.mach, mach)
        if power < self._ab:
            t_idle = lerp2(self.idle, self.alt, h, self.mach, mach)
            return t_idle + (t_mil - t_idle) * power / self._ab
        t_max = lerp2(self.max, self.alt, h, self.mach, mach)
        return t_mil + (t_max - t_mil) * (power - self._ab) / (1.0 - self._ab)

    def power_rate(self, power: float, commanded: float) -> float:
        """Dérivée de la puissance [1/s] (fractions [0, 1])."""
        ab = self._ab
        if commanded >= ab:
            if power >= ab:
                inv_tau, target = 5.0, commanded
            else:
                target = 0.6
                inv_tau = _inverse_time_constant(target - power)
        elif power >= ab:
            inv_tau, target = 5.0, 0.4
        else:
            target = commanded
            inv_tau = _inverse_time_constant(target - power)
        return inv_tau * (target - power)


def _inverse_time_constant(delta: float) -> float:
    """1/τ du moteur selon l'écart de puissance (fraction) : 1 si ≤ 25 %, 0.1 si ≥ 50 %."""
    dp = delta * 100.0
    if dp <= 25.0:
        return 1.0
    if dp >= 50.0:
        return 0.1
    return 1.9 - 0.036 * dp
