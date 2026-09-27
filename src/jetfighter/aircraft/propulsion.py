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

C'est un modèle volontairement simple. En phase 3, il pourra être remplacé par les tables
de poussée publiées (Mach × altitude) du modèle Stevens & Lewis sans changer l'interface.
"""

from __future__ import annotations

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
