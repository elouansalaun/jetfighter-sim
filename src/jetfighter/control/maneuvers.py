"""Primitives géométriques des manœuvres d'évitement classiques (utilisées en phase 10).

Ce sont les références « scriptées » auxquelles l'agent RL sera comparé. Face à une menace
dont on connaît le **gisement** (direction de la menace vue de l'avion, depuis le Nord) :

* **mise en travers** (*beam*) : voler perpendiculairement à la ligne de visée. La vitesse
  de rapprochement devient faible, ce qui gêne les autodirecteurs Doppler et oblige le
  missile à virer ;
* **fuite** (*drag*) : tourner le dos à la menace, pour allonger la poursuite et épuiser
  l'énergie du missile ;
* **virage de dernière seconde** (*break turn*) : virage au facteur de charge maximal, au
  dernier moment, pour créer une erreur de visée que le missile, moins manœuvrant à
  courte distance, ne peut plus rattraper.

Les deux premières fournissent un **cap** au pilote automatique ; le virage serré produit
directement une commande de haut niveau (n_z maximal, inclinaison vers la menace).
"""

from __future__ import annotations

import math

from jetfighter.aircraft.instruments import Instruments
from jetfighter.control.fbw import HighLevelCommand
from jetfighter.control.pid import wrap_angle


def bearing_to(own_north: float, own_east: float, north: float, east: float) -> float:
    """Gisement vrai [rad] d'un point vu depuis l'avion (0 = Nord, π/2 = Est)."""
    return math.atan2(east - own_east, north - own_north)


def relative_bearing(threat_bearing: float, heading: float) -> float:
    """Gisement relatif ∈ [−π, π[ : > 0 si la menace est à droite."""
    return wrap_angle(threat_bearing - heading)


def beam_heading(threat_bearing: float, heading: float) -> float:
    """Cap de mise en travers : perpendiculaire à la menace, du côté le plus proche du cap
    actuel (virage le plus court)."""
    options = (threat_bearing + math.pi / 2, threat_bearing - math.pi / 2)
    best = min(options, key=lambda h: abs(wrap_angle(h - heading)))
    return wrap_angle(best)


def drag_heading(threat_bearing: float) -> float:
    """Cap de fuite : dos à la menace."""
    return wrap_angle(threat_bearing + math.pi)


def break_turn(
    ins: Instruments,
    threat_bearing: float,
    *,
    nz: float = 9.0,
    bank: float = math.radians(80),
    roll_gain: float = 3.0,
    max_roll_rate: float = math.radians(240),
    throttle: float = 1.0,
) -> HighLevelCommand:
    """Virage serré **vers** la menace : inclinaison ``bank`` du côté de la menace, facteur de
    charge ``nz`` (la boucle interne le limite à ce que permettent α_max et n_max)."""
    side = 1.0 if relative_bearing(threat_bearing, ins.heading) >= 0 else -1.0
    bank_error = wrap_angle(side * bank - ins.bank)
    p_cmd = min(max(roll_gain * bank_error, -max_roll_rate), max_roll_rate)
    return HighLevelCommand(nz=nz, roll_rate=p_cmd, throttle=throttle)
