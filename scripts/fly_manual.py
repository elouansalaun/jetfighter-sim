"""Pilotage manuel du F-16 au clavier (ou au joystick), en temps réel.

Le meilleur test de sanité du modèle : si un humain ne peut pas le faire voler, un agent RL
non plus. À la fin (Échap) ou sur demande (S), le vol est sauvegardé dans
``outputs/flights/`` : enregistrement ``.npz`` (rejouable), courbes ``.png`` et fichier
Tacview ``.acmi``.

Touches (repérées par le symbole imprimé : valables en AZERTY comme en QWERTY) :
    flèches haut / bas      piquer / cabrer
    flèches gauche / droite roulis
    W / X                   palonnier gauche / droite
    A / Q                   gaz + / −   (au-delà de 77 % : post-combustion)
    T / G                   trim à cabrer / à piquer
    Espace                  pause
    R                       recommencer
    S                       sauvegarder le vol en cours
    Échap                   quitter (sauvegarde automatique)

Joystick : détecté automatiquement (axes 0 roulis, 1 tangage, 2 lacet, 3 manette).

Installation : ``pip install -e ".[viz]"`` (pygame).

Usage :
    python scripts/fly_manual.py [--model 6dof|3dof] [--altitude 3000] [--speed 200]
"""

from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import matplotlib

matplotlib.use("Agg")  # les courbes sont enregistrées en fichier, sans fenêtre

try:
    import pygame
except ImportError as exc:
    sys.exit(
        f"Impossible d'importer pygame ({exc}).\n"
        f"Python utilisé : {sys.executable}\n"
        "Installez-le dans CE Python :\n"
        '  - environnement uv (.venv) : uv pip install -e ".[viz]"\n'
        f'  - sinon                   : {sys.executable} -m pip install -e ".[viz]"'
    )

from jetsim.viz.manual import HudRenderer, ManualFlight

FPS = 50
KEYS = {
    pygame.K_UP: "pitch_down",  # pousser le manche : piquer
    pygame.K_DOWN: "pitch_up",  # tirer le manche : cabrer
    pygame.K_LEFT: "roll_left",
    pygame.K_RIGHT: "roll_right",
    pygame.K_w: "yaw_left",
    pygame.K_x: "yaw_right",
    pygame.K_a: "throttle_up",
    pygame.K_q: "throttle_down",
    pygame.K_t: "trim_up",
    pygame.K_g: "trim_down",
}
DEADZONE = 0.05


def joystick_axes(joy) -> tuple[float, float, float, float | None]:
    def axis(i: int) -> float:
        if i >= joy.get_numaxes():
            return 0.0
        v = joy.get_axis(i)
        return 0.0 if abs(v) < DEADZONE else v

    throttle = (1.0 - axis(3)) / 2 if joy.get_numaxes() > 3 else None
    # axe 1 : manche vers soi = valeur positive = cabrer
    return axis(0), axis(1), axis(2), throttle


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--model", choices=("6dof", "3dof"), default="6dof")
    parser.add_argument("--altitude", type=float, default=3000.0)
    parser.add_argument("--speed", type=float, default=200.0)
    parser.add_argument("--out", default="outputs/flights")
    args = parser.parse_args()

    pygame.init()
    screen = pygame.display.set_mode((1100, 700))
    pygame.display.set_caption("jetfighter-sim — pilotage manuel")
    clock = pygame.time.Clock()
    hud = HudRenderer(screen.get_size())
    joy = None
    if pygame.joystick.get_count() > 0:
        joy = pygame.joystick.Joystick(0)
        joy.init()

    flight = ManualFlight(args.model, altitude=args.altitude, airspeed=args.speed,
                          frame_dt=1.0 / FPS)  # fmt: skip

    def save() -> None:
        paths = flight.save(args.out)
        for kind, path in paths.items():
            print(f"  {kind:4s} -> {path}")

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    flight.paused = not flight.paused
                elif event.key == pygame.K_r:
                    flight.reset()
                elif event.key == pygame.K_s:
                    print("Sauvegarde du vol en cours :")
                    save()
        pressed_keys = pygame.key.get_pressed()
        pressed = {name for key, name in KEYS.items() if pressed_keys[key]}
        flight.update(pressed, joystick_axes(joy) if joy else None)
        hud.draw(screen, flight, joy.get_name() if joy else None)
        pygame.display.flip()
        clock.tick(FPS)

    print("Vol terminé, sauvegarde :")
    save()
    pygame.quit()


if __name__ == "__main__":
    main()
