# jetfighter-sim — Roadmap du projet (phases 0 à 6)

> **Objectif final** : entraîner par apprentissage par renforcement (RL) un avion de chasse simulé (paramètres inspirés du F-16) à exécuter des manœuvres de vol, jusqu'à l'évitement d'un missile à autodirecteur.
>
> **Principe directeur** : on avance par couches. Chaque couche est **validée physiquement** avant d'être utilisée par la suivante. Un agent RL entraîné sur une physique fausse apprend à exploiter les bugs, pas à piloter.
>
> **Note** : le projet est réparti en deux dépôts. Ce document couvre le simulateur (phases 0 à 6) ; l'environnement d'apprentissage et le RL (phases 7 à 11) sont dans [jetfighter-rl](https://github.com/elouansalaun/jetfighter-rl/blob/main/project_roadmap.md).

---

## Journal des décisions

| Date | Décision | Raison | Conséquences |
|---|---|---|---|
| 27/09/2026 | Avion de référence : **F-16** (au lieu du F-22) | Données aérodynamiques publiques (NASA TP-1538, Stevens & Lewis) | Tables réelles dans le modèle 6-DOF (phase 3) |
| 27/09/2026 | Centrage par défaut **x_cg = 0.30** (option (a) de la phase 3) | Avion naturellement stable, plus simple pour démarrer | x_cg = 0.35 (instable, comme le vrai F-16) reste disponible ; les commandes de vol électriques de la phase 6 le stabilisent |
| 27/09/2026 | Agent **hiérarchique d'abord, bas niveau ensuite** | Apprentissage bien plus rapide ; permet de valider toute la chaîne RL (environnement, récompenses, curriculum) avant d'attaquer le problème difficile | Phases 7, 8 et 10 en deux temps : l'agent commande d'abord (n_z, taux de roulis, manette) via les commandes de vol électriques, puis directement les gouvernes. La commande directe reste l'**objectif final** du projet |
| 28/09/2026 | Pénalité de crash = fixe **+ coût maximal des pas restants** (dans l'horizon 1/(1−γ)) | Loin de la consigne, s'écraser coûtait moins que continuer à voler : incitation à abréger les épisodes | S'écraser est toujours la pire issue ; chaque tâche déclare sa borne de coût par pas (`max_step_cost`) |
| 28/09/2026 | Mode hiérarchique : **action nulle = trajectoire maintenue** (n_z = cos γ / cos μ) | Avec « action nulle = 1 g », la politique devait produire 1/cos μ au centième près pour ne pas monter ou descendre en virage | Ancien comportement disponible (`nz_neutral: one_g`) |
| 28/09/2026 | Commandes de vol : consigne de n_z limitée à 20 g/s + anticipation du limiteur par q | Des inversions brutales de commande (typiques d'un agent RL) dépassaient −4 g (jusqu'à −5 g) | Plus aucune sortie d'enveloppe en inversion ±1 ; réponses aux échelons inchangées |
| 28/09/2026 | Entraînement : récompenses normalisées (`VecNormalize`), torch sur 1 fil, exploration initiale réduite (σ = 0.37) | Variance expliquée du critique ≈ 0 sans normalisation ; débit ÷ 6 avec plusieurs fils ; commandes aléatoires à σ = 1 → crash en quelques secondes | Réglages par défaut de `jetfighter_rl.training` |
| 28/09/2026 | Imitation de la référence disponible **en option** (`pretrain`), RL pur par défaut | Sur le Mac, PPO seul apprend 8.2 → 8.4 avec 3 à 5 M de pas ; l'imitation reste utile pour la voltige (le clone du pilote scripté réussit les 4 figures) | Variantes `*_imitation.yaml` |
| 28/09/2026 | 8.2 et 8.5 : **départ par imitation** dans le programme principal | 3 exécutions de 8.2 en RL pur : 90 %, 60 %, 20 % ; en voltige, RL pur ne découvre que le tonneau | RL pur conservé en variante (`8_2_heading_altitude.yaml`, `8_5_aerobatics.yaml`) |
| 28/09/2026 | Bas niveau (8.6) à **50 Hz**, récompenses par pas × agent_dt / 0.1 s, démarrage par imitation de « pilote auto + commandes de vol » | Les commandes de vol ne tiennent pas l'avion à 10 Hz ; rendements comparables entre cadences | Épisodes 5 fois plus longs en pas d'agent |
| 28/09/2026 | Observation d'attitude : **direction de la pesanteur en axes vent et corps** (au lieu des angles d'Euler) | Au passage de la verticale, μ et φ basculent de 180° : l'agent de voltige montait à la verticale puis s'y arrêtait | Modèles antérieurs incompatibles (lus en mode `euler` automatiquement) ; programme à relancer |
| 29/09/2026 | Projet réparti en **deux dépôts** : `jetfighter-sim` (paquet `jetsim`, phases 0–6) et `jetfighter-rl` (paquet `jetfighter_rl`, phases 7+) | Le simulateur est réutilisable seul ; le dépôt RL en dépend comme d'une bibliothèque (dépendance git versionnée) | Données avion livrées dans le paquet (`jetsim/data/`) ; `jetfighter.rl` devient `jetfighter_rl.training` |

---

## Vue d'ensemble

| Phase | Contenu | Livrable clé | Durée indicative* |
|---|---|---|---|
| 0 | Cadrage, outillage, structure du dépôt | Dépôt initialisé, CI de tests | 2–3 jours |
| 1 | Fondations physiques (repères, atmosphère, intégrateur) | `core/` testé | 1 semaine |
| 2 | Modèle point-masse 3-DOF | Avion « rapide » pour prototyper le RL | 1 semaine |
| 3 | Modèle corps rigide 6-DOF avec gouvernes | Avion « réaliste » piloté par ses commandes | 3–4 semaines |
| 4 | Instruments, capteurs, enveloppe de vol | Tableau de bord + conditions de fin d'épisode | 1 semaine |
| 5 | Visualisation et pilotage manuel | Tracés, export Tacview, pilotage clavier | 1 semaine |
| 6 | Contrôleurs classiques (baseline) | Pilote automatique PID | 1–2 semaines |
| 7 | Environnement Gymnasium | `JetEnv` conforme `check_env`, modes hiérarchique et bas niveau | 1 semaine |
| 8 | Curriculum de manœuvres RL | Agents hiérarchiques, puis bas niveau : stabilisation → virage → voltige | 3–5 semaines |
| 9 | Modèle de missile (générique) | Missile à navigation proportionnelle | 1–2 semaines |
| 10 | RL d'évitement de missile | Agent d'évasion + carte de survie | 3–5 semaines |
| 11 | Extensions | Robustesse, multi-menaces, JSBSim, self-play | ouvert |

\* Estimations pour un développeur seul, en parallèle d'autres activités.

---

## Phase 0 — Cadrage et outillage

**But** : un dépôt propre dès le départ, pour pouvoir itérer vite sans casser l'existant.

- [x] Environnement Python 3.10+ (`uv` ou `venv`) et `pyproject.toml`
- [x] Dépendances de base : `numpy`, `scipy`, `matplotlib`, `pyyaml`, `pytest`
- [x] Dépendances RL (plus tard) : `gymnasium`, `stable-baselines3`, `torch`, `tensorboard`
- [x] Qualité : `ruff` (lint + format), `mypy` optionnel, `pre-commit`
- [x] Git + `.gitignore` (exclure `runs/`, `models/`, `*.acmi`)
- [x] Conventions écrites dans le README : **unités SI partout**, radians en interne, degrés uniquement en affichage

**Structure de dépôt proposée**

```
JetFighter_RL/
├── pyproject.toml
├── README.md
├── project_roadmap.md
├── configs/
│   ├── aircraft/f16.yaml             # masse, géométrie, poussée, coefficients aéro
│   ├── missile/generic_pn.yaml
│   └── training/*.yaml               # hyperparamètres par tâche
├── src/jetsim/
│   ├── core/
│   │   ├── constants.py              # g0, R, gamma air...
│   │   ├── atmosphere.py             # ISA : rho, T, P, a(h)
│   │   ├── frames.py                 # rotations, quaternions, Euler, NED <-> corps
│   │   └── integrators.py            # Euler, RK4
│   ├── aircraft/
│   │   ├── params.py                 # dataclass chargée depuis YAML
│   │   ├── aero.py                   # coefficients CL, CD, CY, Cl, Cm, Cn
│   │   ├── propulsion.py             # poussée(h, Mach, manette), post-combustion, lag moteur
│   │   ├── actuators.py              # limites de débattement / vitesse, retard 1er ordre
│   │   ├── dynamics_3dof.py          # modèle point-masse
│   │   ├── dynamics_6dof.py          # modèle corps rigide
│   │   └── sensors.py                # instruments + bruit
│   ├── missile/
│   │   ├── missile.py
│   │   ├── guidance.py               # navigation proportionnelle
│   │   └── seeker.py                 # champ de vue, limites de cardan
│   ├── control/
│   │   ├── pid.py
│   │   └── autopilot.py              # maintien altitude / cap / vitesse
│   ├── envs/
│   │   ├── base_env.py
│   │   ├── maneuver_envs.py
│   │   ├── evasion_env.py
│   │   └── rewards.py
│   └── viz/
│       ├── plots.py
│       └── tacview.py                # export .acmi
├── scripts/
│   ├── fly_manual.py
│   ├── train.py
│   └── evaluate.py
├── tests/
└── notebooks/                        # validation physique, analyses
```

---

## Phase 1 — Fondations physiques

**But** : les briques mathématiques dont tout le reste dépend. C'est ici que se cachent les bugs les plus sournois (signes, conventions d'angles).

### 1.1 Repères et conventions
- [x] Repère inertiel **NED** (North-East-Down), approximation Terre plate (suffisant à cette échelle)
- [x] Repère **corps** : x vers le nez, y aile droite, z vers le bas
- [x] Repère **aérodynamique / vent** (via α incidence et β dérapage)
- [x] Attitude en **quaternion** en interne (évite le blocage de cardan en voltige), conversion vers Euler (φ roulis, θ tangage, ψ cap) pour l'affichage et les observations
- [x] Normalisation du quaternion à chaque pas

### 1.2 Atmosphère standard (ISA)
- [x] Température, pression, masse volumique ρ(h), vitesse du son a(h) de 0 à 20 km (troposphère + début stratosphère)
- [x] Calcul du Mach et de la pression dynamique q̄ = ½ρV²

### 1.3 Intégrateur
- [x] RK4 à pas fixe, **dt physique = 0.01 s (100 Hz)**
- [x] Séparer la fréquence physique de la fréquence de décision de l'agent (ex. agent à 10–20 Hz → *frame skip* de 5 à 10)

### Tests de validation
- [x] Rotations : aller-retour Euler ↔ quaternion ↔ matrice (tolérance 1e-9)
- [x] ISA : comparer à des valeurs tabulées (ρ₀ = 1.225 kg/m³, a₀ = 340.3 m/s, ρ(11 km) ≈ 0.364 kg/m³)
- [x] Intégrateur : chute libre et oscillateur harmonique contre la solution analytique

---

## Phase 2 — Modèle point-masse 3-DOF (prototype rapide)

**But** : un modèle simple et très rapide pour prototyper les environnements et le RL pendant que le 6-DOF se construit. Il servira aussi pour les grandes campagnes d'évitement de missile.

**État** : position (x, y, h), vitesse V, pente γ, cap χ, masse m
**Commandes** : poussée T (ou manette), incidence α (ou facteur de charge n), angle de gîte μ

**Équations** :

```
V̇  = (T·cos α − D) / m − g·sin γ
γ̇  = [(L + T·sin α)·cos μ − m·g·cos γ] / (m·V)
χ̇  = (L + T·sin α)·sin μ / (m·V·cos γ)
ẋ  = V·cos γ·cos χ     ẏ = V·cos γ·sin χ     ḣ = V·sin γ
L  = q̄·S·CL(α, M)      D = q̄·S·(CD0(M) + k·CL²)
```

- [x] Implémentation + limites : α_max (décrochage), n ∈ [−3 g, +9 g], taux de variation de μ limité
- [x] Traînée : polaire parabolique CD = CD0(M) + k·CL², avec hausse de CD0 en transsonique
- [x] Tests : vol en palier stabilisé, virage à facteur de charge constant (taux de virage ω = g·√(n²−1)/V), montée, plafond

**Réalisé** (`src/jetsim/aircraft/`, validation : `scripts/phase2_performance.py`)
- Formulation **sans singularité** : les équations ci-dessus deviennent singulières à γ = ±90°. Le modèle intègre donc un quaternion du repère vent (angles d'Euler = χ, γ, μ), ce qui permet les loopings.
- Commandes retenues : `[manette, α commandée, taux de roulis autour de la vitesse]`, avec réponse du 1er ordre (α : 0,25 s ; roulis : 0,2 s ; moteur : 1 s), **limiteur d'incidence et de facteur de charge** façon commandes de vol électriques.
- Propulsion : poussée ∝ (ρ/ρ0)^1,15 (calée sur les poussées statiques Stevens & Lewis), effet d'admission linéaire en Mach, borné à 1,3 × la poussée statique au sol.
- Modules : `params.py` (YAML → dataclasses), `aero_polar.py`, `propulsion.py`, `dynamics_3dof.py` (dynamique, trim, instruments), `performance.py` (domaine de vol, virage, plafond).
- Performances obtenues vs F-16 publié : Mach max 1,18 au sol (≈ 1,2) et 2,08 à 11 km (≈ 2,0) ; montée 275 m/s (≈ 250) ; virage soutenu 18,3 °/s à 3 km M0,8 (≈ 18–20) ; ≈ 10 600 pas physiques/s.
- Limites connues : pas de décrochage, donc vitesse mini optimiste (≈ 45 m/s au sol contre ≈ 60–65 m/s réels) ; plafond aérodynamique 17,7 km contre 15,2 km publiés (limite opérationnelle) ; masse constante.

---

## Phase 3 — Modèle corps rigide 6-DOF

**But** : le vrai « avion », piloté par ses **commandes de vol** et non par des consignes abstraites.

### 3.1 Paramètres « F-16-like »
Contrairement au F-22, le F-16 dispose de **données aérodynamiques publiées** (soufflerie NASA, reprises par Stevens & Lewis) : c'est ce qui rend ce choix pertinent. Approche retenue :
- **Masse, géométrie, poussée** (valeurs publiques approximatives, à sourcer dans le YAML)
  - Masse de référence du modèle Stevens & Lewis : 1/m = 1,57·10⁻³ slug⁻¹, soit ≈ 9 295 kg (≈ 20 490 lb) (F-16C : ≈ 8 600 kg à vide, ≈ 19 200 kg max au décollage)
  - Envergure 9,14 m (30 ft), longueur ≈ 15,0 m, surface alaire 27,87 m² (300 ft²), corde moyenne 3,45 m (11,32 ft)
  - 1 moteur (F100-PW-229 ou F110-GE-129) : ≈ 76–79 kN à sec, ≈ 129–131 kN avec PC ; le modèle Stevens & Lewis utilise les tables de poussée d'un F100 plus ancien
  - Pas de vectorisation de poussée
  - Facteur de charge +9 g / −3 g ; Mach max ≈ 2,0 ; plafond ≈ 15 000 m
- **Coefficients aéro** : tables F-16 publiées (Stevens & Lewis ; Nguyen et al., NASA TP-1538), utilisées directement, sans mise à l'échelle
- **Inerties** (Stevens & Lewis) : Ixx = 9 496, Iyy = 55 814, Izz = 63 100, Ixz = 982 slug·ft² → ≈ 12 875 / 75 674 / 85 552 / 1 331 kg·m²
- [x] Tout dans `src/jetsim/data/aircraft/f16.yaml` → pouvoir changer d'avion sans toucher au code

### 3.2 État (13 variables + moteur)
- Position NED (3), vitesse corps u, v, w (3), quaternion (4), vitesses angulaires p, q, r (3)
- + état moteur (régime / poussée effective), + masse (consommation, optionnel)

### 3.3 Commandes de pilotage (entrées)
| Commande | Effet principal | Plage indicative |
|---|---|---|
| Manette des gaz δ_T (+ PC) | Poussée | 0 → 1 (PC au-delà d'un seuil) |
| Stabilisateurs monoblocs δ_e | Tangage (Cm) | ±25° |
| Ailerons / flaperons δ_a | Roulis (Cl) | ±20° |
| Gouvernes de direction δ_r | Lacet (Cn) | ±30° |

### 3.4 Forces et moments
- [x] Aéro : CL(α, δe, M), CD(α, M), CY(β, δr), Cl(β, δa, δr, p, r), Cm(α, δe, q), Cn(β, δa, δr, p, r)
  - Tables d'interpolation en α et Mach (`scipy.interpolate.RegularGridInterpolator`)
  - Dérivées d'amortissement (Cmq, Clp, Cnr) : indispensables, sinon l'avion oscille sans fin
- [x] Propulsion : T(h, M, δ_T) avec baisse en altitude (∝ ρ/ρ₀ environ), dynamique moteur du 1er ordre (constante de temps ~1 s, plus lente pour allumer la PC)
- [x] Gravité en repère NED projetée en corps
- [x] Équations de Newton-Euler (forces en corps + terme ω × V ; moments avec tenseur d'inertie incluant Ixz)

### 3.5 Actionneurs
- [x] Saturation en position **et** en vitesse (ex. 60°/s pour les gouvernes)
- [x] Retard du 1er ordre (τ ≈ 0,05 s)
- → Évite que l'agent RL apprenne des commandes « bang-bang » physiquement impossibles

### Tests de validation (dans `notebooks/`)
- [x] **Trim** : trouver (α, δe, δ_T) pour le vol en palier à plusieurs vitesses / altitudes (optimisation `scipy.optimize`)
- [x] **Modes propres** : linéariser autour du trim, vérifier la présence d'une oscillation d'incidence rapide (*short period*, ~1–3 s) et d'un phugoïde lent (~30–100 s)
- [x] **Réponses indicielles** : échelon de profondeur, d'ailerons, de direction — signes et ordres de grandeur cohérents
- [x] **Conservation** sans aéro ni poussée ni gravité : énergie de rotation et moment cinétique (moteur inclus) conservés
- [x] **Cohérence 3-DOF vs 6-DOF** sur un virage stabilisé (vérifiée en phase 6 avec le pilote automatique : taux de virage à 4 et 5 g à moins de 5 % près)

> ⚠️ Point d'attention : un F-16 réel est **instable par conception** et piloté via des commandes de vol électriques. Pour le RL, deux options : (a) rendre le modèle légèrement stable statiquement (plus facile), (b) garder l'instabilité et ajouter une loi de commande de stabilisation (Phase 6). Commencer par (a).

**Réalisé** (`dynamics_6dof.py`, `f16_aero.py`, `analysis.py`, tables dans `src/jetsim/data/aircraft/f16_sl_tables.yaml`, validation : `scripts/phase3_validation.py`)
- Tables aéro et moteur Stevens & Lewis complètes (données NASA TP-1538), réimplémentées (le code de référence AeroBench est sous GPL-3 : seules les valeurs numériques ont été reprises).
- Vérifications : dérivées identiques à l'implémentation de référence à 2·10⁻⁴ près sur 300 états aléatoires (écart = arrondi des constantes d'inertie du livre) ; cas de vérification publié retrouvé sur 11 dérivées sur 13 (ṗ et ṙ diffèrent de ≈ 0.001 sur Cl dans toutes les implémentations disponibles) ; trim publié retrouvé (manette 0.1385, δe −0.759°).
- **Centrage** : au nominal x_cg = 0.35, l'avion est instable en tangage (temps de doublement ≈ 7 s), comme le vrai F-16. Option (a) retenue : x_cg = 0.30 par défaut → avion stable dans tout le domaine subsonique (sauf divergence lente de second régime à α > 11°). Le 0.35 reste disponible pour la phase 6.
- Modes à 3000 m / 200 m/s : oscillation d'incidence ω = 2.1 rad/s ζ = 0.56 ; phugoïde 107 s ; roulis hollandais 1.8 s ζ = 0.12 ; roulis τ = 0.28 s ; spirale stable.
- Limites : tables sans effet du Mach (fiables jusqu'à ≈ Mach 0.6–0.8) ; ≈ 1 700–2 000 pas physiques/s en Python pur (à accélérer en phase 7) ; cohérence 3-DOF/6-DOF vérifiée sur l'incidence d'équilibre en palier (< 1°), pas encore en virage.

---

## Phase 4 — Instruments, capteurs et enveloppe de vol

### 4.1 Instruments (sorties observables)
- [x] Vitesses : TAS, CAS (approx.), Mach, vitesse verticale
- [x] Altitude, cap, attitude (φ, θ, ψ)
- [x] Incidence α, dérapage β
- [x] Vitesses angulaires p, q, r
- [x] Accélérations en corps, **facteur de charge Nz**
- [x] Énergie spécifique Es = h + V²/2g et excès de puissance Ps (très utile pour l'évitement de missile)
- [x] Option : bruit et biais capteurs (pour la robustesse, Phase 11)

### 4.2 Enveloppe de vol et fins d'épisode
- [x] Collision sol (h < 0 ou h < h_min de sécurité)
- [x] Dépassement de facteur de charge (limite structurelle)
- [x] Décrochage prolongé / vrille (α > α_max pendant > N s)
- [x] Sortie de domaine (altitude > plafond, vitesse < V_min)
- [x] Détection NaN / divergence numérique → fin d'épisode + log

**Réalisé** (`instruments.py`, `sensors.py`, `envelope.py`, limites dans `src/jetsim/data/aircraft/f16.yaml` section `envelope`, démo : `scripts/phase4_dashboard.py`)
- **Tableau de bord commun** aux modèles 3-DOF et 6-DOF (`read_instruments`) : 28 grandeurs (position, TAS/CAS/EAS, Mach, variomètre, γ/χ, inclinaison μ du vecteur portance (ajoutée en phase 6), attitude φ/θ/ψ, α/β, p/q/r, facteurs de charge nx/ny/nz, énergie E et Ps, poussée). Pour le 3-DOF, attitude et vitesses angulaires corps reconstruites à partir du repère vent et de α.
- **CAS** exacte (pression d'impact isentropique en subsonique, formule de Rayleigh en supersonique) et son inverse.
- **Capteurs** : bruit blanc + biais tiré à chaque épisode, par voie, reproductible (générateur fourni) ; parfaits par défaut ; exemple réaliste dans `src/jetsim/data/sensors/realistic.yaml`.
- **Fins d'épisode** : NaN, sol, surcharge (−4 / +10 g), plafond (17 km), vitesse mini (50 m/s), Mach max (2.0 en 3-DOF, 0.95 en 6-DOF), dérapage (30°), décrochage prolongé (α > 30° pendant > 2 s).

---

## Phase 5 — Visualisation et pilotage manuel

- [x] Tracés matplotlib : trajectoire 3D, séries temporelles (V, h, α, Nz, commandes)
- [x] **Export Tacview (.acmi)** : format texte simple, gratuit en lecture, rendu 3D très parlant pour analyser les manœuvres et les engagements missile
- [x] `scripts/fly_manual.py` : pilotage clavier ou joystick (`pygame`) — le meilleur test de sanité : si un humain ne peut pas faire voler le modèle, un agent RL non plus
- [x] Rejeu d'épisodes enregistrés (états + actions en `.npz`)

**Réalisé** (`src/jetsim/viz/` : `recorder.py`, `plots.py`, `tacview.py`, `manual.py` ; scripts `demo_flight.py` et `fly_manual.py`)
- **Enregistreur** : temps, états, commandes, 27 instruments et événements → `.npz` ; **rejeu déterministe** (états retrouvés au bit près) ; le modèle est reconstruit à partir de l'enregistrement.
- **Courbes** : tableau de bord temporel (8 graphiques) et trajectoire 3D avec trace au sol.
- **Tacview** : export ACMI 2.2 (position, attitude, TAS/CAS/Mach/AOA/AOS/Throttle, événements), écrivain multi-objets prêt pour le missile (phase 9). Monde plat posé au-dessus de l'Atlantique (46° N, 6° O).
- **Pilotage manuel** (pygame, extra `[viz]`) : horizon artificiel + instruments, manche à ressort au clavier (AZERTY/QWERTY), trim, joystick détecté automatiquement (non testé sur matériel réel), 3-DOF ou 6-DOF, sauvegarde automatique (.npz, .acmi, .png).
- Démo `demo_flight.py` : looping puis tonneau enchaînés par un séquenceur qui surveille les instruments. En 6-DOF, régler ces figures « à la main » est délicat : motivation directe pour les phases 6 à 8.

---

## Phase 6 — Contrôleurs classiques (baseline)

**But** : disposer d'une référence pour juger l'agent RL, et d'une couche basse optionnelle pour un RL hiérarchique.

- [x] Amortisseurs de tangage / roulis / lacet (*stability augmentation*)
- [x] PID : maintien de vitesse, d'altitude, de cap, virage coordonné (β → 0)
- [x] Option : LQR sur le modèle linéarisé autour du trim
- [~] Heuristiques d'évitement de missile scriptées (pour la Phase 10) — mise en travers (*beam*), fuite (*drag*), virage serré de dernière seconde : primitives géométriques prêtes ; la logique complète attend le missile (phase 9)

**Décision d'architecture** (prise le 27/09/2026, cf. journal des décisions) :

| Option | L'agent commande… | + | − |
|---|---|---|---|
| Bas niveau | directement les gouvernes et la manette | fidèle à l'objectif initial | apprentissage long et instable |
| Hiérarchique | des consignes (Nz, taux de roulis, poussée) suivies par un contrôleur | apprentissage beaucoup plus rapide | dépend de la qualité du contrôleur |

→ **Décision : hiérarchique d'abord, puis bas niveau.** L'agent pilote d'abord via `HighLevelCommand` (n_z, taux de roulis, manette) exécutée par les commandes de vol électriques ; une fois les tâches maîtrisées, on repasse en commande directe des gouvernes (étape 8.6, puis phase 10).

**Réalisé** (`src/jetsim/control/` : `pid.py`, `fbw.py`, `autopilot.py`, `lqr.py`, `maneuvers.py` ; démo : `scripts/phase6_autopilot.py`)
- **Interface commune de haut niveau** `HighLevelCommand(nz, roll_rate, throttle)`, exécutée par une boucle interne propre à chaque modèle (`make_inner_loop`) : c'est l'espace d'action de l'option **hiérarchique** pour le RL.
- **Commandes de vol électriques 6-DOF** : PI sur n_z + amortissement en q, PI sur le taux de roulis, β → 0 + amortisseur de lacet (filtre passe-haut), limiteurs de facteur de charge (−3/+9 g) et d'incidence (≤ 25°), gains programmés en q̄. Gains réglés par recherche systématique en simulation : échelon 1 → 4 g en 0.4–0.9 s ; 90°/s en roulis en 0.2–0.3 s avec |β| < 1°. **Elles stabilisent l'avion aux centrages instables** (0.35, 0.40) : l'option (b) de la phase 3 est désormais disponible.
- **Pilote automatique** (mêmes gains pour 3-DOF et 6-DOF) : altitude → vitesse verticale → pente → n_z (+ 1/cos μ en virage), cap → inclinaison μ → taux de roulis, vitesse → manette. Les deux modèles volent la même mission de façon quasi identique (figure `outputs/phase6_mission.png`).
- **LQR** longitudinal sur le modèle linéarisé : stabilise x_cg = 0.35 et 0.40 autour du point de conception (l'avion libre diverge).
- **Primitives d'évitement** : gisements, cap de mise en travers (*beam*), cap de fuite (*drag*), virage serré vers la menace (*break turn*).
- Enseignement : l'inclinaison à asservir est celle du **vecteur portance** (μ), pas la gîte du fuselage (φ) ; l'écart, négligeable en croisière, fait échouer les virages serrés (ajout de `bank` au tableau de bord).
- Les deux options sont possibles techniquement ; décision prise : **hiérarchique d'abord, bas niveau ensuite** (voir plus haut).

---

---

## Références utiles

- Stevens, Lewis & Johnson — *Aircraft Control and Simulation* (modèle F-16 complet, équations 6-DOF)
- Nguyen et al. (1979) — NASA TP-1538, données aérodynamiques F-16 en soufflerie
- Zarchan — *Tactical and Strategic Missile Guidance* (navigation proportionnelle)
- Documentation [Gymnasium](https://gymnasium.farama.org/) et [Stable-Baselines3](https://stable-baselines3.readthedocs.io/)
- [JSBSim](https://github.com/JSBSim-Team/jsbsim) — moteur de dynamique de vol open source
- Format ACMI de [Tacview](https://www.tacview.net/) pour la visualisation
- DARPA AlphaDogfight Trials (2020) — contexte sur le RL en combat aérien simulé
