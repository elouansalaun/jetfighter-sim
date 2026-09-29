# jetfighter-sim: Project Roadmap (Phases 0 to 6)

> **Final goal**: use reinforcement learning (RL) to train a simulated fighter jet (parameters inspired by the F-16) to fly maneuvers, up to evading a homing missile.
>
> **Guiding principle**: we build in layers. Each layer is **physically validated** before the next one uses it. An RL agent trained on wrong physics learns to exploit bugs, not to fly.
>
> **Note**: the project is split across two repositories. This document covers the simulator (phases 0 to 6); the learning environment and RL (phases 7 to 11) live in [jetfighter-rl](https://github.com/elouansalaun/jetfighter-rl/blob/main/project_roadmap.md).

***

## Decision Log

 Decision | Reason | Consequences |
---|---|---|
 Reference aircraft: **F-16** (instead of the F-22) | Public aerodynamic data (NASA TP-1538, Stevens & Lewis) | Real tables in the 6-DOF model (phase 3) |
 Default CG position **x_cg = 0.30** (option (a) from phase 3) | Naturally stable aircraft, simpler to start with | x_cg = 0.35 (unstable, like the real F-16) remains available; the phase 6 fly by wire system stabilizes it |
 **Hierarchical agent first, low level second** | Much faster learning; lets us validate the whole RL chain (environment, rewards, curriculum) before tackling the hard problem | Phases 7, 8 and 10 run in two stages: the agent first commands (n_z, roll rate, throttle) through the fly by wire system, then the control surfaces directly. Direct control remains the **final goal** of the project |
 Crash penalty = fixed amount **+ maximum cost of the remaining steps** (within the 1/(1−γ) horizon) | Far from the target, crashing cost less than continuing to fly: an incentive to cut episodes short | Crashing is always the worst outcome; each task declares its per step cost bound (`max_step_cost`) |
 Hierarchical mode: **zero action = hold the current flight path** (n_z = cos γ / cos μ) | With "zero action = 1 g", the policy had to output 1/cos μ to within a hundredth to avoid climbing or descending in turns | Previous behavior still available (`nz_neutral: one_g`) |
 Fly by wire: n_z command rate limited to 20 g/s + limiter anticipation using q | Abrupt command reversals (typical of an RL agent) exceeded −4 g (down to −5 g) | No more envelope exits on ±1 reversals; step responses unchanged |
 Training: normalized rewards (`VecNormalize`), torch on 1 thread, reduced initial exploration (σ = 0.37) | Critic explained variance ≈ 0 without normalization; throughput ÷ 6 with multiple threads; random commands at σ = 1 → crash within seconds | Default settings of `jetfighter_rl.training` |
 Imitation of the reference available **as an option** (`pretrain`), pure RL by default | On the Mac, PPO alone learns 8.2 → 8.4 in 3 to 5 M steps; imitation remains useful for aerobatics (the clone of the scripted pilot succeeds at all 4 maneuvers) | `*_imitation.yaml` variants |
 8.2 and 8.5: **imitation warm start** in the main curriculum | 3 pure RL runs of 8.2: 90 %, 60 %, 20 %; in aerobatics, pure RL only discovers the aileron roll | Pure RL kept as a variant (`8_2_heading_altitude.yaml`, `8_5_aerobatics.yaml`) |
 Low level (8.6) at **50 Hz**, per step rewards × agent_dt / 0.1 s, warm start by imitating "autopilot + fly by wire" | The fly by wire system cannot hold the aircraft at 10 Hz; returns stay comparable across rates | Episodes 5 times longer in agent steps |
 Attitude observation: **gravity direction in wind and body axes** (instead of Euler angles) | When passing through the vertical, μ and φ flip by 180°: the aerobatics agent climbed to the vertical and then stopped there | Earlier models incompatible (automatically loaded in `euler` mode); curriculum must be rerun |
 Project split into **two repositories**: `jetfighter-sim` (package `jetsim`, phases 0 to 6) and `jetfighter-rl` (package `jetfighter_rl`, phases 7+) | The simulator is reusable on its own; the RL repository depends on it as a library (pinned git dependency) | Aircraft data shipped inside the package (`jetsim/data/`); `jetfighter.rl` becomes `jetfighter_rl.training` |

***

## Overview

| Phase | Content | Key deliverable | 
|---|---|---|
| 0 | Scoping, tooling, repository structure | Initialized repository, test CI |
| 1 | Physics foundations (frames, atmosphere, integrator) | Tested `core/` |
| 2 | Point mass 3-DOF model | "Fast" aircraft for RL prototyping |
| 3 | Rigid body 6-DOF model with control surfaces | "Realistic" aircraft flown through its controls |
| 4 | Instruments, sensors, flight envelope | Dashboard + episode termination conditions |
| 5 | Visualization and manual flight | Plots, Tacview export, keyboard flying |
| 6 | Classical controllers (baseline) | PID autopilot |
| 7 | Gymnasium environment | `JetEnv` passing `check_env`, hierarchical and low level modes |
| 8 | RL maneuver curriculum | Hierarchical agents, then low level: stabilization → turn → aerobatics |
| 9 | Missile model (generic) | Proportional navigation missile |
| 10 | Missile evasion RL | Evasion agent + survival map |
| 11 | Extensions | Robustness, multiple threats, JSBSim, self play |


***

## Phase 0: Scoping and Tooling

**Goal**: a clean repository from the start.

*  Python 3.10+ environment (`uv` or `venv`) and `pyproject.toml`
*  Core dependencies: `numpy`, `scipy`, `matplotlib`, `pyyaml`, `pytest`
*  RL dependencies (later): `gymnasium`, `stable-baselines3`, `torch`, `tensorboard`
*  Code quality: `ruff` (lint + format), optional `mypy`, `pre-commit`
*  Git + `.gitignore` (exclude `runs/`, `models/`, `*.acmi`)
*  Conventions written in the README: **SI units everywhere**, radians internally, degrees for display only

***

## Phase 1: Physics Foundations

**Goal**: the mathematical building blocks everything else depends on. This is where the sneakiest bugs hide (signs, angle conventions).

### 1.1 Frames and conventions
*  **NED** (North East Down) inertial frame, flat Earth approximation (sufficient at this scale)
*  **Body** frame: x toward the nose, y along the right wing, z pointing down
*  **Aerodynamic / wind** frame (via angle of attack α and sideslip β)
*  Attitude stored internally as a **quaternion** (avoids gimbal lock in aerobatics), converted to Euler angles (φ roll, θ pitch, ψ heading) for display and observations
*  Quaternion normalized at every step

### 1.2 Standard atmosphere (ISA)
*  Temperature, pressure, density ρ(h), speed of sound a(h) from 0 to 20 km (troposphere + lower stratosphere)
*  Mach number and dynamic pressure q̄ = ½ρV²

### 1.3 Integrator
*  Fixed step RK4, **physics dt = 0.01 s (100 Hz)**
*  Physics rate decoupled from the agent decision rate (e.g. agent at 10 to 20 Hz → *frame skip* of 5 to 10)

### Validation tests
*  Rotations: Euler ↔ quaternion ↔ matrix round trip (tolerance 10⁻⁹)
*  ISA: compared against tabulated values (ρ₀ = 1.225 kg/m³, a₀ = 340.3 m/s, ρ(11 km) ≈ 0.364 kg/m³)
*  Integrator: free fall and harmonic oscillator against the analytical solution

***

## Phase 2: Point Mass 3-DOF Model (Fast Prototype)

**Goal**: a simple, very fast model to prototype environments and RL while the 6-DOF model is being built. It will also serve for large scale missile evasion campaigns.

**State**: position (x, y, h), speed V, flight path angle γ, heading χ, mass m
**Controls**: thrust T (or throttle), angle of attack α (or load factor n), bank angle μ

**Equations**:

```
V̇  = (T·cos α − D) / m − g·sin γ
γ̇  = [(L + T·sin α)·cos μ − m·g·cos γ] / (m·V)
χ̇  = (L + T·sin α)·sin μ / (m·V·cos γ)
ẋ  = V·cos γ·cos χ     ẏ = V·cos γ·sin χ     ḣ = V·sin γ
L  = q̄·S·CL(α, M)      D = q̄·S·(CD0(M) + k·CL²)
```

*  Implementation + limits: α_max (stall), n ∈ [−3 g, +9 g], rate limited μ
*  Drag: parabolic polar CD = CD0(M) + k·CL², with a CD0 rise in the transonic regime
*  Tests: steady level flight, constant load factor turn (turn rate ω = g·√(n²−1)/V), climb, ceiling

**Done** (`src/jetsim/aircraft/`, validation: `scripts/phase2_performance.py`)
* **Singularity free** formulation: the equations above become singular at γ = ±90°. The model therefore integrates a wind frame quaternion (Euler angles = χ, γ, μ), which makes loops possible.
* Chosen controls: `[throttle, commanded α, roll rate about the velocity vector]`, with a first order response (α: 0.25 s; roll: 0.2 s; engine: 1 s), plus fly by wire style **angle of attack and load factor limiters**.
* Propulsion: thrust ∝ (ρ/ρ0)^1.15 (fitted to the Stevens & Lewis static thrust values), linear ram effect with Mach, capped at 1.3 × sea level static thrust.
* Modules: `params.py` (YAML → dataclasses), `aero_polar.py`, `propulsion.py`, `dynamics_3dof.py` (dynamics, trim, instruments), `performance.py` (flight envelope, turn, ceiling).
* Performance vs published F-16 figures: max Mach 1.18 at sea level (≈ 1.2) and 2.08 at 11 km (≈ 2.0); climb 275 m/s (≈ 250); sustained turn 18.3 °/s at 3 km M0.8 (≈ 18 to 20); ≈ 10,600 physics steps/s.
* Known limitations: no stall, hence an optimistic minimum speed (≈ 45 m/s at sea level vs ≈ 60 to 65 m/s in reality); aerodynamic ceiling of 17.7 km vs 15.2 km published (an operational limit); constant mass.

***

## Phase 3: Rigid Body 6-DOF Model

**Goal**: the real "aircraft", flown through its **flight controls** rather than abstract setpoints.

### 3.1 "F-16 like" parameters
Unlike the F-22, the F-16 has **published aerodynamic data** (NASA wind tunnel tests, reproduced by Stevens & Lewis): this is what makes it a sensible choice. Chosen approach:
* **Mass, geometry, thrust** (approximate public values, sourced in the YAML)
  * Reference mass of the Stevens & Lewis model: 1/m = 1.57·10⁻³ slug⁻¹, i.e. ≈ 9,295 kg (≈ 20,490 lb) (F-16C: ≈ 8,600 kg empty, ≈ 19,200 kg max takeoff weight)
  * Span 9.14 m (30 ft), length ≈ 15.0 m, wing area 27.87 m² (300 ft²), mean chord 3.45 m (11.32 ft)
  * 1 engine (F100-PW-229 or F110-GE-129): ≈ 76 to 79 kN dry, ≈ 129 to 131 kN with afterburner; the Stevens & Lewis model uses the thrust tables of an older F100
  * No thrust vectoring
  * Load factor +9 g / −3 g; max Mach ≈ 2.0; ceiling ≈ 15,000 m
* **Aero coefficients**: published F-16 tables (Stevens & Lewis; Nguyen et al., NASA TP-1538), used directly, without scaling
* **Inertias** (Stevens & Lewis): Ixx = 9,496, Iyy = 55,814, Izz = 63,100, Ixz = 982 slug·ft² → ≈ 12,875 / 75,674 / 85,552 / 1,331 kg·m²
*  Everything lives in `src/jetsim/data/aircraft/f16.yaml` → the aircraft can be swapped without touching the code

### 3.2 State (13 variables + engine)
* NED position (3), body velocity u, v, w (3), quaternion (4), angular rates p, q, r (3)
* \+ engine state (spool / effective thrust), + mass (fuel burn, optional)

### 3.3 Pilot controls (inputs)
| Control | Main effect | Typical range |
|---|---|---|
| Throttle δ_T (+ afterburner) | Thrust | 0 → 1 (afterburner above a threshold) |
| All moving stabilators δ_e | Pitch (Cm) | ±25° |
| Ailerons / flaperons δ_a | Roll (Cl) | ±20° |
| Rudder δ_r | Yaw (Cn) | ±30° |

### 3.4 Forces and moments
*  Aero: CL(α, δe, M), CD(α, M), CY(β, δr), Cl(β, δa, δr, p, r), Cm(α, δe, q), Cn(β, δa, δr, p, r)
  * Interpolation tables in α and Mach (`scipy.interpolate.RegularGridInterpolator`)
  * Damping derivatives (Cmq, Clp, Cnr): essential, otherwise the aircraft oscillates forever
*  Propulsion: T(h, M, δ_T) with altitude lapse (roughly ∝ ρ/ρ₀), first order engine dynamics (time constant ~1 s, slower for afterburner light up)
*  Gravity expressed in the NED frame and projected onto body axes
*  Newton Euler equations (body forces + ω × V term; moments with an inertia tensor including Ixz)

### 3.5 Actuators
*  Position **and** rate saturation (e.g. 60°/s for the control surfaces)
*  First order lag (τ ≈ 0.05 s)
* → Prevents the RL agent from learning physically impossible "bang bang" commands

### Validation tests (in `notebooks/`)
*  **Trim**: find (α, δe, δ_T) for level flight at several speeds and altitudes (`scipy.optimize`)
*  **Eigenmodes**: linearize around trim, check for a fast angle of attack oscillation (*short period*, ~1 to 3 s) and a slow phugoid (~30 to 100 s)
*  **Step responses**: elevator, aileron and rudder steps, with consistent signs and magnitudes
*  **Conservation** without aero, thrust or gravity: rotational energy and angular momentum (engine included) are conserved
*  **3-DOF vs 6-DOF consistency** in a steady turn (verified in phase 6 with the autopilot: turn rates at 4 and 5 g match within 5 %)

> ⚠️ Caution: a real F-16 is **unstable by design** and flown through fly by wire. For RL, there are two options: (a) make the model slightly statically stable (easier), (b) keep the instability and add a stabilizing control law (Phase 6). Start with (a).

**Done** (`dynamics_6dof.py`, `f16_aero.py`, `analysis.py`, tables in `src/jetsim/data/aircraft/f16_sl_tables.yaml`, validation: `scripts/phase3_validation.py`)
* Complete Stevens & Lewis aero and engine tables (NASA TP-1538 data), reimplemented (the AeroBench reference code is GPL-3 licensed: only the numerical values were reused).
* Checks: derivatives match the reference implementation within 2·10⁻⁴ over 300 random states (the gap comes from rounded inertia constants in the book); the published check case is reproduced on 11 of 13 derivatives (ṗ and ṙ differ by ≈ 0.001 on Cl in every available implementation); the published trim is reproduced (throttle 0.1385, δe −0.759°).
* **CG position**: at the nominal x_cg = 0.35, the aircraft is unstable in pitch (time to double ≈ 7 s), like the real F-16. Option (a) was chosen: x_cg = 0.30 by default → stable across the whole subsonic envelope (except for a slow divergence on the back side of the power curve at α > 11°). 0.35 remains available for phase 6.
* Modes at 3,000 m / 200 m/s: short period ω = 2.1 rad/s ζ = 0.56; phugoid 107 s; dutch roll 1.8 s ζ = 0.12; roll τ = 0.28 s; stable spiral.
* Limitations: tables have no Mach effects (reliable up to ≈ Mach 0.6 to 0.8); ≈ 1,700 to 2,000 physics steps/s in pure Python (to be sped up in phase 7); 3-DOF/6-DOF consistency checked on the level flight trim angle of attack (< 1°), not yet in turns.

***

## Phase 4: Instruments, Sensors and Flight Envelope

### 4.1 Instruments (observable outputs)
*  Speeds: TAS, CAS (approx.), Mach, vertical speed
*  Altitude, heading, attitude (φ, θ, ψ)
*  Angle of attack α, sideslip β
*  Angular rates p, q, r
*  Body accelerations, **load factor Nz**
*  Specific energy Es = h + V²/2g and specific excess power Ps (very useful for missile evasion)
*  Option: sensor noise and bias (for robustness, Phase 11)

### 4.2 Flight envelope and episode termination
*  Ground collision (h < 0 or h < a safety h_min)
*  Load factor exceedance (structural limit)
*  Sustained stall / spin (α > α_max for more than N s)
*  Leaving the envelope (altitude > ceiling, speed < V_min)
*  NaN / numerical divergence detection → episode ends + log

**Done** (`instruments.py`, `sensors.py`, `envelope.py`, limits in the `envelope` section of `src/jetsim/data/aircraft/f16.yaml`, demo: `scripts/phase4_dashboard.py`)
* **Shared dashboard** for the 3-DOF and 6-DOF models (`read_instruments`): 28 quantities (position, TAS/CAS/EAS, Mach, vertical speed, γ/χ, lift vector bank angle μ (added in phase 6), attitude φ/θ/ψ, α/β, p/q/r, load factors nx/ny/nz, energy E and Ps, thrust). For the 3-DOF model, body attitude and angular rates are reconstructed from the wind frame and α.
* Exact **CAS** (isentropic impact pressure in subsonic flow, Rayleigh formula in supersonic flow) and its inverse.
* **Sensors**: white noise + a bias drawn at each episode, per channel, reproducible (random generator supplied); perfect by default; a realistic example lives in `src/jetsim/data/sensors/realistic.yaml`.
* **Episode termination**: NaN, ground, overload (−4 / +10 g), ceiling (17 km), minimum speed (50 m/s), max Mach (2.0 in 3-DOF, 0.95 in 6-DOF), sideslip (30°), sustained stall (α > 30° for more than 2 s).

***

## Phase 5: Visualization and Manual Flight

*  matplotlib plots: 3D trajectory, time series (V, h, α, Nz, controls)
*  **Tacview export (.acmi)**: simple text format, free viewer, very telling 3D rendering to analyze maneuvers and missile engagements
*  `scripts/fly_manual.py`: keyboard or joystick flying (`pygame`). This is the best sanity check: if a human cannot fly the model, neither can an RL agent
*  Replay of recorded episodes (states + actions in `.npz`)

**Done** (`src/jetsim/viz/`: `recorder.py`, `plots.py`, `tacview.py`, `manual.py`; scripts `demo_flight.py` and `fly_manual.py`)
* **Recorder**: time, states, controls, 27 instruments and events → `.npz`; **deterministic replay** (states reproduced bit for bit); the model is rebuilt from the recording.
* **Plots**: time series dashboard (8 charts) and 3D trajectory with ground track.
* **Tacview**: ACMI 2.2 export (position, attitude, TAS/CAS/Mach/AOA/AOS/Throttle, events), with a writer that handles multiple objects, ready for the missile (phase 9). Flat world placed over the Atlantic (46° N, 6° W).
* **Manual flight** (pygame, `[viz]` extra): artificial horizon + instruments, spring centered keyboard stick (AZERTY/QWERTY), trim, automatically detected joystick (not tested on real hardware), 3-DOF or 6-DOF, automatic save (.npz, .acmi, .png).
* `demo_flight.py` demo: a loop followed by a roll, chained by a sequencer that watches the instruments. In 6-DOF, tuning these maneuvers "by hand" is tricky: a direct motivation for phases 6 to 8.

***

## Phase 6: Classical Controllers (Baseline)

**Goal**: a reference to judge the RL agent against, and an optional lower layer for hierarchical RL.

*  Pitch / roll / yaw dampers (*stability augmentation*)
*  PID: speed hold, altitude hold, heading hold, coordinated turn (β → 0)
*  Option: LQR on the model linearized around trim
*  Scripted missile evasion heuristics (for Phase 10): *beam*, *drag*, last second break turn. The geometric primitives are ready; the full logic waits for the missile (phase 9)

**Architecture decision** (made on Sep 27, 2026, see the decision log):

| Option | The agent commands… | Pros | Cons |
|---|---|---|---|
| Low level | the control surfaces and throttle directly | faithful to the original goal | slow, unstable training |
| Hierarchical | setpoints (Nz, roll rate, thrust) tracked by a controller | much faster training | depends on the controller quality |

→ **Decision: hierarchical first, then low level.** The agent first flies through `HighLevelCommand` (n_z, roll rate, throttle), executed by the fly by wire system; once the tasks are mastered, we switch back to direct control surface commands (step 8.6, then phase 10).

**Done** (`src/jetsim/control/`: `pid.py`, `fbw.py`, `autopilot.py`, `lqr.py`, `maneuvers.py`; demo: `scripts/phase6_autopilot.py`)
* **Common high level interface** `HighLevelCommand(nz, roll_rate, throttle)`, executed by an inner loop specific to each model (`make_inner_loop`): this is the action space of the **hierarchical** RL option.
* **6-DOF fly by wire**: PI on n_z + q damping, PI on roll rate, β → 0 + yaw damper (high pass filter), load factor (−3/+9 g) and angle of attack (≤ 25°) limiters, gains scheduled on q̄. Gains were tuned by systematic search in simulation: 1 → 4 g step in 0.4 to 0.9 s; 90°/s roll in 0.2 to 0.3 s with |β| < 1°. **It stabilizes the aircraft at unstable CG positions** (0.35, 0.40): option (b) from phase 3 is now available.
* **Autopilot** (same gains for 3-DOF and 6-DOF): altitude → vertical speed → flight path angle → n_z (+ 1/cos μ in turns), heading → bank μ → roll rate, speed → throttle. Both models fly the same mission almost identically (figure `outputs/phase6_mission.png`).
* Longitudinal **LQR** on the linearized model: stabilizes x_cg = 0.35 and 0.40 around the design point (the free aircraft diverges).
* **Evasion primitives**: bearings, *beam* heading, *drag* heading, *break turn* toward the threat.
* Lesson learned: the bank angle to control is that of the **lift vector** (μ), not the fuselage roll angle (φ); the difference is negligible in cruise but makes tight turns fail (`bank` was added to the dashboard).
* Both options are technically feasible; the decision is: **hierarchical first, low level second** (see above).

***

## Useful References

* Stevens, Lewis & Johnson, *Aircraft Control and Simulation* (complete F-16 model, 6-DOF equations)
* Nguyen et al. (1979), NASA TP-1538: F-16 wind tunnel aerodynamic data
* Zarchan, *Tactical and Strategic Missile Guidance* (proportional navigation)
* [Gymnasium](https://gymnasium.farama.org/) and [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) documentation
* [JSBSim](https://github.com/JSBSim-Team/jsbsim): open source flight dynamics engine
* [Tacview](https://www.tacview.net/) ACMI format for visualization
* DARPA AlphaDogfight Trials (2020): background on RL in simulated air combat
