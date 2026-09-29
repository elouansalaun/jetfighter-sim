# Notebooks: a guided tour of jetfighter-sim

One notebook per phase of the simulator. Each one explains the choices made, gives the key
equations, and runs small examples on the real `jetsim` code, with plots. They are meant to
be read in order, but each stands on its own.

| # | Notebook | Content |
|---|---|---|
| 0 | [00_project_overview](00_project_overview.ipynb) | Goal, architecture, design decisions, conventions, validation at a glance |
| 1 | [01_physics_foundations](01_physics_foundations.ipynb) | Reference frames, quaternions vs gimbal lock, ISA atmosphere, RK4 |
| 2 | [02_point_mass_3dof](02_point_mass_3dof.ipynb) | Point-mass F-16: drag polar, thrust, trim, flight envelope, turn performance, a loop |
| 3 | [03_rigid_body_6dof](03_rigid_body_6dof.ipynb) | Rigid-body F-16 with NASA tables: stability, CG, actuators, natural modes, control responses |
| 4 | [04_instruments_envelope](04_instruments_envelope.ipynb) | Instruments, TAS/CAS/EAS, energy-maneuverability diagram, noisy sensors, envelope monitor |
| 5 | [05_visualization](05_visualization.ipynb) | Flight recording, bit-exact replay, plots, Tacview export, manual flight |
| 6 | [06_classical_control](06_classical_control.ipynb) | PID, fly-by-wire, autopilot, LQR, evasion primitives |



## Running them

```bash
uv venv && uv pip install -e ".[dev,notebooks]"
uv run jupyter lab notebooks/
```

