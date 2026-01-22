# Refactor: ped-robot force naming confusion

## Summary
The configuration flag `peds_have_obstacle_forces` is misleading: it only controls
pedestrian *obstacle* forces (static map obstacles), but it is often read as also
controlling pedestrian–robot repulsion. Ped-robot repulsion is configured separately
via `SimulationSettings.prf_config.is_active` and is enabled by default.

This naming ambiguity risks incorrect experiment setup and inconsistent benchmark
protocols.

## Current behavior (as implemented)
- `peds_have_obstacle_forces`:
  - **True** → includes `ObstacleForce` in pedestrian forces.
  - **False** → removes `ObstacleForce` from the force list.
- `SimulationSettings.prf_config.is_active`:
  - **True** → adds `PedRobotForce` (pedestrian repulsion from the robot).
  - **False** → disables ped-robot repulsion.

## Why this is a problem
- The flag name suggests a broader meaning than obstacle-only behavior.
- Documentation and factory docstrings currently imply ped-robot interaction may
  be tied to this flag, which is incorrect.
- Misconfiguration is likely in training and benchmark setups.

## Suggested refactor (future issue)
1) Introduce explicit flags with clear naming:
   - `peds_have_obstacle_forces` → rename to `peds_have_static_obstacle_forces` (or similar)
   - Add `peds_have_robot_repulsion` (or `ped_robot_repulsion_enabled`)
2) Deprecate old name with warnings and a compatibility layer.
3) Update docs and configs.
4) Add a migration note in `docs/dev_guide.md` and/or changelog.

## Impact on Issue 403 training
We should **not block** the training experiment on this rename. It risks
API churn and cross-branch conflicts. Instead:
- Document the current behavior in Issue 403 spec.
- Use explicit config values during training.
- Open a separate refactor issue for safe renaming after training kickoff.

## References
- `robot_sf/sim/simulator.py` (force assembly)
- `robot_sf/sim/sim_config.py` (`prf_config`)
- `robot_sf/ped_npc/ped_robot_force.py` (PedRobotForce implementation)
