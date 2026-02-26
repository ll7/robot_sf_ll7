# Issue #454 Execution Note (Kinematics Contract + Parity)

## Scope Landed
- Runtime scenario `robot_config` now controls benchmark robot kinematics (`differential_drive`, `bicycle_drive`, `holonomic`).
- Holonomic robot support added with dual command modes:
  - `vx_vy`
  - `unicycle_vw`
- Benchmark artifacts now include `algorithm_metadata.kinematics_feasibility` diagnostics.
- Camera-ready campaign supports `kinematics_matrix` execution and emits parity + skipped-combination tables.

## Interpretation Caveat
- Kinematics parity should be interpreted jointly with:
  - `planner_kinematics.execution_mode` (native vs adapter),
  - `planner_kinematics.robot_command_space`,
  - `kinematics_feasibility` intervention rates.

High projection/infeasible rates indicate that constraints or adapter routing, not planner intent
alone, may drive metric deltas.
