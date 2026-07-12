# Issue #5355: Prediction-MPC 2x2 Factorial Preregistration

**Issue:** [#5355](https://github.com/ll7/robot_sf_ll7/issues/5355)
**Date:** 2026-07-12
**Base planner:** prediction-MPC framework (option 1, maintainer-confirmed)
**Status:** preregistration — no GPU submission until dependencies resolve

## Design

2x2 factorial on the prediction-MPC framework (`PredictionMPCPlannerAdapter`):

| Arm | Factor A (prediction) | Factor B (constraints + escape) | Algo config |
|-----|----------------------|--------------------------------|-------------|
| A0_B0 | OFF (`predictor_backend: none`) | OFF (`hard_pedestrian_constraints_enabled: false`) | `configs/algos/prediction_mpc_factorial_A0_B0.yaml` |
| A0_B1 | OFF | ON (`hard_pedestrian_constraints_enabled: true`, `local_min_escape_enabled: true`) | `configs/algos/prediction_mpc_factorial_A0_B1.yaml` |
| A1_B0 | ON (`predictor_backend: constant_velocity`) | OFF | `configs/algos/prediction_mpc_factorial_A1_B0.yaml` |
| A1_B1 | ON | ON | `configs/algos/prediction_mpc_factorial_A1_B1.yaml` |

All four arms share:
- Base optimizer: `NMPCSocialPlannerAdapter` (SLSQP)
- Observation contract: SocNav fields
- Action space: unicycle `(v, w)`
- Kinematics: differential drive
- Runtime budget: identical solver config
- Tuning budget: identical weights
- Scenario matrix: `configs/scenarios/classic_interactions_francis2023.yaml`
- Seed set: `eval` (seeds 111, 112, 113)

## Factor toggle mechanism

### Factor A: prediction consumption

- **ON:** `predictor_backend: constant_velocity` — the `ConstantVelocityPedestrianPredictor` computes horizon-step pedestrian futures from current positions and ego-frame velocities. These futures feed hard clearance constraints.
- **OFF:** `predictor_backend: none` — the `NullPedestrianPredictor` repeats the currently observed pedestrian positions across the horizon. It consumes no velocity or trajectory prediction, while preserving Factor B hard-clearance constraints.

### Factor B: constraint + local-minimum handling

- **ON:** `hard_pedestrian_constraints_enabled: true` — SLSQP `NonlinearConstraint` enforces predicted pedestrian clearance at each horizon step. `local_min_escape_enabled: true` — when speed is below `local_min_escape_speed_threshold` and goal distance exceeds `local_min_escape_distance`, forces a progress-seeking action toward the goal heading.
- **OFF:** `hard_pedestrian_constraints_enabled: false` — `_optimizer_constraints()` returns empty tuple; planner relies solely on soft cost in NMPC objective. `local_min_escape_enabled: false` — no progress escape.

## Primary endpoints

1. **collision_free_completion:** binary per-episode; robot reaches goal without collision within horizon.
2. **near_miss_exposure_normalized:** near-miss count normalized per interaction opportunity.
3. **censored_time_to_goal:** time-to-goal with right-censoring at episode timeout.

## Preregistered contrasts

1. **A main effect:** prediction ON vs OFF (averaged over constraint levels).
2. **B main effect:** constraints ON vs OFF (averaged over prediction levels).
3. **AB interaction:** does the effect of prediction depend on constraint level?

## Multiplicity policy

Holm-Bonferroni over the three preregistered contrasts, family-wise alpha = 0.05.

## Seed budget

- Seeds: `{111, 112, 113}` from `configs/benchmarks/seed_sets_v1.yaml` `eval` set.
- Each seed applied identically to all four arms (paired design).
- Pairing key: `(scenario_id, seed)`.

## Stop rules

- **Continue:** all arms have complete paired episodes for all scenarios and seeds.
- **Negative:** any arm fails to produce collision-free episodes in >50% of scenarios.
- **Blocked:** missing paired episodes, provenance mismatch, or unresolvable fallback execution.

## Dependencies

- [#5351](https://github.com/ll7/robot_sf_ll7/issues/5351): hierarchical paired analysis with multiplicity (open).
- [#5353](https://github.com/ll7/robot_sf_ll7/issues/5353): matched-capability fairness contract (open).

## Campaign config

`configs/research/prediction_mpc_factorial_v1.yaml`

## Configuration provenance

The durable registry pins the preregistration config path, SHA-256, and producing commit in
[`preregistration_config_registry.json`](evidence/issue_5355_prediction_mpc_factorial_preregistration/preregistration_config_registry.json).
It records a design contract only, not a completed campaign or benchmark evidence.

## Reproduction (CPU smoke only)

```bash
# Validate arm configs load correctly:
uv run python -c "
from robot_sf.planner.prediction_mpc import build_prediction_mpc_config, PredictionMPCPlannerAdapter
import yaml
for arm in ['A0_B0', 'A0_B1', 'A1_B0', 'A1_B1']:
    cfg = yaml.safe_load(open(f'configs/algos/prediction_mpc_factorial_{arm}.yaml'))
    config = build_prediction_mpc_config(cfg)
    adapter = PredictionMPCPlannerAdapter(config=config)
    print(f'{arm}: prediction={config.predictor_backend}, constraints={config.hard_pedestrian_constraints_enabled}, escape={config.local_min_escape_enabled}')
"
```
