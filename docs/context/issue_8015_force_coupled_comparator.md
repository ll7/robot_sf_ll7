# Force-Coupled Potential-Field Diagnostic Comparator Method Card (Issue #8015)

**Status:** experimental / diagnostic-comparator / smoke-only — implementation-integrity and local comparative evidence. Not a benchmark ranking, not a release-roster change, not social compliance certification.
**Issue:** [#8015](https://github.com/ll7/robot_sf_ll7/issues/8015) (parent research [#7882](https://github.com/ll7/robot_sf_ll7/issues/7882), prerequisite implementation [#7889](https://github.com/ll7/robot_sf_ll7/issues/7889)).
**Harness Module:** `robot_sf/benchmark/force_coupled_comparator.py`.
**CLI Entry Point:** `scripts/benchmark/check_force_coupled_comparator.py`.
**Receipt Schema:** `robot_sf/benchmark/schemas/force_coupled_comparator_receipt.v1.json`.
**Contract Tests:** `tests/benchmark/test_force_coupled_comparator.py`.

Plain-language summary: a deterministic diagnostic comparison harness that evaluates the opt-in `force_coupled_potential_field` local planner against reference baselines (`pure_pursuit_goal`, attractive-dominant ablation, and repulsive-dominant ablation) on canonical analytic scenarios (`analytic_static_obstacle`, `analytic_pedestrian_interaction`, `analytic_symmetric_obstacle`, `analytic_unobstructed`).

## Compared Planner Configurations

| Planner ID | Description | Role |
|---|---|---|
| `force_coupled_potential_field` | Canonical opt-in force-coupled potential-field local planner from #7889 | Candidate under test |
| `pure_pursuit_goal` | Unicycle pure pursuit straight to goal with bounded speeds and rate limits | Non-avoiding reference baseline |
| `ablation_attractive_dominant` | Force-coupled planner with near-zero repulsive weight (`repulsive_weight=0.001`) | Ablation baseline |
| `ablation_repulsive_dominant` | Force-coupled planner with near-zero attractive weight (`attractive_weight=0.001`) | Ablation baseline |

## Canonical Analytic Scenarios

| Scenario ID | Seed | Start $(x,y,\theta)$ | Goal $(x,y)$ | Obstacles | Pedestrians |
|---|---:|---|---|---|---|
| `analytic_static_obstacle` | 1 | `(0.0, 0.0, 0.0)` | `(4.0, 0.0)` | `[(1.0, 0.5)]` | `[]` |
| `analytic_pedestrian_interaction` | 7 | `(0.0, 0.0, 0.0)` | `(4.0, 0.0)` | `[]` | `[(1.0, 0.0)]` |
| `analytic_symmetric_obstacle` | 42 | `(0.0, 0.0, 0.0)` | `(4.0, 0.0)` | `[(2.0, 0.0)]` | `[]` |
| `analytic_unobstructed` | 1 | `(0.0, 0.0, 0.0)` | `(4.0, 0.0)` | `[]` | `[]` |

## Aggregate Summary Results

| Planner ID | Runs | Success Rate | Collision Rate | Near-Miss Rate | Mean Path (m) | Mean Jerk Metric | Mean Latency (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `force_coupled_potential_field` | 4 | 1.0000 | 0.0000 | 0.2500 | 6.0333 | 0.7217 | 0.1396 |
| `pure_pursuit_goal` | 4 | 0.5000 | 0.5000 | 0.0000 | 3.7501 | 5.2554 | 0.0122 |
| `ablation_attractive_dominant` | 4 | 0.5000 | 0.5000 | 0.0000 | 3.7651 | 0.8402 | 0.1400 |
| `ablation_repulsive_dominant` | 4 | 0.5000 | 0.0000 | 0.0000 | 8.9043 | 0.5993 | 0.1376 |

## Evidence Boundary

- Target: implementation integrity and local comparative diagnostic proof on canonical analytic scenarios only.
- Exclusions: no SLURM execution, no large-scale campaign, no simulator dataset promotion, no benchmark leaderboard ranking, no paper-facing claim.
