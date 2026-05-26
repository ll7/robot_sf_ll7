# Issue #1504 Ego-Conditioned Feature Contract

Related issues: [Issue #1504](https://github.com/ll7/robot_sf_ll7/issues/1504),
[Issue #1490](https://github.com/ll7/robot_sf_ll7/issues/1490),
[Issue #1427](https://github.com/ll7/robot_sf_ll7/issues/1427).

## Decision

The `predictive_ego_v1` 9D row shape is already wired through the predictive planner stack, but
the code paths do not currently populate channels [4:6] with one identical semantic payload.
This note documents the current implementation rather than inventing a cleaner contract than the
code actually provides.

Current producer behavior:

- The Issue #1504 same-seed pipeline uses `scripts/training/collect_predictive_hardcase_data.py`
  for both base and hard-seed collection, and that collector writes channels [4:6] from
  observation `robot.speed`.
- Runtime planner inference in `robot_sf/planner/socnav.py` also writes channels [4:6] from
  `robot.speed`.
- The standalone collector `scripts/training/collect_predictive_planner_data.py` uses
  `robot.velocity_xy` when present and falls back to `robot.speed`.

The issue-scoped contract therefore focuses on width, ordering, comparability, and current
producer behavior. It does not overclaim that every path already exposes world-frame velocity for
those two channels.

## Contract Sources

- Canonical contract YAML:
  `configs/training/predictive/predictive_ego_features_contract_v1.yaml`
- Schema constants:
  `robot_sf/planner/obstacle_features.py`
- Runtime model input:
  `robot_sf/planner/socnav.py` (`_build_model_input`)
- Data collection:
  `scripts/training/collect_predictive_planner_data.py` (`_frames_to_samples`)
  `scripts/training/collect_predictive_hardcase_data.py`

## Feature Layout (`predictive_ego_v1`, input_dim=9)

Per-agent feature vector in robot ego frame, shape `(max_agents, 9)`, dtype `float32`:

| Index | Name | Units | Description |
| --- | --- | --- | --- |
| 0 | `x_rel` | m | Pedestrian relative x in ego frame |
| 1 | `y_rel` | m | Pedestrian relative y in ego frame |
| 2 | `vx_rel` | m/s | Pedestrian velocity x in ego frame |
| 3 | `vy_rel` | m/s | Pedestrian velocity y in ego frame |
| 4 | `motion_0` | producer-dependent | See producer notes below |
| 5 | `motion_1` | producer-dependent | See producer notes below |
| 6 | `goal_dir_x` | unitless | Unit vector to goal, x in ego frame |
| 7 | `goal_dir_y` | unitless | Unit vector to goal, y in ego frame |
| 8 | `goal_dist` | m | Distance to current goal |

### Current producer notes for channels [4:6]

| Path | Channel 4 | Channel 5 | Why it matters |
| --- | --- | --- | --- |
| `scripts/training/collect_predictive_hardcase_data.py` | `robot.speed[0]` (`linear_speed`) | `robot.speed[1]` (`angular_speed`) | This is the collection path used by the committed Issue #1504 same-seed configs |
| `robot_sf/planner/socnav.py::_build_model_input` | `robot.speed[0]` (`linear_speed`) | `robot.speed[1]` (`angular_speed`) | This keeps training/runtime semantics aligned for the same-seed path |
| `scripts/training/collect_predictive_planner_data.py` | `robot.velocity_xy[0]` when present, else `robot.speed[0]` | `robot.velocity_xy[1]` when present, else `robot.speed[1]` | This standalone collector is not the same-seed training path and should not be used to reinterpret the committed same-seed configs |

### Normalization

Features are used **raw** by the GNN predictor. No min/max normalization is applied during
collection, training, or inference. This is distinct from the gym observation normalization
documented in `docs/dev/observation_contract.md`. The GNN operates on world-scale values
directly.

### Missing-Value Behavior

| Feature | Missing Condition | Behavior |
| --- | --- | --- |
| `motion_0`, `motion_1` | Source field absent or shorter than 2 elements | Missing elements zero-filled; standalone collector falls back from `robot.velocity_xy` to `robot.speed` |
| `goal_dir_x`, `goal_dir_y` | `goal_dist < 1e-6` | Direction set via `goal_rel / max(goal_dist, 1e-6)`; yields `(0, 0)` at-goal |
| `goal_dist` | Always available from `goal.current` and `robot.position` | Computed directly |
| Inactive agent slots | `ped_count < max_agents` | All 9 features set to `0.0`; `mask[i] = 0.0` |

### Frame Semantics

- Features [0:4] are in **robot ego frame** (rotated by robot heading).
- Features [4:6] are **producer-dependent today**. The same-seed collection/runtime path uses
  `robot.speed`; the standalone collector uses `robot.velocity_xy` when available.
- Features [6:8] are in **robot ego frame**.
- Feature [8] is a **scalar** (world-frame distance).

This mixed/pathed layout matches the current code. Pedestrian state is ego-relative, goal direction
is ego-relative, and the self-motion channels remain code-path-dependent. The committed same-seed
configs preserve comparability because both training collection and runtime inference use
`robot.speed` on those two channels.

## Compatibility with Issue #1427 Obstacle Features

Obstacle features are appended **after** the base feature vector:

| Variant | Base schema | Base dim | Obstacle dim | Total dim | Schema name |
| --- | --- | ---: | ---: | ---: | --- |
| Obstacle-only (Issue #1427) | `predictive_legacy_v1` | 4 | 6 | 10 | `predictive_obstacle_features_v1` |
| Ego+obstacle | `predictive_ego_v1` | 9 | 6 | 15 | `predictive_obstacle_features_v1` |

The schema name `predictive_obstacle_features_v1` is shared between both composed variants;
`base_schema` and `base_feature_dim` in the checkpoint metadata distinguish them.

Obstacle feature shape: `(max_agents, 6)` with indices `[distance, normal_x, normal_y, tangent_x, tangent_y, valid_mask]`.
Unavailable sentinel: `[50.0, 0.0, 0.0, 0.0, 0.0, 0.0]`.

## Four-Way Same-Seed Comparison Matrix

All four variants use the same committed seed manifest, scenario matrix, training budget, and
evaluation surface inherited from Issue #1427:

| # | Variant | Schema | Input dim | Config |
| --- | --- | --- | ---: | --- |
| 1 | Baseline | `predictive_legacy_v1` | 4 | `predictive_br07_same_seed_issue_1427.yaml` |
| 2 | Obstacle-only | `predictive_obstacle_features_v1` | 10 | `predictive_obstacle_features_same_seed_issue_1427.yaml` |
| 3 | Ego-only | `predictive_ego_v1` | 9 | `predictive_ego_features_same_seed_issue_1504.yaml` |
| 4 | Ego+obstacle | `predictive_obstacle_features_v1` | 15 | `predictive_ego_obstacle_features_same_seed_issue_1504.yaml` |

Rows 1-2 are from Issue #1427 and are referenced, not duplicated. Rows 3-4 are added by
Issue #1504. All four share:

- Base seed manifest: `configs/training/predictive/predictive_same_seed_issue_1427_base_seed_manifest.yaml`
- Hard seed manifest: `configs/benchmarks/predictive_hard_seeds_v1.yaml`
- Scenario matrix: `configs/scenarios/classic_interactions.yaml`
- Planner grid: `configs/benchmarks/predictive_sweep_planner_grid_v1.yaml`
- Base collection: `max_steps=200`, `horizon_steps=8`, `max_agents=24`
- Hard-seed collection: `max_steps=220`, `horizon_steps=8`, `max_agents=24`
- Training budget: 40 epochs, batch 128, lr 3e-4, hidden_dim 128, message_passing_steps 3
- Quality gates: `max_val_ade=1.2`, `max_val_fde=2.0`
- Evaluation surface: `horizon=120`, `dt=0.1`, `workers=1`, `campaign_workers=2`

## Metric Separation Policy

ADE/FDE forecast metrics and closed-loop navigation metrics must be kept separate:

- **Forecast metrics** (ADE, FDE, val_loss): diagnostic signals for predictive model quality.
- **Navigation metrics** (success rate, collision rate, near misses, min distance, runtime):
  evidence for downstream planner behavior.

Promotion requires **closed-loop navigation improvement**, not ADE/FDE gain alone. This is
enforced by the same-seed contract: a variant with better ADE/FDE but worse success/collision
must not be treated as superior.

## Non-Benchmark Evidence Categories

Per `docs/context/issue_691_benchmark_fallback_policy.md`, the following are non-benchmark
evidence and must be reported as caveats, not successes:

- Dry-runs and local-only `output/` files,
- `readiness_status=fallback` or `readiness_status=degraded`,
- `availability_status=not_available` or `availability_status=partial-failure`,
- Sentinel-only obstacle rows (all `valid_mask=0.0`),
- Launch packets,
- Proxy-eval artifact failures.

This note is a contract/reproducibility surface. It does **not** claim benchmark-strengthening
evidence for ego conditioning on its own, and it does **not** treat fallback or degraded execution
as a successful benchmark outcome.

## Validation

Schema validation checks (no training required for contract):

```bash
# YAML syntax check for contract manifest
uv run python -c "import yaml; yaml.safe_load(open('configs/training/predictive/predictive_ego_features_contract_v1.yaml'))" && echo "YAML parse OK"

# YAML syntax check for new same-seed configs
for cfg in configs/training/predictive/predictive_ego_features_same_seed_issue_1504.yaml \
           configs/training/predictive/predictive_ego_obstacle_features_same_seed_issue_1504.yaml; do
    uv run python -c "import yaml; yaml.safe_load(open('$cfg'))" && echo "$(basename $cfg): OK"
done

# Schema metadata consistency check
uv run python -c "
from robot_sf.planner.obstacle_features import (
    predictive_feature_schema_metadata,
    PREDICTIVE_LEGACY_FEATURE_SCHEMA,
    PREDICTIVE_EGO_FEATURE_SCHEMA,
    PREDICTIVE_OBSTACLE_FEATURE_SCHEMA,
)
# Baseline
m = predictive_feature_schema_metadata(model_family=PREDICTIVE_LEGACY_FEATURE_SCHEMA)
assert m['input_dim'] == 4, f'legacy input_dim={m[\"input_dim\"]}'
# Ego
m = predictive_feature_schema_metadata(model_family=PREDICTIVE_EGO_FEATURE_SCHEMA)
assert m['input_dim'] == 9, f'ego input_dim={m[\"input_dim\"]}'
# Obstacle on legacy
m = predictive_feature_schema_metadata(model_family=PREDICTIVE_OBSTACLE_FEATURE_SCHEMA, ego_conditioning=False)
assert m['input_dim'] == 10, f'obs+legacy input_dim={m[\"input_dim\"]}'
assert m['base_feature_dim'] == 4
# Obstacle on ego
m = predictive_feature_schema_metadata(model_family=PREDICTIVE_OBSTACLE_FEATURE_SCHEMA, ego_conditioning=True)
assert m['input_dim'] == 15, f'obs+ego input_dim={m[\"input_dim\"]}'
assert m['base_feature_dim'] == 9
assert m['base_schema'] == PREDICTIVE_EGO_FEATURE_SCHEMA
print('Schema metadata checks passed')
"

# Path existence for referenced files
for path in configs/training/predictive/predictive_same_seed_issue_1427_base_seed_manifest.yaml \
            configs/benchmarks/predictive_hard_seeds_v1.yaml \
            configs/scenarios/classic_interactions.yaml \
            configs/benchmarks/predictive_sweep_planner_grid_v1.yaml; do
    [ -f "$path" ] && echo "OK: $path" || echo "MISSING: $path"
done

# Diff/doc proof checks for the issue-scoped patch
git --no-pager diff --check origin/main...HEAD
BASE_REF=origin/main scripts/dev/check_docs_proof_consistency_diff.sh
```

## Follow-Up Boundary

Issue #1505 (preflight) should:
- Run obstacle-row preflights for the ego+obstacle config.
- Verify non-sentinel obstacle rows exist with `base_feature_dim=9`.
- Confirm collection shape contracts: `input_dim=9` for ego-only, `input_dim=15` for ego+obstacle.

Issue #1506 (training matrix) should:
- Submit bounded training through the worker/Slurm path using all four configs.

Issue #1507 (transfer analysis) should:
- Compare forecast-to-control transfer across the four variants.
- Keep ADE/FDE and navigation metrics separate in the report.

Issue [#1519](https://github.com/ll7/robot_sf_ll7/issues/1519) should:
- Decide whether slots [4:6] stay producer-specific or converge on one motion-channel source.
- Add producer metadata or tests before future work mixes standalone collection with the
  same-seed/runtime path.
