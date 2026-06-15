# Issue #2768 Learned Prediction Readiness Contract (2026-06-13)

Related issue: <https://github.com/ll7/robot_sf_ll7/issues/2768>

Related surfaces:

- Forecast baseline evaluation (PR #2741 and linked forecast-baseline issue)
- Predictive planner training: `docs/training/predictive_planner_training.md`
- Predictive planner baseline: `docs/baselines/prediction_planner.md`
- Predictive planner PR readiness: `docs/context/predictive_planner_pr_readiness_2026-02-20.md`
- Prediction planner probabilistic search: `docs/context/issue_591_prediction_planner_probabilistic_search.md`
- Gap prediction benchmark: `docs/context/issue_671_gap_prediction_benchmark.md`
- Predictive obstacle pipeline: `docs/context/issue_1167_predictive_obstacle_pipeline.md`
- Simulation trace export schema: `docs/context/issue_1689_simulation_trace_export_schema.md`
- Learned policy registry: `docs/context/policy_search/learned_policy_registry.md`
- Benchmark fallback policy: `docs/context/issue_691_benchmark_fallback_policy.md`

## Archetype Metadata

```yaml
archetype: preflight
evidence_tier: launch_packet
linked_policy:
  - docs/context/issue_1512_issue_archetypes.md
  - docs/context/artifact_evidence_vocabulary.md
  - docs/context/issue_691_benchmark_fallback_policy.md
```

## Status: NOT-TRAINING-READY

Learned pedestrian prediction training is **blocked** until every prerequisite in this contract
is satisfied. The validator at
`scripts/validation/validate_learned_prediction_readiness.py` fails closed when split metadata
or baseline evidence is missing.

## Prerequisites

Before selecting or unblocking learned-prediction work under this contract, run the lane routing guard:

```bash
rtk uv run python scripts/dev/validate_prediction_dependency_graph.py docs/context/prediction_lane_dependency_graph.json
```

and read `docs/context/prediction_lane_dependency_graph.json` to confirm dependency order and blockers.

Blocked learned-prediction candidates under this issue should not be executed until their listed
upstream nodes and gates are satisfied in the graph, including #2836, #2838, #2839, #2840,
\#2841, \#2843, and their derived readiness gates.


### 1. Trace Dataset Registry

A durable trace dataset registry must name every source that will feed learned prediction
training. Each entry must record:

- **source_id**: stable identifier for the trace source (for example `sim_classic_v1`,
  `carla_replay_v1`, `oracle_imitation_v1`).
- **source_type**: `simulation`, `replay`, `oracle_imitation`, `external`, or `mixed`.
- **episode_count**: number of episodes or traces available from this source.
- **actor_types**: list of dynamic actor types covered (for example `pedestrian`, `cyclist`,
  `vehicle`).
- **horizon_seconds**: maximum continuous trace horizon available per episode.
- **frame_rate_hz**: sampling rate of the trace data.
- **semantic_inputs**: which semantic fields are present per frame (for example
  `map_geometry`, `signal_state`, `intent_label`, `group_membership`).
- **provenance**: commit hash, config, seed, or external URL that reproduces the trace set.
- **license_or_access**: access boundary (local, W&B artifact, external, unknown).

No learned prediction training may begin until at least one source entry exists with
`episode_count > 0` and `actor_types` covering the target prediction family.

### 2. Train / Validation / Test Split Metadata

Every trace source used for training must declare an explicit, leakage-free split:

- **split_strategy**: `temporal`, `scenario_family`, `seed_hash`, `episode_partition`, or
  `heldout_map`. The strategy must be named and justified.
- **train_fraction**: fraction of episodes or trace-frames allocated to training.
- **validation_fraction**: fraction reserved for validation and early stopping.
- **test_fraction**: fraction reserved for held-out evaluation only.
- **leakage_prevention**: explicit statement of what constitutes leakage for this split
  (for example same seed, same scenario instance, same map geometry, same agent identity)
  and how the split prevents it.
- **split_manifest_path**: path to a committed YAML or JSON manifest that records which
  episode IDs or seed values belong to each split.
  `ForecastDataset.v1` manifests are valid when `split_policy.leakage_prevention` is
  explicitly populated as a list.

Splits must sum to 1.0. The test split must never be used for model selection, hyperparameter
tuning, or early stopping.

### 3. Target Horizon Definition

The prediction target horizon must be explicitly defined:

- **horizon_seconds**: primary prediction horizon in seconds.
- **horizon_steps**: number of discrete steps at the configured prediction dt.
- **dt_seconds**: time step between predicted positions.
- **multi_horizon**: whether multiple horizons are trained simultaneously. If true, list each
  horizon.

Machine-checkable readiness fields use this key form:

- horizon_seconds: blocked until a bounded Issue #2837 recommendation is selected.
- horizon_steps: blocked until a bounded Issue #2837 recommendation is selected.
- dt_seconds: blocked until a bounded Issue #2837 recommendation is selected.
- horizon_recommendation: blocked pending Issue #2837.
- timestep_recommendation: blocked pending Issue #2837.

The horizon definition must be compatible with the downstream planner's
`predictive_horizon_steps` and `predictive_rollout_dt` configuration.

### 4. Dynamic Actor Types

The learned predictor must declare which dynamic actor types it predicts:

- **supported_types**: list of actor types (for example `pedestrian`, `cyclist`,
  `personal_mobility_device`, `vehicle`).
- **type_encoding**: how actor type is encoded in the observation and target tensors.
- **max_agents**: maximum number of concurrent agents the predictor can handle per frame.
- **occlusion_handling**: whether the predictor handles occluded or partially observed actors,
  and how.

### 5. Semantic Input Contract

The predictor's semantic input surface must be documented:

- **required_fields**: minimum set of per-frame fields required at inference time.
- **optional_fields**: fields that improve prediction quality but are not required.
- **fallback_behavior**: what the predictor does when an optional field is missing
  (for example zero-fill, sentinel value, or fail closed).
- **map_geometry_source**: how static map geometry is provided to the predictor
  (for example occupancy grid, SVG line segments, topological graph).
- **signal_state_support**: whether traffic-light or signal-state semantics are included.

### 6. Calibration Metrics

The learned predictor must report at minimum the following calibration metrics on the
validation and test splits:

- **ADE**: Average Displacement Error over the prediction horizon.
- **FDE**: Final Displacement Error at the horizon endpoint.
- **minADE / minFDE**: minimum over K multimodal samples if multimodal.
- **MR**: Miss Rate, fraction of predictions where FDE exceeds a collision-relevant threshold.
- **ECE**: Expected Calibration Error if probabilistic or multimodal outputs are produced.

Metrics must be reported per actor type and per scenario family, not only as a global average.

### 7. Collision-Relevance Metrics

Prediction quality alone is not sufficient. The predictor must be evaluated on metrics that
relate to downstream collision avoidance:

- **collision_proxy_rate**: fraction of predictions where the predicted trajectory intersects
  the robot's planned path within a safety buffer.
- **ttc_error**: error in predicted time-to-collision compared to ground truth.
- **near_miss_preservation**: whether the predictor preserves the ordering of near-miss
  severity compared to ground truth traces.
- **planner_impact**: downstream planner success rate and collision rate when using the learned
  predictor versus a deterministic baseline, measured on the same test split.

### 8. Deterministic / Semantic Baselines

Before any learned predictor is trained, the following baselines must be evaluated on the
same test split with the same metrics:

- **constant_velocity**: extrapolate current velocity linearly over the prediction horizon.
- **constant_heading**: extrapolate current heading with constant speed.
- **zero_velocity**: predict the agent stays at its current position.
- **semantic_baseline**: if available, a rule-based or intent-conditioned semantic predictor
  (for example goal-directed straight-line to known destination).

The learned predictor must demonstrate statistically significant improvement over at least
the constant_velocity baseline on ADE, FDE, and one collision-relevance metric before it
is considered for planner integration.

### 9. Comparison Protocol

The comparison between learned predictor and baselines must follow:

- **same_test_split**: all predictors evaluated on the identical held-out test split.
- **same_seeds**: episode seeds and scenario configurations are identical across predictors.
- **same_horizon**: prediction horizon and dt are identical.
- **same_actor_filter**: the same set of actors is evaluated for all predictors.
- **statistical_test**: report confidence intervals or p-values for metric differences when
  the sample size supports them.
- **downstream_planner**: if comparing planner-level impact, use the same planner grid,
  same scenario matrix, and same evaluation surface.

## Training Block Conditions

Learned prediction training **must not** begin until:

1. At least one trace source exists in the registry with `episode_count > 0` and
   `actor_types` covering the target prediction family.
2. A committed split manifest exists with explicit `split_strategy`, leakage metadata, and
   no leakage across splits.
3. Baseline metrics exist for the required target (default: `constant_velocity`) with ADE/FDE on
   the test split.
4. Calibration report recommendation is `continue`.
5. Transferability report contains both oracle and deployable observation tiers.
6. Closed-loop coupling gate recommendation is `continue`.
7. Horizon and timestep recommendations are documented with `horizon_seconds`,
   `horizon_steps`, and `dt_seconds`.
8. The validator at `scripts/validation/validate_learned_prediction_readiness.py`
   exits with code 0 and all prerequisite statuses are `passed`.

If any condition is not met, the training block remains active and any attempt to train
a learned predictor should be classified as `blocked` or `not_benchmark_evidence`.

## Evidence Boundary

This document is a launch packet and readiness contract only. It does not claim:

- that any learned predictor has been trained,
- that any learned predictor outperforms baselines,
- that any benchmark or paper-facing claim is supported,
- that any neural architecture has been selected.

All such claims require durable, versioned evidence produced after this contract is satisfied
and training is unblocked.

## Validator Stub

The readiness validator at `scripts/validation/validate_learned_prediction_readiness.py`
checks the following fail-closed conditions:

- Trace registry file exists and contains at least one source entry with `episode_count > 0`.
- Split manifest file exists and contains valid train/validation/test fractions or dataset schema
  leakage metadata.
- Baseline evidence file exists and contains the named target metrics for ADE and FDE.
- Calibration report is present and recommends `continue`.
- Transferability report is present and includes oracle + deployable observation tiers.
- Closed-loop coupling gate is present and recommends `continue`.
- Horizon/recommendation fields are present in the readiness doc.

Each prerequisite reports pass/fail/blocked in the validator output.

If any check fails, the validator exits with code 2 and a structured error message.

## Validation

```bash
uv run python scripts/validation/validate_learned_prediction_readiness.py \
  --readiness-doc docs/context/issue_2768_learned_prediction_readiness.md
```

Expected result before prerequisites are met: exit code 2 with explicit missing-prerequisite
diagnostics.
