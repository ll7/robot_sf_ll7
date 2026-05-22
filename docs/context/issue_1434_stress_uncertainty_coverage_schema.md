# Issue #1434 Stress/Uncertainty Coverage Schema v1 (2026-05-22)

Related issue: <https://github.com/ll7/robot_sf_ll7/issues/1434>  
Implementation follow-up: <https://github.com/ll7/robot_sf_ll7/issues/1445>

## Goal

Define a conservative, versioned `stress_uncertainty_coverage.v1` schema contract for benchmark
reports and downstream parsers. The contract specifies which fields are required, optional, or
explicitly unavailable; when statistical summaries are mandatory versus advisory; how scenario and
failure-mode coverage axes are reported; and the fail-closed behavior consumers must apply when
fields are unknown or missing.

This note is **schema and interpretation only**. It does not change SNQI formulas, metric
computations, or normalization anchors. It does not make paper-facing statistical claims.

## Scope Boundary

- **In scope**: field cardinality, coverage axes, interpretation boundaries, consumer fail-closed
  rules, backward-compatibility test plan.
- **Out of scope**: new metric implementations, SNQI weight or baseline changes, scenario
  certification rules, camera-ready promotion logic.

## Schema Version

`stress_uncertainty_coverage.v1` is additive over existing `episode.schema.v1` and
`aggregate.schema.v1` surfaces. It does not replace them. A compliant report **must** declare:

```json
{
  "schema_version": "stress_uncertainty_coverage.v1",
  "schema_mode": "required|advisory|diagnostic"
}
```

- `required`: the consumer must reject the report if any required field is missing or malformed.
- `advisory`: missing fields downgrade the report to `advisory` status but do not block parsing.
- `diagnostic`: the block is for human review only; automated consumers may ignore it.

## Field Cardinality

### Required Fields

A `stress_uncertainty_coverage.v1` block must contain:

| Field | Type | Semantics |
|---|---|---|
| `schema_version` | string | const `stress_uncertainty_coverage.v1` |
| `report_id` | string | unique report identifier |
| `generated_at_utc` | string | ISO-8601 timestamp |
| `campaign_config_hash` | string | SHA-256 of the resolved campaign config |
| `scenario_matrix_hash` | string | SHA-256 of the resolved scenario matrix |
| `coverage_axes` | object | scenario-parameter and failure-mode coverage (see below) |
| `metric_groups` | object | at least one of `safety`, `comfort`, `efficiency` |
| `aggregate_mode` | string | `mean`, `median`, or `descriptive_only`; must match the statistical summary policy below |
| `availability_status` | string | one of `available`, `partial-failure`, `failed`, `not_available` per [issue #691](./issue_691_benchmark_fallback_policy.md) |

### Optional Fields

| Field | Type | Semantics |
|---|---|---|
| `bootstrap_ci` | object | `{samples, confidence, seed, lower, upper}`; required only when `aggregate_mode` demands CIs |
| `repeated_seed_interval` | object | `{seeds[], metric_min, metric_max}`; required only for seed-sensitivity reports per [issue #1271](./issue_1271_seed_sensitivity_explorer.md) |
| `quantile_summary` | object | `{q25, q50, q75, q90, q95}`; advisory unless the metric group explicitly requires it |
| `scenario_coverage_entropy` | number | normalized Shannon entropy over scenario feature tokens per [issue #1240](./issue_1240_scenario_coverage_entropy.md) |
| `hazard_coverage_summary` | object | typed `hazard_traceability.v1` coverage summary per [docs/hazard_traceability.md](../hazard_traceability.md) |
| `snqi_contract_status` | string | `pass`, `warn`, or `fail`; advisory at this schema level because SNQI is governed by its own contract |
| `fallback_degraded_rows` | array | list of planner rows that ran in `fallback` or `degraded` mode per [issue #691](./issue_691_benchmark_fallback_policy.md) |
| `missing_fields` | array | list of optional fields that were requested but unavailable; consumers must not treat missing optional fields as errors |

### Unavailable / Explicitly Absent Fields

If a field is unavailable, the report must set it to `null` or omit it, and must record it in
`missing_fields` when the omission is meaningful. Unavailable fields include:

- `bootstrap_ci` when bootstrap was disabled (`samples == 0`).
- `repeated_seed_interval` when the campaign used a single seed.
- `quantile_summary` when raw episode samples are absent or when the metric is binary
  (e.g., `success`).
- `scenario_coverage_entropy` when the scenario set was not curated with coverage metadata.
- `hazard_coverage_summary` when no `hazard_traceability.v1` mapping was loaded.
- Per-pedestrian force quantiles when `pedestrians == 0` (returns NaN by existing contract).

## Statistical Summary Requirements vs Advisory

The v1 schema distinguishes three evidence tiers. Consumers must fail closed when a tier's
requirements are not met.

### Tier 1: Required (Benchmark-Success Claims)

For any report that contributes to `availability_status=available` benchmark-success claims:

- `mean` aggregation is required.
- Bootstrap CIs are **required** for aggregate metric means that support resampling
  (`snqi_mean`, `collisions_mean`, `comfort_exposure_mean`, `time_to_goal_norm_mean`). The CI
  must use `bootstrap-samples > 0` with a declared seed. See
  [issue #1286](./issue_1286_snqi_bootstrap_stability.md).
- Outcome frequencies (`success`, `collision`) must be reported as fractions with the raw
  numerator/denominator exposed (e.g., `success_count / total_episodes`).
- `scenario_coverage_entropy` is **advisory**; absence does not block benchmark-success.
- `hazard_coverage_summary` is **advisory**; absence does not block benchmark-success.

### Tier 2: Advisory (Diagnostic / Exploratory Reports)

For reports marked `advisory` or `diagnostic`:

- Bootstrap CIs are **advisory**; missing CIs downgrade the report but do not invalidate it.
- `quantile_summary` is **advisory**; when present, it must include at least `q50` and `q95`.
- `repeated_seed_interval` is **required** only when the report is explicitly a seed-sensitivity
  surface per [issue #1271](./issue_1271_seed_sensitivity_explorer.md) and
  [issue #1294](./issue_1294_seed_sensitivity_perturbations.md); otherwise advisory.
- Descriptive-only summaries (e.g., raw episode tables without aggregation) are valid but must be
  tagged `aggregate_mode: descriptive_only`.

### Tier 3: Descriptive-Only (Raw Artifact Logs)

For raw JSONL episode logs and trace dumps:

- No aggregation or CI is required.
- `schema_mode` should be `diagnostic`.
- Coverage axes may be omitted entirely.

## Coverage Axes

### Scenario-Parameter Coverage Axis

`coverage_axes.scenario_parameters` is a dictionary keyed by parameter name. Each entry must
contain:

| Key | Required? | Description |
|---|---|---|
| `observed_values` | required | list of values that actually appeared in the campaign |
| `required_values` | required | list of values the scenario matrix claims to cover |
| `coverage_status` | required | `full`, `partial`, or `missing` |
| `entropy` | optional | normalized Shannon entropy when the parameter supports tokenization per [issue #1240](./issue_1240_scenario_coverage_entropy.md) |

Canonical parameter names reuse existing repo concepts and should be documented as
`field_name: description` entries:

- `archetype`: scenario family token such as `classic_interactions`, `nominal_v1`,
  or `station_platform`.
- `density_label`: density token such as `low`, `medium`, or `high`.
- `ped_density_bucket`: coarse density bucket per [issue #1240](issue_1240_scenario_coverage_entropy.md).
- `flow_type`: interaction token such as `bidirectional`, `crossing`, or `merging`.
- `map_name`: scenario map identifier.
- `kinematics_mode`: robot model token such as `differential_drive` or `holonomic`.
- `horizon_steps`: episode horizon bucket such as `100` or `500`.
- `observation_level`: observation contract token such as `oracle_full_state` or `lidar_2d`.
- `optional_stress_marker`: boolean stress-marker token per
  [issue #1240](issue_1240_scenario_coverage_entropy.md).
- `seed_count_bucket`: seed-count bucket for campaign breadth.

Interpretation boundary:

- `full` coverage does not prove scenario adequacy; it only proves the matrix executed the
  declared values.
- `partial` or `missing` coverage downgrades the report to `advisory` unless the missing values
  were explicitly excluded by the campaign config.

### Failure-Mode Coverage Axis

`coverage_axes.failure_modes` classifies which failure-mode classes were observed, unobserved, or
unavailable. The vocabulary reuses [issue #1056](./issue_1056_h500_failure_classification.md):

| Class | Required? | Interpretation |
|---|---|---|
| `time_budget_clean_relief` | optional | observed count and scenario list |
| `exposure_enabled_completion` | optional | observed count and scenario list |
| `safety_regressed_long_horizon` | optional | observed count and scenario list |
| `persistent_low_progress_timeout` | optional | observed count and scenario list |
| `scenario_contract_blocker` | optional | observed count and scenario list |
| `unsupported_wait_then_go_hypothesis` | optional | observed count and scenario list |
| `collision` | required | episode count and fraction |
| `near_miss` | required | episode count and fraction |
| `timeout_without_progress` | required | episode count and fraction |

Each class entry must contain:

- `observed_episodes` (integer)
- `observed_fraction` (number in `[0, 1]`)
- `scenario_ids` (array of strings, optional)
- `classification_source` (string, e.g., `manual`, `h500_classifier_v1`, `trace_review`)

Fail-closed rule:

- If `classification_source` is `unknown` or missing, the consumer must treat the class as
  `unavailable` and must not count it toward coverage totals.
- If `collision_event=true` in an episode but `collisions` metric is missing or `null`, the
  consumer must treat the metric block as degraded per [issue #1398](./issue_1398_metric_rollup_reconciliation.md).

## Metric Groups and Interpretation Boundaries

### Safety Metrics

| Metric | Required? | Unit | Interpretation Boundary |
|---|---|---|---|
| `collisions` | required | count or fraction | Exact collision flag from `outcome.collision_event` is the ground-truth source; sampled metrics must be floored to `> 0` when the flag is true. See [issue #1398](./issue_1398_metric_rollup_reconciliation.md). |
| `collision_rate` | required | fraction | `collisions / total_episodes`; binary episodes only. |
| `near_misses` | required | count | `d_coll <= min_distance < d_near`; zero near misses does not prove safety when force/comfort exposure is elevated. See [issue #1055](./issue_1055_exposure_aware_h500_tables.md). |
| `min_distance` | required | meters | Minimum robot-pedestrian clearance over the episode. |
| `force_exceed_events` | optional | count | Count of force-threshold exceedances; required when comfort exposure is reported. |
| `shield_intervention_rate` | optional | fraction | Diagnostic only; not a safety proof. See [issue #1247](./issue_1247_safety_shield_contract.md). |

Safety boundary rule:

- A report must not claim "safe" or "collision-free" on the basis of `near_misses=0` alone when
  `force_exceed_events` or `comfort_exposure` are non-zero or unavailable.
- Missing `collisions` when `collision_event=true` is a **degraded** row; the consumer must flag it
  and must not treat the row as benchmark-success.

### Comfort Metrics

| Metric | Required? | Unit | Interpretation Boundary |
|---|---|---|---|
| `comfort_exposure` | required | fraction | `force_exceed_events / (ped_count * effective_steps)`. Required for stress/uncertainty reports when force data is available. |
| `jerk_mean` | optional | m/s^3 | Mean robot jerk; zero on straight lines. |
| `curvature_mean` | optional | 1/m | Mean path curvature; excluded when `speed <= epsilon`. |
| `ped_force_q50` | optional | N | Median per-pedestrian force from the metrics suite. NaN when K=0. |
| `ped_force_q90` | optional | N | 90th percentile per-pedestrian force from the metrics suite. NaN when K=0. |
| `ped_force_q95` | optional | N | 95th percentile per-pedestrian force from the metrics suite. NaN when K=0. |

Comfort boundary rule:

- `comfort_exposure` increases monotonically with density in expectation, but a single campaign
  does not prove monotonicity. Do not claim density scaling without repeated-seed evidence.
- Missing `comfort_exposure` when force arrays are absent is **unavailable**, not zero.

### Efficiency Metrics

| Metric | Required? | Unit | Interpretation Boundary |
|---|---|---|---|
| `success` | required | fraction | Goal reached before horizon without collision. |
| `time_to_goal_norm` | required | fraction | `steps_to_goal / horizon` on success; `1.0` on failure or timeout. |
| `path_efficiency` | optional | fraction | `L_shortest / L_actual`, clipped `<= 1`. |
| `avg_speed` | optional | m/s | Diagnostic sanity check only; not part of SNQI. |

Efficiency boundary rule:

- `time_to_goal_norm` and `path_efficiency` are anti-correlated in expectation but not guaranteed
  to be monotonically related per episode.
- Timeout episodes must record `time_to_goal_norm = 1.0` or `null`; a consumer must reject reports
  that omit timeout semantics.

### SNQI Composite

| Field | Required? | Interpretation Boundary |
|---|---|---|
| `snqi_mean` | required for SNQI-bearing reports | Composite index using the campaign-declared weights and baseline. The v1 schema does not redefine the formula; see [issue #635](./issue_635_snqi_v3_paper_contract.md). |
| `snqi_contract_status` | advisory | `pass`, `warn`, or `fail`; consumers must not override the campaign-level SNQI diagnostics with this field. |
| `snqi_bootstrap_ci` | advisory | Bootstrap CI on `snqi_mean`; required only when the report mode requests CIs. See [issue #1286](./issue_1286_snqi_bootstrap_stability.md). |

SNQI boundary rule:

- This schema treats SNQI as an **input metric** carried forward from existing contracts. It does
  not modify weights, baselines, or normalization.
- Missing `snqi_mean` when the campaign config requests SNQI is a **degraded** row.

## Fail-Closed / Degraded Behavior for Unknown or Missing Fields

Consumers of `stress_uncertainty_coverage.v1` must implement the following fail-closed rules:

1. **Unknown `schema_version`**: reject the report with `not_available` and log the version string.
2. **Missing required field**: if `schema_mode == required`, reject with `failed` and list missing
   fields. If `schema_mode == advisory`, accept but downgrade `availability_status` to
   `partial-failure`.
3. **Unknown metric field inside `metric_groups`**: ignore unknown fields (forward compatibility).
   Do not crash. Log the unknown field names at `DEBUG` level.
4. **Missing `bootstrap_ci` when Tier-1 requires it**: downgrade the metric group to `advisory`;
   do not block other groups.
5. **Missing `collisions` with `collision_event=true`**: treat the episode/row as `degraded` and
   exclude it from benchmark-success claims.
6. **Non-success fallback metadata**: if `readiness_status` is `fallback` or `degraded`, or if
   `availability_status` is `partial-failure`, `failed`, or `not_available`, the consumer must not
   promote the report to benchmark-success, regardless of other field presence.
7. **Null or NaN metric values**: treat as `unavailable` for that metric; do not coerce to zero
   unless the existing metric contract explicitly requires it (e.g., `collisions=0` when no
   pedestrians are present).

These rules align with the canonical fail-closed benchmark fallback policy in
[issue #691](./issue_691_benchmark_fallback_policy.md).

## Backward Compatibility with Older Reports

### Parser / Report Test Plan

The following test plan verifies that a v1 consumer can ingest pre-v1 reports without crashing
and without misinterpreting missing fields as evidence.

#### Test 1: Pre-v1 aggregate summary ingestion

Input: an `aggregate.schema.v1` JSON that lacks `stress_uncertainty_coverage.v1` metadata
entirely (e.g., a legacy camera-ready campaign summary from before this schema).

Expected behavior:

- The consumer detects the missing `schema_version`.
- It treats the report as `aggregate.schema.v1` only.
- It does not fail; it simply does not provide stress/uncertainty/coverage annotations.
- `availability_status` defaults to `not_available` for the coverage block, but the underlying
  aggregate metrics remain readable.

#### Test 2: Partial v1 block ingestion

Input: a report that declares `schema_version: stress_uncertainty_coverage.v1` but omits
`coverage_axes.failure_modes` and `bootstrap_ci`.

Expected behavior:

- If `schema_mode: required`, the consumer rejects the report and lists the missing fields.
- If `schema_mode: advisory`, the consumer accepts the report, sets `availability_status` to
  `partial-failure`, and records the missing fields in `missing_fields`.

#### Test 3: Unknown field forward compatibility

Input: a report that includes a future metric `future_metric_xyz` inside `metric_groups.safety`.

Expected behavior:

- The consumer parses known fields.
- It ignores `future_metric_xyz` without crashing.
- It logs the unknown field name.

#### Test 4: Degraded collision-row handling

Input: an episode record where `outcome.collision_event == true` but `metrics.collisions` is
missing or `0`.

Expected behavior:

- The consumer flags the row as `degraded`.
- It excludes the row from benchmark-success aggregation.
- It emits a warning with `event="degraded_collision_row"`.

#### Test 5: Bootstrap CI presence validation

Input: a Tier-1 report where `bootstrap_ci` is `null` for a continuous metric.

Expected behavior:

- The consumer downgrades the metric to `advisory`.
- It does not block ingestion of the rest of the report.
- It emits a warning naming the metric and the missing CI.

### Regression Fixture Policy

If automated tests are added for this schema, they should use:

- A minimal valid v1 JSON fixture under `tests/fixtures/stress_uncertainty_coverage/`.
- A pre-v1 legacy fixture to verify the non-crash path.
- A malformed v1 fixture to verify the fail-closed rejection path.

No code or test changes are part of this docs-only issue.

## Related Surfaces

- Benchmark fallback policy: [issue #691](./issue_691_benchmark_fallback_policy.md)
- Scenario coverage entropy: [issue #1240](./issue_1240_scenario_coverage_entropy.md)
- SNQI bootstrap stability: [issue #1286](./issue_1286_snqi_bootstrap_stability.md)
- SNQI v3 paper-facing contract: [issue #635](./issue_635_snqi_v3_paper_contract.md)
- Metric rollup reconciliation: [issue #1398](./issue_1398_metric_rollup_reconciliation.md)
- Exposure-aware h500 tables: [issue #1055](./issue_1055_exposure_aware_h500_tables.md)
- H500 failure classification: [issue #1056](./issue_1056_h500_failure_classification.md)
- Seed-sensitivity explorer: [issue #1271](./issue_1271_seed_sensitivity_explorer.md)
- Seed-sensitivity perturbations: [issue #1294](./issue_1294_seed_sensitivity_perturbations.md)
- Safety shield contract: [issue #1247](./issue_1247_safety_shield_contract.md)
- Hazard traceability: [docs/hazard_traceability.md](../hazard_traceability.md)
- Metric definitions: [docs/dev/issues/social-navigation-benchmark/metrics_spec.md](../dev/issues/social-navigation-benchmark/metrics_spec.md)
- Episode schema: `robot_sf/benchmark/schemas/episode.schema.v1.json`
- Aggregate schema: `robot_sf/benchmark/schemas/aggregate.schema.v1.json`

## Follow-Up: Issue #1445

Issue #1445 tracks the runtime implementation of this schema:

- JSON Schema file under `robot_sf/benchmark/schemas/`
- Typed loader and validator
- Integration with `robot_sf_bench aggregate` and campaign report writers
- Parser regression fixtures for the five test cases above
- Documentation sync once the schema is machine-validated

This note remains the canonical interpretation contract until #1445 closes.

## Validation

Validation performed for this note:

- All linked issue numbers verified against existing `docs/context/` notes.
- All metric names verified against `docs/dev/issues/social-navigation-benchmark/metrics_spec.md`.
- All schema paths verified against `robot_sf/benchmark/schemas/`.
- Fail-closed rules cross-checked against [issue #691](./issue_691_benchmark_fallback_policy.md).
- Bootstrap and SNQI references cross-checked against [issue #1286](./issue_1286_snqi_bootstrap_stability.md) and [issue #635](./issue_635_snqi_v3_paper_contract.md).

Commands:

```bash
git diff --check
```

Expected: no whitespace errors in owned files.
