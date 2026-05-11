# Issue 1065 Route-Clearance Warning Audit

Date: 2026-05-09

Related issue: <https://github.com/ll7/robot_sf_ll7/issues/1065>

Follow-up repair/certification issue: <https://github.com/ll7/robot_sf_ll7/issues/1105>

Evidence bundle:
[evidence/issue_1065_route_clearance_audit_2026-05-09/](evidence/issue_1065_route_clearance_audit_2026-05-09/)

## Goal

Inspect current paper-matrix and h500 route-clearance warnings before using affected scenario
failures as planner-behavior evidence.

This audit is intentionally interpretation-only. It does not change planner behavior, benchmark
metrics, maps, or scenario configs.

## Source Commands

Fixed paper matrix:

```bash
rtk uv run python scripts/tools/run_camera_ready_benchmark.py \
  --config configs/benchmarks/paper_experiment_matrix_v1.yaml \
  --mode preflight \
  --label issue1065_paper \
  --log-level WARNING
```

H500 scenario-horizon surface:

```bash
rtk uv run python scripts/tools/run_camera_ready_benchmark.py \
  --config configs/benchmarks/paper_experiment_matrix_v1_scenario_horizons_h500.yaml \
  --mode preflight \
  --label issue1065_h500 \
  --log-level WARNING
```

Both commands wrote local, ignored artifacts under `output/benchmarks/camera_ready/`. The compact
table promoted for review is
[evidence/issue_1065_route_clearance_audit_2026-05-09/route_clearance_warning_classification.csv](evidence/issue_1065_route_clearance_audit_2026-05-09/route_clearance_warning_classification.csv).

Tracked cross-checks:

- [Issue #1023 Scenario-Horizon Preflight](evidence/issue_1023_scenario_horizons_preflight_2026-05-06/preflight/validate_config.json)
- [Issue #1023 Candidate-Augmented Preflight](evidence/issue_1023_candidate_augmented_preflight_2026-05-06/preflight/validate_config.json)
- [Camera-Ready All-Planners SLURM Check (2026-05-04)](camera_ready_all_planners_slurm_2026-05-04.md)
- [Issue #1057 Semantic Blocker Audit](issue_1057_semantic_blocker_audit.md)

## Current Warnings

Both current preflights report the same warning count and scenario set:

- `route_clearance_warning_count`: 18
- `warning_scope`: `scenario` for every row
- warning threshold: `0.5 m`

The scenarios are:

- `classic_cross_trap_high`
- `classic_cross_trap_low`
- `classic_cross_trap_medium`
- `classic_doorway_high`
- `classic_doorway_low`
- `classic_doorway_medium`
- `classic_merging_low`
- `classic_merging_medium`
- `classic_overtaking_low`
- `classic_overtaking_medium`
- `classic_station_platform_medium`
- `classic_t_intersection_low`
- `classic_t_intersection_medium`
- `francis2023_entering_elevator`
- `francis2023_entering_room`
- `francis2023_exiting_elevator`
- `francis2023_exiting_room`
- `francis2023_narrow_doorway`

## Classification

| Class | Scenarios | Interpretation | Planner-attribution boundary |
|---|---|---|---|
| `benchmark_blocking_geometry_issue` | `classic_merging_low`, `classic_merging_medium`, `classic_station_platform_medium` | Route centerline distance is `0.0 m`, yielding `-1.0 m` margin for a `1.0 m` robot. | Unsafe for planner attribution until repaired or explicitly certified/excluded. |
| `scenario_geometry_caveat` with tangent geometry | Francis elevator/room/narrow-doorway rows | Route centerline distance equals robot radius, yielding `0.0 m` margin. | Unsafe for positive-clearance claims; planner attribution requires an explicit tangent-geometry caveat. |
| `scenario_geometry_caveat` with low positive margin | cross-trap, doorway, overtaking, and T-intersection rows | Route is technically positive-clearance but below the `0.5 m` warning threshold. | Usable only with low-clearance caveat; static-clearance-sensitive failures need trace or certification evidence. |

No warning row should be called a harmless metadata warning today. The less severe rows may still
be useful as constrained scenarios, but they must not be silently treated as ordinary geometry.

## Attribution Rules

- Do not use `classic_merging_low`, `classic_merging_medium`, or
  `classic_station_platform_medium` failures as planner-mechanism evidence until #1105 repairs,
  excludes, or certifies the route geometry.
- Treat zero-margin Francis rows as stress/tangent-geometry cases; avoid any wording that a planner
  failed a positive-clearance route unless certification is updated.
- Low-positive-margin classic rows can remain in aggregate comparisons only with caveats. If a
  failure explanation depends on static clearance, require trace-backed evidence or defer to #1105.
- Continue applying the fallback/degraded policy from
  [Issue #691 Benchmark Fallback Policy](issue_691_benchmark_fallback_policy.md); route-clearance warnings do not make
  fallback or degraded execution successful benchmark evidence.

## Follow-Up Boundary

Issue #1105 tracks the actual map/route repair or certification decision. It should decide whether
each warning scenario is repaired, certified as intentional stress geometry, or excluded from
planner-attribution claims.

No planner-code issue is justified by this audit alone. Planner follow-ups should wait until the
geometry blockers are repaired or explicitly accepted as stress-only benchmark cases.

## Validation

Validation performed for this audit:

```bash
rtk uv run python scripts/tools/run_camera_ready_benchmark.py \
  --config configs/benchmarks/paper_experiment_matrix_v1.yaml \
  --mode preflight \
  --label issue1065_paper \
  --log-level WARNING
```

```bash
rtk uv run python scripts/tools/run_camera_ready_benchmark.py \
  --config configs/benchmarks/paper_experiment_matrix_v1_scenario_horizons_h500.yaml \
  --mode preflight \
  --label issue1065_h500 \
  --log-level WARNING
```

```bash
jq '{campaign_id, route_clearance_warning_count, scenarios: [.route_clearance_warnings[].scenario]}' \
  output/benchmarks/camera_ready/paper_experiment_matrix_v1_issue1065_paper_20260509_192843/preflight/validate_config.json
```

```bash
jq '{campaign_id, route_clearance_warning_count, scenarios: [.route_clearance_warnings[].scenario]}' \
  output/benchmarks/camera_ready/paper_experiment_matrix_v1_scenario_horizons_h500_issue1065_h500_20260509_192843/preflight/validate_config.json
```

Additional validation for the PR should include `rtk git diff --check` and the repository PR
readiness gate.
