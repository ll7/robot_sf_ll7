# Issue #1082 Paper Cross-Kinematics Parity Sweep

Issue: [#1082](https://github.com/ll7/robot_sf_ll7/issues/1082)

Status date: 2026-05-09

## Goal

Add a versioned benchmark surface that runs the same planner/scenario/seed slice across the
currently supported Robot SF robot kinematics so cross-kinematics interpretation is based on
campaign evidence instead of interface availability alone.

## Added Surface

The new paper-facing profile is:

- `configs/benchmarks/paper_cross_kinematics_v1.yaml`
- `paper_profile_version: paper-cross-kinematics-v1`
- `kinematics_matrix`: `differential_drive`, `bicycle_drive`, `holonomic`
- `holonomic_command_mode: vx_vy`
- scenario surface: `configs/scenarios/sets/paper_cross_kinematics_v1.yaml`
- compatibility manifest: `configs/benchmarks/paper_cross_kinematics_v1_compatibility.yaml`

The scenario surface is intentionally small: `classic_cross_trap_low` with seed `111`. This makes
the profile a smoke/parity surface, not a full paper benchmark replacement.

## Compatibility Manifest

The compatibility manifest records the first supported planner/kinematics pairs:

- `goal`
- `social_force`
- `orca`

Each planner has entries for `differential_drive`, `bicycle_drive`, and `holonomic`, with a
`supported` status and a reason. Deferred planners such as `ppo`, `prediction_planner`, `sacadrl`,
and `socnav_sampling` are recorded as `degraded` with reasons rather than being silently omitted.

This keeps the campaign surface explicit without forcing learned or adapter-sensitive planners into
motion models whose artifacts or provenance need a separate audit.

## Runner Contract

The existing camera-ready runner already writes the cross-kinematics artifacts needed by this
profile:

- `reports/kinematics_parity_table.csv`
- `reports/kinematics_parity_table.md`
- `reports/kinematics_skipped_combinations.csv`
- `reports/kinematics_skipped_combinations.md`

The paper-facing validation guard now remains profile-specific:

- `paper-matrix-v1` stays locked to `differential_drive`.
- `paper-cross-kinematics-v1` allows exactly the three-mode parity matrix.

## Validation Commands

Preflight:

```bash
uv run python scripts/tools/run_camera_ready_benchmark.py \
  --config configs/benchmarks/paper_cross_kinematics_v1.yaml \
  --mode preflight \
  --output-root output/benchmarks/issue_1082 \
  --campaign-id paper_cross_kinematics_v1_preflight
```

Smoke run:

```bash
uv run python scripts/tools/run_camera_ready_benchmark.py \
  --config configs/benchmarks/paper_cross_kinematics_v1.yaml \
  --mode run \
  --output-root output/benchmarks/issue_1082 \
  --campaign-id paper_cross_kinematics_v1_smoke
```

Expected interpretation:

- all configured supported pairs should produce rows in `kinematics_parity_table`,
- skipped or unavailable rows must be interpreted through the skip/failure reason,
- projection and infeasible rates must be reviewed before claiming planner-invariant behavior.

Observed local smoke result for this PR:

- campaign id: `paper_cross_kinematics_v1_smoke`
- total runs: `9`
- successful runs: `9`
- total episodes: `9`
- `benchmark_success: true`
- `kinematics_parity_table.csv`: 9 data rows plus header
- `kinematics_skipped_combinations.csv`: 0 rows

## Boundaries

This profile does not add new kinematics classes, does not promote deferred planners into
cross-kinematics evidence, and does not replace the frozen `paper-matrix-v1` campaign. It provides
an auditable parity smoke surface for the currently supported core rows.
