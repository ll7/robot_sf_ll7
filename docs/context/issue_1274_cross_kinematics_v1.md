# Issue #1274 General Cross-Kinematics Parity Sweep

Issue: [#1274](https://github.com/ll7/robot_sf_ll7/issues/1274)

Status date: 2026-05-16

## Goal

Add a non-paper-facing `cross_kinematics_v1` benchmark profile for checking planner behavior across
the supported Robot SF kinematics modes under a fixed scenario and seed. This profile is a parity
and compatibility smoke surface, not a planner leaderboard.

## Added Surface

- `configs/benchmarks/cross_kinematics_v1.yaml`
- `paper_facing: false`
- `kinematics_matrix`: `differential_drive`, `bicycle_drive`, `holonomic`
- `holonomic_command_mode: vx_vy`
- scenario surface: `configs/scenarios/sets/cross_kinematics_v1.yaml`
- compatibility manifest: `configs/benchmarks/cross_kinematics_v1_compatibility.yaml`

The first scenario surface is intentionally small: `classic_cross_trap_low` with seed `111`. That
keeps local smoke runs cheap while exercising the same planner/scenario/seed slice across the three
kinematics modes.

## Compatibility Boundary

The first supported planner rows are `goal`, `social_force`, and `orca` across all three kinematics
modes. Deferred learned or adapter-sensitive planners are recorded in the compatibility manifest as
`degraded`, and placeholder adapter rows such as `rvo` and `dwa` are recorded as `unsupported`.

`degraded` and `unsupported` rows are not successful benchmark evidence. They are explicit
compatibility caveats so parity reports do not silently mix incompatible planner/motion-model pairs.

## Validation Commands

Config and manifest tests:

```bash
uv run pytest \
  tests/benchmark/test_camera_ready_campaign.py::test_load_cross_kinematics_v1_campaign_config \
  tests/benchmark/test_camera_ready_campaign.py::test_cross_kinematics_v1_compatibility_manifest -q
```

Preflight:

```bash
uv run python scripts/tools/run_camera_ready_benchmark.py \
  --config configs/benchmarks/cross_kinematics_v1.yaml \
  --mode preflight \
  --output-root output/benchmarks/issue_1274 \
  --campaign-id cross_kinematics_v1_preflight
```

Optional smoke run:

```bash
uv run python scripts/tools/run_camera_ready_benchmark.py \
  --config configs/benchmarks/cross_kinematics_v1.yaml \
  --mode run \
  --output-root output/benchmarks/issue_1274 \
  --campaign-id cross_kinematics_v1_smoke
```

## Interpretation Boundary

Use this profile to inspect cross-kinematics compatibility and parity grouping. Do not use it as a
paper-facing result table, full robustness claim, or replacement for stress/adversarial benchmark
surfaces.
