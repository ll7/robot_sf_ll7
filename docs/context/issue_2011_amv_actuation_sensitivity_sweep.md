# Issue #2011 AMV Actuation Sensitivity Sweep

Issue: [#2011](https://github.com/ll7/robot_sf_ll7/issues/2011)

Status: diagnostic pilot completed on 2026-06-01. This is not benchmark-strengthening evidence
because all executed rows are classified `accepted_unavailable_only`.

## Goal

Create a compact AMV actuation-envelope sensitivity sweep that separates:

- TRL-backed longitudinal acceleration/braking proxy values,
- synthetic latency stress,
- synthetic yaw-rate/angular-acceleration stress,
- synthetic update-rate stress.

The sweep must preserve the issue #2001 source boundary: longitudinal acceleration and braking can
use platform-class proxy values, while yaw rate, angular acceleration, latency, and update rate
remain synthetic stress factors.

## Implementation

The sweep manifest is
`configs/benchmarks/issue_2011_amv_actuation_sensitivity_sweep_v0.yaml`. It uses the issue #1556
AMV actuation stress slice as the base config, then materializes 12 concrete pilot configs:

- `longitudinal_proxy_low|nominal|high`,
- `latency_synthetic_low|medium|high`,
- `yaw_rate_synthetic_low|medium|high`,
- `update_rate_synthetic_low|medium|high`.

`scripts/tools/run_amv_actuation_sensitivity_sweep.py` materializes the generated configs, runs
camera-ready preflight or pilot mode, and aggregates effect-size rows by field group, level,
planner, and scenario family. The pilot uses two scenarios, one seed, and two planners:
`classic_cross_trap_high`, `francis2023_intersection_wait`, seed `111`, planners `goal` and
`social_force`.

The implementation also adds the missing `latency_stress_profile` pass-through from
`robot_sf.benchmark.runner.run_batch` to map-runner execution, and extends synthetic update-rate
stress with `2.5hz-hold` so the update-rate group has distinct low/nominal/high levels.

## Evidence

Compact evidence:

- `docs/context/evidence/issue_2011_amv_actuation_sensitivity_pilot_2026-06-01/`

Validation commands run:

```bash
scripts/dev/run_worktree_shared_venv.sh -- ruff check \
  robot_sf/benchmark/runner.py \
  robot_sf/benchmark/synthetic_actuation.py \
  scripts/tools/run_amv_actuation_sensitivity_sweep.py \
  tests/benchmark/test_runner_latency_stress_pass_through.py \
  tests/tools/test_run_amv_actuation_sensitivity_sweep.py

scripts/dev/run_worktree_shared_venv.sh -- pytest \
  tests/benchmark/test_runner_latency_stress_pass_through.py \
  tests/tools/test_run_amv_actuation_sensitivity_sweep.py \
  tests/benchmark/test_latency_stress.py \
  tests/benchmark/test_map_runner_utils.py::test_run_map_episode_records_synthetic_actuation_metrics

scripts/dev/run_worktree_shared_venv.sh -- python scripts/tools/run_amv_actuation_sensitivity_sweep.py \
  --manifest configs/benchmarks/issue_2011_amv_actuation_sensitivity_sweep_v0.yaml \
  --output output/issue_2011_amv_actuation_sensitivity_preflight \
  --mode preflight \
  --log-level WARNING

scripts/dev/run_worktree_shared_venv.sh -- python scripts/tools/run_amv_actuation_sensitivity_sweep.py \
  --manifest configs/benchmarks/issue_2011_amv_actuation_sensitivity_sweep_v0.yaml \
  --output output/issue_2011_amv_actuation_sensitivity_pilot_summary \
  --mode pilot \
  --log-level WARNING
```

Pilot outcome:

- 12 generated configs passed camera-ready preflight.
- 12 pilot campaigns ran, 4 episodes each, for 48 total episode records.
- Each campaign was classified `accepted_unavailable_only`.
- Each planner row was `not_available` with reason
  `latency_stress_profile is preflight/provenance-only; runtime latency metrics are not implemented`.

The effect-size summary therefore remains diagnostic only. In this two-scenario, one-seed pilot,
success and collision deltas were zero across field groups; actuation trace metrics such as
`command_clip_fraction`, `yaw_rate_saturation_fraction`, and `signed_braking_peak_m_s2` varied in
some rows, but the row status prevents treating those variations as benchmark evidence.

## Follow-Up Boundary

The next evidence-strengthening step is to implement or explicitly disable runtime latency metrics
for this sweep. Until then, issue #2011 outputs should be described as reproducible diagnostic
plumbing evidence, not as planner-performance evidence.
