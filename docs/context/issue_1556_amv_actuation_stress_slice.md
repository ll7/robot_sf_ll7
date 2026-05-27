# Issue #1556 synthetic AMV actuation stress slice

Related issue: <https://github.com/ll7/robot_sf_ll7/issues/1556>

Reference design note:

- [`docs/context/issue_1546_amv_actuation_envelope_stress_slice.md`](issue_1546_amv_actuation_envelope_stress_slice.md)

## Goal

Implement the first compact synthetic AMV actuation-envelope diagnostic slice without changing
benchmark gates or making paper-facing or hardware-calibration claims.

## Implemented boundary

- The benchmark config is `paper_facing: false` and `kinematics_matrix: [differential_drive]`.
- The slice uses the compact scenario candidate set from `#1546`:
  `classic_overtaking_medium`, `classic_bottleneck_high`, `classic_cross_trap_high`,
  `francis2023_blind_corner`, and `francis2023_intersection_wait`.
- The checked-in config now carries slice-local `scenario_amv_overrides` for those five scenarios so
  AMV coverage artifacts and compact actuation summaries do not depend on unrelated global scenario
  files carrying the same taxonomy.
- The seed policy stays on the named `eval` seed set for the checked-in config.
- Synthetic profile provenance is carried in preflight output, campaign manifests, episode
  `scenario_params`, planner-row summaries, and the diagnostic
  `reports/actuation_envelope_summary.{json,md}` artifacts.
- The diagnostic actuation summary now also records AMV coverage status, compact scenario-level AMV
  rows, and planner command-space/projection-policy metadata so issue-1572-style evidence review can
  distinguish scenario taxonomy gaps from adapter metadata gaps without reopening raw campaign rows.
- Derived saturation metrics currently reported when the command path supports them are:
  `command_clip_fraction`, `yaw_rate_saturation_fraction`, and `signed_braking_peak_m_s2`.
- If the synthetic profile cannot be applied, the runner fails closed instead of silently dropping
  the profile.

## Claim boundary

- This is a synthetic software stress slice only.
- The profile values are not a real AMV hardware specification and are not a calibration claim.
- The added actuation report is a diagnostic supplement; it does not redefine benchmark success.
- Fallback, degraded, unavailable, skipped, and failed rows remain non-success evidence under
  [`docs/context/issue_691_benchmark_fallback_policy.md`](issue_691_benchmark_fallback_policy.md).

## Validation path

Use focused proof instead of a broad campaign:

```bash
source .venv/bin/activate
pytest tests/benchmark/test_issue_1556_amv_actuation_stress_slice.py \
       tests/benchmark/test_camera_ready_campaign.py \
       tests/benchmark/test_map_runner_utils.py

python scripts/tools/run_camera_ready_benchmark.py \
       --config configs/benchmarks/issue_1556_amv_actuation_stress_slice_v0.yaml \
       --mode preflight \
       --label issue1556-local \
       --output-root output/benchmarks/camera_ready_issue1556 \
       --log-level WARNING
```

The key contract checks are:

1. config provenance for scenario candidates, eval seeds, and synthetic profile fields,
2. preflight propagation of candidate resolution and synthetic profile metadata,
3. campaign-summary/report artifact emission for the actuation supplement,
4. slice-local scenario AMV override propagation into preflight and compact actuation artifacts,
5. map-runner episode metrics and fail-closed behavior for non-differential-drive scenarios.
6. config-loader fail-closed behavior for malformed scenario candidate, scenario AMV override, and synthetic profile
   payloads.
