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
- The seed policy stays on the named `eval` seed set for the checked-in config.
- Synthetic profile provenance is carried in preflight output, campaign manifests, episode
  `scenario_params`, planner-row summaries, and the diagnostic
  `reports/actuation_envelope_summary.{json,md}` artifacts.
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
4. map-runner episode metrics and fail-closed behavior for non-differential-drive scenarios.
5. config-loader fail-closed behavior for malformed scenario candidate and synthetic profile
   payloads.

## Issue #1569 local smoke result (2026-05-27)

- **Verdict:** `compact smoke run`
- **Evidence bundle:** [`docs/context/evidence/issue_1569_amv_actuation_smoke_2026-05-27/`](evidence/issue_1569_amv_actuation_smoke_2026-05-27/README.md)
- **Commands:** preflight, compact smoke, and analyzer runs used the checked-in config
  `configs/benchmarks/issue_1556_amv_actuation_stress_slice_v0.yaml` with local labels
  `issue1569-local` and `issue1569-smoke`.

Observed local outcome:

1. The compact 45-episode smoke completed with `campaign_execution_status=completed`,
   `evidence_status=valid`, and `row_status_summary={successful_evidence_rows: 3,
   accepted_unavailable_rows: 0, unexpected_failed_rows: 0, fallback_or_degraded_rows: 0}`.
2. The analyzer reported no internal consistency findings for the generated campaign artifacts.
3. The synthetic actuation report emitted non-zero clip fractions for all three planner rows
   (`goal=0.0407`, `orca=0.1047`, `social_force=0.2346`) while `yaw_rate_saturation_fraction`
   stayed `0.0000` for all rows.
4. Episode-level performance was still poor: `success_mean=0.0000` for all three planners, and the
   consensus hardest scenario was `classic_cross_trap_high`.

Interpretation boundary:

- `benchmark_success=true` in this local smoke means the three planner rows satisfied the executable
  benchmark contract (`native` or accepted `adapter` execution with valid artifacts). It does **not**
  mean any planner solved the slice.
- This result is still **synthetic diagnostic only**. It does not promote the slice to a
  paper-facing claim, and it does not support hardware-calibration language.
- The AMV claim map is **unchanged**. The smoke confirms that the issue-1556 slice is runnable and
  emits the intended actuation diagnostics locally, but it does not strengthen any AMV performance
  claim.
- `amv_coverage_status` remained `warn` because the resolved scenario rows in
  `configs/scenarios/classic_interactions_francis2023.yaml` still expose empty `amv` metadata
  blocks for this slice.
- Adapter diagnostics remain part of the caveat surface: ORCA reported a high command projection
  rate (`0.8018`), and the smoke evidence records this without downgrading the row from accepted
  adapter execution. Follow-up issue #1572 owns the remaining scenario and adapter metadata gaps.
