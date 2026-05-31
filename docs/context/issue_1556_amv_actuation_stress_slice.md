# Issue #1556 Synthetic AMV Actuation Stress Slice

Related issue: <https://github.com/ll7/robot_sf_ll7/issues/1556>

Reference design note:

- [`docs/context/issue_1546_amv_actuation_envelope_stress_slice.md`](issue_1546_amv_actuation_envelope_stress_slice.md)

## Goal

Implement the first compact synthetic AMV actuation-envelope diagnostic slice without changing
benchmark gates or making paper-facing or hardware-calibration claims.

## Implemented Boundary

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

## Claim Boundary

- This is a synthetic software stress slice only.
- The profile values are not a real AMV hardware specification and are not a calibration claim.
- The added actuation report is a diagnostic supplement; it does not redefine benchmark success.
- Fallback, degraded, unavailable, skipped, and failed rows remain non-success evidence under
  [`docs/context/issue_691_benchmark_fallback_policy.md`](issue_691_benchmark_fallback_policy.md).

## Validation Path

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

## Issue #1569 Local Smoke Result (2026-05-27)

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

Interpretation Boundary:

- `benchmark_success=true` in this local smoke means the three planner rows satisfied the executable
  benchmark contract (`native` or accepted `adapter` execution with valid artifacts). It does **not**
  mean any planner solved the slice.
- This result is still **synthetic diagnostic only**. It does not promote the slice to a
  paper-facing claim, and it does not support hardware-calibration language.
- The AMV claim map remained **unchanged as a paper-facing claim**. The smoke confirms that the
  issue-1556 slice is runnable and emits the intended actuation diagnostics locally, but it does not
  strengthen any AMV performance claim.
- `amv_coverage_status` remained `warn` because the resolved scenario rows in
  `configs/scenarios/classic_interactions_francis2023.yaml` still expose empty `amv` metadata
  blocks for this slice.
- Adapter diagnostics remain part of the caveat surface: ORCA reported a high command projection
  rate (`0.8018`), and the smoke evidence records this without downgrading the row from accepted
  adapter execution.

## Issue #1572 / #1582 Metadata Contract Closure (2026-05-31)

Issue #1572 and its decision split #1582 are now closed. Merged PR #1580 accepted the conservative
metadata contract for future synthetic actuation diagnostics:

1. The checked-in config may use slice-local synthetic AMV taxonomy overrides when those fields
   remain tied to the versioned synthetic profile and diagnostic-only claim scope.
2. Compact actuation summaries should expose scenario AMV rows plus planner command-space and
   projection-policy metadata.
3. Unknown, unavailable, fallback, degraded, skipped, and failed rows remain caveats and must not
   count as actuation-envelope success evidence.

This closure improves the interpretability of future `actuation_envelope_summary.*` artifacts. It
does not change the Issue #1569 smoke's historical raw-campaign caveats, and it does not create
calibrated or paper-facing actuation evidence.

## Issue #1570 Claim-Map Verdict (2026-05-31)

The claim-map boundary after the smoke verdict is:

- **Synthetic diagnostics:** supported as a software stress diagnostic from Issue #1556. The profile
  is `amv-actuation-stress-v0`, `paper_facing: false`, and `claim_scope: synthetic-only`.
- **Compact smoke evidence:** supported only as non-paper-facing local evidence that the slice runs
  and emits clip/yaw/braking diagnostics. The Issue #1569 smoke had valid executable rows but
  `success_mean=0.0` for all planners, so it is not an AMV performance claim.
- **Calibrated/paper-facing evidence:** still blocked. Issue
  [#1559](https://github.com/ll7/robot_sf_ll7/issues/1559) remains the gate for a durable AMV
  calibration source, calibrated-vs-synthetic profile separation, and validation that prevents
  synthetic values from being reported as hardware evidence.

Updated claim map:
[`issue_1542_manuscript_claim_evidence_map.md`](issue_1542_manuscript_claim_evidence_map.md).
