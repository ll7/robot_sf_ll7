# AMV Actuation Diagnostic Suite

```yaml
suite_id: amv_actuation_diagnostic
benchmark_track: synthetic_amv_actuation_diagnostic
status: runnable_local_diagnostic
```

## Purpose

Run synthetic actuation-envelope diagnostics for AMV-style command stress. This suite is designed
to reveal command clipping, yaw saturation, braking peaks, projection rates, and row-status
caveats; it is not calibrated hardware evidence.

## Scenarios And Seeds

Two related local surfaces exist:

- Policy-search smoke stage:
  - Stage config: `configs/policy_search/funnel.yaml`
  - Scenario matrix: `configs/scenarios/sets/classic_cross_trap_subset.yaml`
  - Scenario filter: `classic_cross_trap_high`
  - Seed: `111`
  - Horizon: `80`
- Camera-ready-style diagnostic config:
  - Config: `configs/benchmarks/issue_1556_amv_actuation_stress_slice_v0.yaml`
  - Scenario candidates: `classic_overtaking_medium`, `classic_bottleneck_high`,
    `classic_cross_trap_high`, `francis2023_blind_corner`,
    `francis2023_intersection_wait`
  - Seed policy: eval seed set from `configs/benchmarks/seed_sets_v1.yaml`
  - Horizon: `100`

## Eligible Planners

For the compact Issue #1569 smoke evidence: `goal`, `orca`, and `social_force`. For policy-search
experiments: only candidates whose config intentionally declares an AMV actuation diagnostic
hypothesis and can preserve fail-closed row status.

## Metrics

Success, collision, near misses, command clip fraction, yaw-rate saturation fraction, signed braking
peak, projection rate, infeasible rate, runtime, AMV coverage status, row-status summary, and SNQI
when produced by the camera-ready analyzer.

## Canonical Commands

Policy-search candidate diagnostic:

```bash
uv run python scripts/validation/run_policy_search_candidate.py \
  --candidate <candidate_id> \
  --stage amv_actuation_smoke \
  --output-dir output/policy_search/<candidate_id>/amv_actuation_smoke/manual \
  --workers 1
```

Camera-ready-style diagnostic:

```bash
uv run python scripts/tools/run_camera_ready_benchmark.py \
  --config configs/benchmarks/issue_1556_amv_actuation_stress_slice_v0.yaml \
  --label manual-amv-diagnostic \
  --output-root output/benchmarks/manual_amv_actuation \
  --log-level WARNING
```

## Expected Runtime

The policy-search smoke is expected to be quick. The camera-ready-style diagnostic is broader and
should be treated as a local benchmark run whose runtime depends on planner availability and
analysis settings.

## Claim Boundary

Synthetic AMV actuation diagnostics are non-paper-facing unless a separate claim-map decision
promotes a specific evidence bundle. The tracked Issue #1569 evidence explicitly says it does not
strengthen AMV performance claims.

## Caveats

AMV coverage warnings, command-space gaps, projection-heavy rows, excluded rows, fallback rows, and
unavailable rows remain visible limitations. Do not reinterpret diagnostic rows as calibrated
hardware evidence.
