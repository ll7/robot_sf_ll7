# Issue #2966 Planner-Consumed Forecast Slice (2026-06-19)

[Issue #2966](https://github.com/ll7/robot_sf_ll7/issues/2966) runs the smallest
planner-consumed forecast slice after
[Issue #2960](https://github.com/ll7/robot_sf_ll7/issues/2960), with fail-closed
semantics. This is **diagnostic/smoke evidence** only.

## Evidence Boundary

This is **not** benchmark evidence. It proves that forecast variants are consumed
by `PredictionPlannerAdapter` and produce forecast-brake replay closed-loop
metrics. It does not prove that any forecast variant improves safety, success,
runtime, or paper-facing benchmark performance.

## Fixture Limitations

Current validation scripts only support:
- One deterministic SocNav observation fixture (consumer smoke, seed 2960)
- One trace fixture (`dense_pedestrian_stress` seed 2765, live replay)

A 3-scenario map benchmark is not available without additional trace fixtures or
a scenario-specific map-runner config. The selection below documents why 3
motion-rich scenarios were **not** faked.

## Commands

```bash
# Consumer smoke (exit 0)
scripts/dev/run_worktree_shared_venv.sh -- uv run python scripts/validation/validate_forecast_planner_consumer.py \
  --output-dir output/issue_2966_forecast_slice/consumer_smoke \
  --variants none cv semantic interaction_aware risk_filtered

# Live replay full matrix (exit 0)
scripts/dev/run_worktree_shared_venv.sh -- uv run python scripts/validation/validate_live_forecast_replay_gate.py \
  --full-matrix \
  --output-dir output/issue_2966_forecast_slice/live_replay_full_matrix
```

## Results

### Consumer Smoke (deterministic SocNav observation)

| Variant | Classification | Prediction Changed vs none |
|---|---|---|
| `none` | native | — (baseline) |
| `cv` | degraded | false |
| `semantic` | degraded | false |
| `interaction_aware` | native | true |
| `risk_filtered` | degraded | false |

All variants execute in `native` mode (no missing-component fallback). Degraded
classification means the predicted futures match the no-forecast baseline.

### Live Replay Full Matrix (dense_pedestrian_stress trace)

| Variant | Classification | Collision | Progress | False-Positive Stops |
|---|---|---|---|---|
| `none` | native | false | 0.545 m | 0 |
| `cv` | degraded | false | 0.0 m | 20 |
| `semantic` | degraded | false | 0.0 m | 20 |
| `interaction_aware` | degraded | false | 0.0 m | 20 |
| `risk_filtered` | degraded | false | 0.0 m | 20 |

All non-none variants produce **identical** closed-loop metrics. The
forecast-brake replay policy stops unconditionally for all forecast variants.

## Map-Runner Slice Decision

**STOP.** All non-none live-replay variants produce identical degraded
closed-loop metrics (20 false-positive stops, 0.0 progress). Consumer smoke
confirms cv/semantic/risk_filtered predictions match baseline. A 3-scenario
map-runner slice with `allow_fallback=false` would not produce new evidence.

## What Was Not Claimed

- No benchmark-strengthening claim
- No planner-superiority claim
- No safety-improvement claim
- No paper-grade claim
- No scenario-diversity claim (only 2 fixtures used)
- No statistical-significance claim

## Same-Seed Caveats

- Consumer smoke uses a single deterministic SocNav observation with seed 2960.
- Live replay uses a single trace (`dense_pedestrian_stress` seed 2765).
- Neither fixture provides scenario diversity; results are fixture-specific.

## Durable Artifacts

- Evidence summary:
  [issue_2966_planner_consumed_forecast_slice_summary.json](evidence/issue_2966_planner_consumed_forecast_slice_summary.json)
- Consumer smoke: `output/issue_2966_forecast_slice/consumer_smoke/`
- Live replay: `output/issue_2966_forecast_slice/live_replay_full_matrix/`

## Next Recommended Step

Tracked as follow-up [Issue #3146](https://github.com/ll7/robot_sf_ll7/issues/3146).

To expand beyond diagnostic smoke:
1. Create motion-rich trace fixtures for at least 3 scenario families (crossing,
   bottleneck, corridor) with forecast-affecting pedestrian dynamics.
2. Tune the forecast-brake replay policy to avoid uniform false-positive stopping.
3. Rerun the live replay full matrix on the diversified fixtures.
