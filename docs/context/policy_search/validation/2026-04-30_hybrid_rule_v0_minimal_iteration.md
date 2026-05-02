# Iteration Report: hybrid_rule_v0_minimal

## Change Summary

Added `hybrid_rule_local_planner` with the `hybrid_rule_v0_minimal` variant:

- deterministic DWA-style candidate generation,
- path-following and stop/creep safety candidates,
- hard static/dynamic collision filtering,
- normalized score terms for progress, alignment, clearance, TTC, smoothness, effort, freezing,
  and oscillation,
- per-episode planner diagnostics under `algorithm_metadata.planner_runtime`,
- config-first entry points under `configs/algos/` and `configs/policy_search/candidates/`.

## Hypothesis

A transparent non-learning DWA-style control variant should be a clean starting point for later
social, ORCA, TEB-like, recovery, and ensemble mechanisms. The first proof target is only that v0
runs in the real benchmark path and does not fail the open sanity route.

## Benchmark Result

Command:

```bash
LOGURU_LEVEL=WARNING uv run python scripts/validation/run_policy_search_candidate.py \
  --candidate hybrid_rule_v0_minimal \
  --stage smoke \
  --workers 1 \
  --output-dir output/policy_search/hybrid_rule_v0_minimal_smoke
```

Smoke output:

| Planner | Episodes | Success | Collision | Timeout | Near Miss | Mean Avg Speed |
|---|---:|---:|---:|---:|---:|---:|
| hybrid_rule_v0_minimal | 1 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 1.9082 |

Direct baseline comparison on `configs/scenarios/single/planner_sanity_simple.yaml`, default
scenario seeds `[101, 102, 103]`, `horizon=80`, `dt=0.1`:

| Planner | Episodes | Success | Collision | Timeout | Time Norm | Path Eff. | Energy |
|---|---:|---:|---:|---:|---:|---:|---:|
| goal | 3 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 0.3844 |
| hybrid_rule_v0_minimal | 3 | 1.0000 | 0.0000 | 0.0000 | 0.8833 | 1.0000 | 16.1306 |

Nominal sanity command:

```bash
LOGURU_LEVEL=WARNING uv run python scripts/validation/run_policy_search_candidate.py \
  --candidate hybrid_rule_v0_minimal \
  --stage nominal_sanity \
  --workers 2 \
  --output-dir output/policy_search/hybrid_rule_v0_minimal_nominal_sanity
```

Nominal sanity output:

| Planner | Episodes | Success | Collision | Timeout | Near Miss | Mean MinDist | Mean Avg Speed |
|---|---:|---:|---:|---:|---:|---:|---:|
| hybrid_rule_v0_minimal | 18 | 0.1667 | 0.4444 | 0.3889 | 0.2222 | 3.9469 | 1.7281 |

Artifacts:

- `output/policy_search/hybrid_rule_v0_minimal_smoke/summary.json`
- `output/policy_search/hybrid_rule_v0_minimal_smoke/smoke__hybrid_rule_v0_minimal.jsonl`
- `output/policy_search/hybrid_rule_v0_minimal_baseline_compare/goal.jsonl`
- `output/policy_search/hybrid_rule_v0_minimal_baseline_compare/hybrid_rule_local_planner.jsonl`
- `output/policy_search/hybrid_rule_v0_minimal_nominal_sanity/summary.json`
- `output/policy_search/hybrid_rule_v0_minimal_nominal_sanity/nominal_sanity__hybrid_rule_v0_minimal.jsonl`

## Main Improvements

- The planner executes through the real map-runner policy path.
- The open sanity route completes on all three default scenario seeds with no collisions or near
  misses.
- Diagnostics now include selected command, top-k candidate score terms, rejection counts, nearest
  pedestrian/static-obstacle distance, predicted TTC, progress windows, selected source counts, and
  fallback count.

## Main Regressions

- Energy is higher than `goal` because v0 uses a faster open-space cap to complete the short smoke
  horizon.
- Nominal sanity failed the gate: 0.1667 success, 0.4444 collision-rate-equivalent termination,
  and 0.2222 near-miss rate.
- Classic and Francis dynamic scenes are not solved by the minimal constant-command DWA variant.

## Failure Mode Analysis

- Smoke/open-sanity failures: none after the 2.0 m/s cap.
- Nominal sanity failures:
  - `static_collision`: 8
  - `timeout_low_progress`: 5
  - `near_miss_intrusive`: 2
- Diagnostics show large `static_collision` rejection counts in crossing, doorway, overtaking, and
  Francis following-human scenarios, plus many `dynamic_collision` rejections in doorway/head-on
  scenes. The current v0 has no side-passing, recovery, or corridor commitment, so this pattern is
  expected rather than evidence for promotion.

## Diagnostic Evidence

The smoke episode recorded `planner_runtime.planner_variant=hybrid_rule_v0_minimal`,
`fallback_count=0`, selected sources dominated by `dynamic_window`, and per-step feasibility
metadata with `projection_rate=0.0`. Nominal-sanity failure rows contain populated
`planner_runtime.rejection_counts`, which is sufficient to drive the next mechanism choice.

## Decision

modify

## Next Action

Add one mechanism at a time:

1. strengthen static obstacle handling/path-corridor adherence or add TEB-like skeletons for the
   `static_collision` failures,
2. then add VO/TTC and passing-side logic for the dynamic rejection/near-miss cases,
3. then add recovery only if low-progress timeouts remain after static/dynamic handling improves.
