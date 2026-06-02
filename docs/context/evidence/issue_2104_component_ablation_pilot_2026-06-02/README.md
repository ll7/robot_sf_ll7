# Issue #2104 Component Ablation Pilot Evidence

Date: 2026-06-02

This bundle records a compact retrospective pilot for issue
[#2104](https://github.com/ll7/robot_sf_ll7/issues/2104). It reuses the durable
S10/H500 candidate campaign from issue #1454 instead of running a new benchmark campaign.

## Source Evidence

- Manifest:
  `configs/policy_search/ablation_manifests/issue_2104_component_ablation_pilot.yaml`
- Source bundle:
  `docs/context/evidence/issue_1454_s10_h500_candidates_2026-05-23/README.md`
- Source table:
  `docs/context/evidence/issue_1454_s10_h500_candidates_2026-05-23/reports/seed_variability_by_scenario.csv`
- Source campaign command:

```bash
uv run python scripts/tools/run_camera_ready_benchmark.py \
  --config configs/benchmarks/issue_1454_s10_scenario_horizons_h500_candidates.yaml \
  --campaign-id issue1454-s10-h500-candidates
```

The source campaign ran at commit `4941ac48f1f4e65053bbfcbbc94a55a336fad9ea`.
The raw archive is preserved at
<https://github.com/ll7/robot_sf_ll7/releases/tag/artifact/issue1454-s10-h500-candidates-2026-05-23>
with SHA-256
`44ec1d4eb89d450eb204398a3807185ce9bdd4aae0eeb5e55af0704fd4a8b0fc`.

## Method

The source CSV stores one row per scenario, planner, and seed, with across-seed summary
columns repeated for each seed. This pilot deduplicates by `(scenario_id, planner_key)`,
then averages the across-seed means over the 48 scenarios for each candidate row.

Comparisons are paired by shared scenario id. The deltas below are `A - B`.

This is a retrospective grouped-component analysis. It separates interpretable historical
rows, but it does not prove one-factor causality because several candidates combine multiple
mechanisms.

## Candidate Summary

| Planner | Scenarios | Success | Collision | Near miss | Runtime norm | SNQI |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `hybrid_rule_v3_fast_progress` | 48 | 0.7875 | 0.0292 | 21.9917 | 0.7385 | -0.1160 |
| `hybrid_rule_v3_fast_progress_static_escape` | 48 | 0.8646 | 0.0354 | 21.8875 | 0.7118 | -0.1069 |
| `hybrid_rule_v3_fast_progress_static_escape_continuous` | 48 | 0.8771 | 0.0250 | 18.9146 | 0.7221 | -0.0972 |
| `scenario_adaptive_hybrid_orca_v1` | 48 | 0.8729 | 0.0333 | 20.7771 | 0.7083 | -0.1037 |
| `scenario_adaptive_hybrid_orca_v2_collision_guard` | 48 | 0.8729 | 0.0333 | 20.7771 | 0.7084 | -0.1037 |
| `orca` | 48 | 0.7750 | 0.1604 | 13.8250 | 0.6972 | -0.2476 |

## Effect-Size Pilot

| Comparison | Scenarios | dSuccess | dCollision | dNear miss | dRuntime norm | dSNQI | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `static_escape_minus_fast_progress` | 48 | +0.0771 | +0.0063 | -0.1042 | -0.0267 | +0.0091 | Success wins/losses 13/1; collision wins/losses 3/0. |
| `continuous_static_checks_minus_static_escape` | 48 | +0.0125 | -0.0104 | -2.9729 | +0.0103 | +0.0097 | Success wins/losses 8/5; collision wins/losses 0/4. |
| `scenario_adaptive_v1_minus_static_escape` | 48 | +0.0083 | -0.0021 | -1.1104 | -0.0034 | +0.0032 | Success wins/losses 1/0; collision wins/losses 0/1. |
| `collision_guard_v2_minus_v1` | 48 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | -0.0000 | Success wins/losses 0/0; collision wins/losses 0/0. |
| `scenario_adaptive_v1_minus_orca` | 48 | +0.0979 | -0.1271 | +6.9521 | +0.0111 | +0.1439 | Success wins/losses 18/14; collision wins/losses 0/20. |

## Interpretation

The grouped static-escape/recenter row improves success by about `+0.077` over
the fast-progress base on this S10/H500 surface, with a small collision increase
of about `+0.006`. The continuous-static-check row further improves success by
about `+0.013`, lowers collision by about `-0.010`, and reduces near misses by
about `-2.97`, but it also slightly worsens normalized runtime.

The scenario-adaptive ORCA selector adds only a small aggregate change over the
static-escape row on this slice, while the v2 collision-guard row is effectively
identical to v1 in the preserved table. Against plain ORCA, the hybrid selector
has much lower collision and higher success, but more near misses; that supports
using ORCA as a reference row, not as proof that the selector mechanism is
causal by itself.

## Caveats

- This pilot is `diagnostic_only`; it is not a new benchmark campaign.
- SNQI is diagnostic only because the source evidence note records an SNQI contract failure.
- Runtime uses `time_to_goal_norm`, not wall-clock planner runtime.
- Several rows are grouped-component candidates, so interaction effects remain unresolved.
- A true one-factor ablation still needs a purpose-built manifest that toggles static escape,
  recentering, route guidance, speed envelope, dynamic collision horizon, and guard overlays
  independently on identical scenarios and seeds.
