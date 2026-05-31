# Candidate Report: planner_selector_v2_diagnostic (smoke)

## Decision

pass

## Hypothesis

A deterministic, auditable selector can choose among existing local-policy candidates using predeclared topology and seed-sensitivity diagnostics plus current local pedestrian context. This row is diagnostic only: smoke success proves runnable wiring and diagnostics, not benchmark-strength planner-selection improvement.


## Evaluation Scope

- Stage: `smoke`
- Algorithm: `planner_selector_v2_diagnostic`
- Scenario matrix: `configs/scenarios/single/planner_sanity_simple.yaml`
- Seed manifest: `suite default`
- Summary JSON: `output/policy_search/planner_selector_v2_diagnostic/smoke/issue1806_local/summary.json`
- Git commit: `9e0c2170de35397edc7202f110d041439c1163fc`

## Aggregate Results

| Episodes | Success | Collision | Near Miss | Mean MinDist | Mean AvgSpeed |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.0000 | 0.0000 | 0.0000 | n/a | 1.8480 |

## Scenario-Family Split

| Family | Episodes | Success | Collision | Near Miss |
|---|---:|---:|---:|---:|
| nominal | 1 | 1.0000 | 0.0000 | 0.0000 |

## Failure Taxonomy

- No failures recorded.

## Selector Diagnostics

- Selected candidate: `fast_progress_static_escape`
- Trigger reason: `predeclared_seed_sensitive_low_progress_risk`
- Step count: `76`
- No-leakage boundary: uses predeclared scenario/seed flags and current robot/pedestrian positions;
  outcome, metric, future-observation, benchmark-rank, success, and collision fields are excluded.
- Evidence sources for predeclared flags:
  `docs/context/issue_1608_seed_sensitivity_analysis.md` and
  `docs/context/issue_1692_topology_hypothesis_probe.md`

## Selector Comparison

A same-worktree smoke comparison was run for `planner_selector_v1` with the same stage and seed,
writing only ignored output under `output/policy_search/planner_selector_v1/smoke/issue1806_compare`.
It completed the command path but recorded `0.0000` success, `0.0000` collision, and one
`timeout_low_progress` failure. The comparison supports the narrow claim that selector v2 is wired
and can choose a faster progress head on the smoke scenario; it is not benchmark-strength evidence.

## Claim Boundary

This smoke run proves that `planner_selector_v2_diagnostic` is registered, runnable through the
policy-search runner, and emits selected-candidate/no-leakage diagnostics. It does not prove that
the selector improves benchmark ranking, safety, comfort, or general navigation quality.

## Baseline Deltas

| Baseline | Success Delta | Collision Delta | Near-Miss Delta |
|---|---:|---:|---:|
| goal | +0.9858 | -0.2411 | n/a |
| orca | +0.8156 | -0.0355 | n/a |
| ppo | +0.7518 | -0.0993 | n/a |
