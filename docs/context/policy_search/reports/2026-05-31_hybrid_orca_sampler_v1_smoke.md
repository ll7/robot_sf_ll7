# Candidate Report: hybrid_orca_sampler_v1 (smoke)

## Decision

pass

## Hypothesis

Keep ORCA-like safety behavior while allowing a short-horizon sampler to recover progress in constrained geometry.


## Evaluation Scope

- Stage: `smoke`
- Algorithm: `hybrid_orca_sampler`
- Scenario matrix: `configs/scenarios/single/planner_sanity_simple.yaml`
- Seed manifest: `suite default`
- Summary JSON: `output/policy_search/hybrid_orca_sampler_v1/smoke/issue1829_speed2_final/summary.json`
- Git commit: `a9200811585003f471d1529a82b247ac1a4e15ae`

## Aggregate Results

| Episodes | Success | Collision | Near Miss | Mean MinDist | Mean AvgSpeed |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.0000 | 0.0000 | 0.0000 | n/a | 1.9043 |

## Scenario-Family Split

| Family | Episodes | Success | Collision | Near Miss |
|---|---:|---:|---:|---:|
| nominal | 1 | 1.0000 | 0.0000 | 0.0000 |

## Failure Taxonomy

- No failures recorded.

## Issue #1829 Comparison - 2026-05-31

This run retunes the experimental candidate's speed envelope from the inherited `1.1 m/s` cap to
`2.0 m/s` for both the ORCA head and MPPI sampler, and keeps the route-stall sampler handoff
diagnostic added in the same branch.

Tracked predecessor smoke evidence recorded `0/3` successes, `0/3` collisions, and
`timeout_low_progress=3` on `planner_sanity_simple`. A same-worktree 3-seed probe with
`seed_list: [101, 102, 103]` after this change produced `3/3` successes, `0/3` collisions,
`0/3` near misses, and no failures:
`output/policy_search/hybrid_orca_sampler_v1/smoke/issue1829_speed2_3seeds_final/summary.json`.

This is smoke evidence only. It repairs the short-horizon progress failure on the nominal sanity
smoke surface, but it does not promote `hybrid_orca_sampler_v1` beyond experimental policy-search
candidate status.

## Baseline Deltas

| Baseline | Success Delta | Collision Delta | Near-Miss Delta |
|---|---:|---:|---:|
| goal | +0.9858 | -0.2411 | n/a |
| orca | +0.8156 | -0.0355 | n/a |
| ppo | +0.7518 | -0.0993 | n/a |
