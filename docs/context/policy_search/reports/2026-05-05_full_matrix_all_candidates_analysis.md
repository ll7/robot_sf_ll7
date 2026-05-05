# Full-Matrix Policy Search Analysis (2026-05-05)

## Scope

This note summarizes the completed all-implemented policy-search full-matrix SLURM sweep:

- Slurm job: `12275`
- Run id: `policy_search_full_matrix_all_20260505`
- Stage: `full_matrix`
- Candidate file:
  `output/policy_search/sweeps/policy_search_full_matrix_all_20260505/candidates_full_matrix.txt`
- Candidate reports: 29 tracked full-matrix reports under `docs/context/policy_search/reports/`
- Portfolio overview:
  `docs/context/policy_search/portfolio_overview_2026-05-05.md`
- Comparison artifacts:
  `docs/context/policy_search/reports/comparisons/2026-05-05_full_matrix_all_candidates/`
- Promotion checks:
  `docs/context/policy_search/reports/promotions/2026-05-05_full_matrix_top_candidates/`

All 29 array tasks completed with exit code `0:0`. `learned_risk_model_v1` remains
`slurm_handoff_required` and was not part of this implemented-candidate sweep.

## Provenance Caveat

The sweep used the shared mutable worktree while commits were being made during the array. The
resulting reports therefore span three git commits:

| Commit | Reports | Notes |
|---|---:|---|
| `1d7acbaac53b32dd4d656c5a31466b018dd131f6` | 2 | first array tasks before portfolio tooling commit |
| `a395f943ce0c1d10c9cf88e7e2a52ca302ebc38e` | 11 | intermediate reports after the first portfolio commit |
| `3da4af0a6f424ee819cc3c7904d54745b45ac3c8` | 16 | later reports after intermediate-results commit |

This is acceptable as policy-search triage evidence, but it is not a clean release-grade comparison.
Before promoting a candidate into a camera-ready or paper-facing benchmark surface, rerun the top
candidates from a pinned clean commit or isolated worktree.

## Top Full-Matrix Candidates

| Candidate | Success | Collision | Near Miss | Classic Success | Francis Success | Main Failure |
|---|---:|---:|---:|---:|---:|---|
| `scenario_adaptive_hybrid_orca_v1` | 0.2778 | 0.0139 | 0.3403 | 0.1594 | 0.3867 | `timeout_low_progress` (64) |
| `hybrid_rule_v3_progress_2p4` | 0.2708 | 0.0139 | 0.3403 | 0.1739 | 0.3600 | `timeout_low_progress` (63) |
| `hybrid_rule_v3_fast_progress_static_escape` | 0.2639 | 0.0139 | 0.3403 | 0.1594 | 0.3600 | `timeout_low_progress` (64) |
| `hybrid_rule_v3_waypoint2_route_lookahead8_static02` | 0.2639 | 0.0139 | 0.3333 | 0.1739 | 0.3467 | `timeout_low_progress` (65) |
| `hybrid_rule_v3_waypoint2_route_lookahead8_static05` | 0.2639 | 0.0139 | 0.3333 | 0.1739 | 0.3467 | `timeout_low_progress` (65) |
| `hybrid_rule_v0_minimal` | 0.2569 | 0.0139 | 0.3264 | 0.1594 | 0.3467 | `timeout_low_progress` (65) |
| `hybrid_rule_v3_fast_progress` | 0.2569 | 0.0139 | 0.3333 | 0.1594 | 0.3467 | `timeout_low_progress` (64) |
| `hybrid_rule_v4_recovery_aware` | 0.2500 | 0.0139 | 0.3125 | 0.1594 | 0.3333 | `timeout_low_progress` (68) |

## Model And Policy Analysis

The rule-based hybrid family still dominates the current candidate set. The strongest candidates all
share the same basic shape: conservative static collision filtering, moderate speed/progress pressure,
and local route-following adjustments. Their collision rates are consistently low at `0.0139`, but
they are clustered tightly in success between `0.2500` and `0.2778`, so the real differentiator is
which candidate recovers a few more low-progress episodes without reopening static collisions.

`scenario_adaptive_hybrid_orca_v1` is the best observed candidate and passes the configured `tier_b`
promotion gate. Its advantage is mostly Francis-family behavior: it reaches `0.3867` Francis success
while preserving low collision. It is not a broad breakthrough, though; it is only one successful
episode ahead of `hybrid_rule_v3_progress_2p4` over 144 episodes.

`hybrid_rule_v3_progress_2p4` also passes `tier_b` and is the cleaner policy-only release candidate:
it has slightly better classic success than the scenario-adaptive candidate, the same collision rate,
and no scenario-selector complexity. It should be treated as the simpler ablation/control candidate
for any rerun.

The near-threshold candidates (`hybrid_rule_v3_fast_progress_static_escape`,
`hybrid_rule_v3_waypoint2_route_lookahead8_static02`, and
`hybrid_rule_v3_waypoint2_route_lookahead8_static05`) are useful ablations but not release candidates
under the exact configured gate: they miss the `tier_b` success threshold of `0.264`.

The model-heavy or learned candidates are not competitive in this sweep:

- `mpc_clearance_sampler_v1`: `0.2361` success but `0.2847` collision; not safe enough.
- `risk_guarded_ppo_v1`: `0.1181` success and `0.1736` collision; no release case.
- `planner_selector_v1`: `0.0486` success and `0.3056` collision; selector policy is currently
  harmful.
- `hybrid_orca_sampler_v1`: `0.0139` success; the sampler preserves low near-miss exposure but
  fails to make progress.
- `scenario_adaptive_orca_v1`: low collision (`0.0347`) but only `0.0486` success; ORCA alone is
  still too conservative for this matrix.

## Runtime And Performance Notes

Most rule-based candidates completed in roughly 13-15 minutes with about 1.45-1.50 GB max RSS.
`mpc_clearance_sampler_v1` was the slowest task at `00:16:59` and still had a high collision rate.
`risk_guarded_ppo_v1` finished quickly (`00:04:11`) but used the most memory (`9636780K`) and did
not perform well. Runtime does not currently justify prioritizing the learned/model-heavy branch.

The dominant behavioral failure remains `timeout_low_progress`, usually around 63-68 episodes for
the best rule-based candidates. Static collisions are controlled in the top group (2/144), but
progress recovery is still weak.

## Release-Candidate Decision

Policy-search release candidates:

1. `scenario_adaptive_hybrid_orca_v1` - passes `tier_b`; best current success/collision tradeoff.
2. `hybrid_rule_v3_progress_2p4` - passes `tier_b`; simpler policy-only candidate and best ablation
   control.

Not release candidates yet:

- `hybrid_rule_v3_fast_progress_static_escape` and route-lookahead static variants are close but
  need revision or a pinned rerun because they miss the exact `tier_b` threshold.
- Model-heavy and learned candidates should not be promoted from this sweep.

These are experimental policy-search release candidates, not camera-ready paper candidates. A clean
pinned rerun is required before broader benchmark promotion.

## Recommended Next Steps

1. Rerun only `scenario_adaptive_hybrid_orca_v1` and `hybrid_rule_v3_progress_2p4` from a clean
   pinned commit/worktree. Do not commit into that worktree while the SLURM job is running.
2. Add a sweep-runner guard that records the starting commit and fails or warns if the worktree is
   dirty or the commit changes before each array task starts.
3. If both candidates reproduce, promote `scenario_adaptive_hybrid_orca_v1` as the experimental
   policy-search leader and keep `hybrid_rule_v3_progress_2p4` as the simpler control.
4. Focus the next design iteration on low-progress recovery, not static safety. The top candidates
   already control static collisions; their shared bottleneck is route progress under conservative
   filtering.
5. Deprioritize `mpc_clearance_sampler_v1`, `planner_selector_v1`, `risk_guarded_ppo_v1`, and
   `hybrid_orca_sampler_v1` until their collision/progress contracts are repaired.
