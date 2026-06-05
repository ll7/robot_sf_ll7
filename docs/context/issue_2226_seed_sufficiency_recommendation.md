# Issue #2226 Seed Sufficiency Recommendation

Related issue: <https://github.com/ll7/robot_sf_ll7/issues/2226>

Status: diagnostic recommendation from existing durable evidence; no new benchmark run.

## Question

How many seeds are necessary before planner rankings and scenario-family conclusions stabilize?

The answer from currently tracked evidence is deliberately split:

- S3 and S5 can support smoke and staged robustness checks when the campaign contract is matched.
- S10 can support diagnostic ranking and seed-sensitivity analysis on the historical h500 candidate
  surface.
- S20/S30 remains the paper-facing gate when close planner rankings, safety deltas, or scenario
  family conclusions matter.

No currently tracked durable bundle gives a single identical planner/scenario matrix at S3, S5,
S10, and S20. That means issue #2226 cannot honestly conclude full seed sufficiency across all four
budgets. It can close with a bounded recommendation and a data-gap contract.

## Evidence Inventory

| Evidence | Seed budget | Use | Boundary |
| --- | ---: | --- | --- |
| [issue_832_paper_matrix_extended_seed_schedule.md](issue_832_paper_matrix_extended_seed_schedule.md) | S3, S5 executed; S10/S20 planned | Matched S3-to-S5 paper-matrix comparison. Aggregate SNQI ranking stayed stable; scenario winners and aggregate means shifted enough for `review`. | Output roots are worktree-local in that note; S10/S20 are policy/planning, not durable completed comparisons there. |
| [issue_2125_seed_sufficiency_ranking_stability.md](issue_2125_seed_sufficiency_ranking_stability.md) | Analyzer supports multiple roots | Defines the seed-sufficiency analyzer contract and interpretation boundary. | Needs campaign roots with `seed_variability_by_scenario.json` and `seed_episode_rows.csv`. |
| [issue_1545_power_aware_seed_budget_planning.md](issue_1545_power_aware_seed_budget_planning.md) | S1/S3/S10/S20+ methodology | Defines smoke, nominal sanity, compact benchmark, and paper-facing tiers. | Planning note; does not itself produce a new stability run. |
| [issue_1608_seed_sensitivity_analysis.md](issue_1608_seed_sensitivity_analysis.md) | S10 | Classifies seed-sensitive scenarios for the top-four planners on the durable Issue #1454 S10/h500 candidate bundle. | Historical diagnostic surface with `legacy_track_unknown`; not paper-facing significance evidence. |
| [issue_1454_s10_h500_candidates_2026-05-23/](evidence/issue_1454_s10_h500_candidates_2026-05-23/) | S10 | Best durable compact candidate surface for top-planner diagnostic ranking and family-level seed inspection. | Exploratory h500 candidate surface; SNQI contract is `fail`; not matched to S3/S5/S20. |

## Data Availability

| Budget | Durable matched data for issue #2226? | Current interpretation |
| --- | --- | --- |
| S3 | Partial | Exists for paper-matrix and AMV-style surfaces, but not as a complete match to the S10 h500 top-planner candidate set. |
| S5 | Partial in context note | Issue #832 records S5 execution and S3-to-S5 comparison, but there is no local tracked compact S5 bundle suitable for rerunning the issue #2125 analyzer here. |
| S10 | Yes, diagnostic | Issue #1454 S10/h500 candidate bundle and Issue #1608 derived analysis support diagnostic seed sensitivity. |
| S20 | No | S20 is a planned seed policy and paper-facing gate, not a completed durable evidence bundle in the tracked context. |

Verdict: `insufficient_for_full_sufficiency_claim`.

## S10 Diagnostic Result

The strongest available diagnostic surface is the Issue #1608 analysis over the Issue #1454
S10/h500 candidate bundle. It selected the top four benchmark-success rows by success, collision,
`time_to_goal_norm`, near misses, and planner key:

- `hybrid_rule_v3_fast_progress_static_escape_continuous`;
- `scenario_adaptive_hybrid_orca_v1`;
- `scenario_adaptive_hybrid_orca_v2_collision_guard`;
- `hybrid_rule_v3_fast_progress_static_escape`.

Across 48 scenarios and 10 seeds:

| Classification | Count |
| --- | ---: |
| `seed_sensitive` | 25 |
| `not_seed_sensitive` | 23 |
| `inconclusive` | 0 |

Family-level observations for the issue #2226 requested families:

| Family slice | S10 diagnostic result | Interpretation |
| --- | --- | --- |
| `bottleneck` | 4/4 scenarios `not_seed_sensitive`; seed-success range `0.0000` in the derived table. | Stable on this historical top-four h500 surface. |
| `crossing` / `group_crossing` | 5 crossing-like scenarios seed-sensitive; all 3 `classic_group_crossing_*` rows seed-sensitive. | Do not use small-seed crossing/group conclusions without caveats. |
| `join_group` | `francis2023_join_group` seed-sensitive with seed-success range `1.0000`. | Needs larger or held-out seed checks before scenario-family claims. |
| `leave_group` | `francis2023_leave_group` seed-sensitive with seed-success range `0.5000`. | Diagnostic instability remains. |
| `cross_trap` | high and medium seed-sensitive; low not seed-sensitive. | Topology/trap claims should separate scenario difficulty levels. |

The hardest seed id in the Issue #1608 analysis is `116`, with mean top-planner success `0.7396`
and 12 hard scenario rows. Use seed `116` as a targeted trace-review seed, not as a universal hard
seed outside this source surface.

## Recommendations

| Claim tier | Recommended seed budget | Rationale |
| --- | ---: | --- |
| Smoke / path proof | S1-S3 | Enough for schema, dependency, fail-closed, and gross-regression checks only. Do not make ranking claims. |
| Diagnostic planner comparison | S10 | Current strongest durable diagnostic basis. Still too narrow for close paper-facing comparisons and scenario-family stability claims. |
| Scenario-family interpretation | S10 minimum, escalate when seed-sensitive | Bottleneck-like stable rows may be discussed as diagnostic; crossing/group/join/leave need caveats or larger held-out seed checks. |
| Paper-facing ranking or close safety deltas | S20+; S30 when ranks still flip | Required because S5 already showed scenario-winner/mean-drift sensitivity, and S10 shows many seed-sensitive scenarios. |

Practical policy:

1. Use S3/S5 for staged robustness checks and smoke-to-diagnostic transitions.
2. Use S10 for top-planner diagnostic ranking and hard-seed discovery.
3. Escalate to S20/S30 before claiming paper-facing ranking stability, close safety deltas, or
   scenario-family conclusions.
4. When S20 is unavailable, state the blocker rather than averaging S3/S5/S10 evidence across
   non-matched surfaces.

## Next Data Contract

A future issue can turn this recommendation into a stronger result only if it provides matched
campaign roots for the same planner set, scenario matrix, metrics, and seed ordering at S3, S5,
S10, and S20. The issue #2125 analyzer command should then be:

```bash
uv run python scripts/tools/analyze_seed_sufficiency.py \
  --campaign-root output/benchmarks/camera_ready/<s3_campaign_id> \
  --campaign-root output/benchmarks/camera_ready/<s5_campaign_id> \
  --campaign-root output/benchmarks/camera_ready/<s10_campaign_id> \
  --campaign-root output/benchmarks/camera_ready/<s20_campaign_id> \
  --output-dir output/benchmarks/camera_ready/<analysis_id>/reports/seed_sufficiency
```

Required tracked summary outputs:

- `seed_sufficiency_analysis.json`;
- `planner_rank_stability.csv`;
- `scenario_family_instability.csv`;
- `outcome_counts.csv`;
- a compact Markdown interpretation with fallback/degraded/not-available row counts.

## Validation

This docs-only recommendation was checked with:

```bash
BASE_REF=origin/main scripts/dev/check_docs_proof_consistency_diff.sh
git diff --check origin/main...HEAD
```
