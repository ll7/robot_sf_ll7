# Issue #1608 Scenario Seed Sensitivity Analysis (2026-05-30)

Related issue: <https://github.com/ll7/robot_sf_ll7/issues/1608>

## Goal

Classify benchmark scenarios by whether top-planner success depends strongly on the fixed random
seed. This answers the issue #1608 aggregation question without rerunning a benchmark campaign.

## Evidence Surface

The analysis uses the durable compact issue #1454 S10/h500 candidate campaign bundle:

- `docs/context/evidence/issue_1454_s10_h500_candidates_2026-05-23/reports/campaign_table.csv`
- `docs/context/evidence/issue_1454_s10_h500_candidates_2026-05-23/reports/seed_episode_rows.csv`
- provenance note:
  `docs/context/evidence/issue_1454_s10_h500_candidates_2026-05-23/README.md`

The Issue #1454 source run was generated from
`configs/benchmarks/issue_1454_s10_scenario_horizons_h500_candidates.yaml` at commit `4941ac48`.
The raw campaign archive is preserved in the non-package artifact release linked from the Issue #1454 evidence README.

## Method

The reusable entry point is:

```bash
uv run python scripts/tools/analyze_scenario_seed_sensitivity.py \
  --campaign-root docs/context/evidence/issue_1454_s10_h500_candidates_2026-05-23 \
  --output-dir docs/context/evidence/issue_1608_seed_sensitivity_2026-05-30
```

Planner selection is deliberately predeclared and deterministic:

1. keep only `benchmark_success=true` planner rows,
2. rank by `success_mean` descending,
3. break ties by lower `collisions_mean`, lower `time_to_goal_norm_mean`, lower
   `near_misses_mean`, then `planner_key`,
4. select the top four rows.

The selected planners were:

- `hybrid_rule_v3_fast_progress_static_escape_continuous`
- `scenario_adaptive_hybrid_orca_v1`
- `scenario_adaptive_hybrid_orca_v2_collision_guard`
- `hybrid_rule_v3_fast_progress_static_escape`

For each scenario and seed, the analyzer averages success across the selected planners. A scenario
is classified as `seed_sensitive` when all selected planner/seed cells are present, at least three
seeds are available, the across-seed mean-success range is at least `0.5`, and the scenario has at
least one hard seed (`mean_success <= 0.5`) and one easy seed (`mean_success >= 0.75`). Missing rows
or fewer than three seeds make a scenario `inconclusive`.

## Result

The issue #1454 bundle has complete top-four coverage for all 48 scenarios and 10 seeds.

| Classification | Count |
|---|---:|
| `seed_sensitive` | 25 |
| `not_seed_sensitive` | 23 |
| `inconclusive` | 0 |

The hardest seed id across the 48 scenarios was `116`: mean top-planner success `0.7396`, with 12
hard scenario rows. The next hardest seeds were `117`, `111`, `115`, and `112`.

The highest-range seed-sensitive scenarios include:

| Scenario | Mean success | Seed-success range | Hard seeds |
|---|---:|---:|---|
| `classic_cross_trap_high` | 0.7000 | 1.0000 | `111 112 115` |
| `classic_cross_trap_medium` | 0.8000 | 1.0000 | `111 115` |
| `classic_doorway_high` | 0.6750 | 1.0000 | `112 113 117 118` |
| `classic_doorway_medium` | 0.6250 | 1.0000 | `114 115 117 118` |
| `francis2023_join_group` | 0.6000 | 1.0000 | `112 113 115 116` |

Full tables are in:

- `docs/context/evidence/issue_1608_seed_sensitivity_2026-05-30/seed_sensitivity_analysis.md`
- `docs/context/evidence/issue_1608_seed_sensitivity_2026-05-30/scenario_seed_sensitivity.csv`
- `docs/context/evidence/issue_1608_seed_sensitivity_2026-05-30/seed_difficulty_summary.csv`
- `docs/context/evidence/issue_1608_seed_sensitivity_2026-05-30/seed_sensitivity_analysis.json`

## Interpretation Boundary

This is a diagnostic prioritization analysis, not causal mechanism proof. It shows which scenarios
and seeds deserve follow-up inspection, but it does not explain why those seeds are hard. Use
Issue #1609 or a trace-level mechanism review for causal interpretation.

The result also should not be reused as paper-facing significance evidence. The source Issue #1454
campaign is an exploratory candidate surface, and SNQI is not used here because the source bundle
marks SNQI ordering as diagnostic only.

Observation-track metadata boundary: the source Issue #1454 bundle predates explicit
`benchmark_track` and `track_schema_version` metadata. This derived analysis therefore inherits
`legacy_track_unknown` status per
[issue_1721_benchmark_track_metadata_audit.md](issue_1721_benchmark_track_metadata_audit.md). Use it
within the historical S10/h500 surface only; do not aggregate it with track-aware result rows or use
it for cross-observation-track comparisons.

## Validation

- `uv run pytest tests/tools/test_analyze_scenario_seed_sensitivity.py -q`
  - `5 passed`
- `uv run python scripts/tools/analyze_scenario_seed_sensitivity.py --campaign-root docs/context/evidence/issue_1454_s10_h500_candidates_2026-05-23 --output-dir docs/context/evidence/issue_1608_seed_sensitivity_2026-05-30`
  - `25 seed-sensitive, 23 not seed-sensitive, 0 inconclusive`
