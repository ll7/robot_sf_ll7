# Issue #2125 Seed Sufficiency And Ranking Stability

Related issue: <https://github.com/ll7/robot_sf_ll7/issues/2125>

## Scope

Issue #2125 adds a diagnostic analyzer for existing seed-level camera-ready benchmark report
bundles. It does not rerun benchmark campaigns, change seed schedules, or promote small-seed
surfaces into benchmark-strength evidence.

Canonical command:

```bash
uv run python scripts/tools/analyze_seed_sufficiency.py \
  --campaign-root output/benchmarks/camera_ready/<s3_campaign_id> \
  --campaign-root output/benchmarks/camera_ready/<s5_campaign_id> \
  --output-dir output/benchmarks/camera_ready/<analysis_id>/reports/seed_sufficiency
```

The output directory receives:

- `seed_sufficiency_analysis.json`
- `seed_count_interval_width.csv`
- `planner_rank_stability.csv`
- `scenario_family_instability.csv`
- `outcome_counts.csv`
- `seed_sufficiency_summary.md`
- `fig_seed_interval_width.png`

## Input Contract

Each `--campaign-root` must contain the issue #595 seed-variability bundle:

- `reports/seed_variability_by_scenario.json`
- `reports/seed_episode_rows.csv`
- optional `reports/statistical_sufficiency.json`

The analyzer reads existing bootstrap confidence intervals from
`seed_variability_by_scenario.json`. It does not resample. Planner ranks and scenario-family
winners use deterministic sorting so repeated runs over the same inputs produce stable CSV and JSON
ordering.

## Interpretation Boundary

Use the analyzer to answer diagnostic questions:

- How quickly do recorded interval widths narrow as seed schedules expand?
- Did planner-level ranking change between available schedules?
- Did scenario-family winners change between available schedules?
- Which single-seed or incomplete-seed surfaces should be treated as advisory?

Do not use it to claim scenario coverage, external validity, statistical significance, or paper-ready
planner superiority. Single-seed and incomplete-seed campaign roots are marked advisory. Rank flips
and scenario-family winner changes are caveats that should narrow the claim or trigger a larger
pre-specified seed schedule, not be averaged away.

The raw `success_count`, `collision_count`, and `episode_count` rows are preserved in
`outcome_counts.csv` so downstream reports can inspect numerator and denominator evidence for
binary safety outcomes instead of relying only on percentages.

## Related Context

- [issue_595_seed_variability_contract.md](issue_595_seed_variability_contract.md) defines the
  seed-variability artifact contract.
- [issue_832_paper_matrix_extended_seed_schedule.md](issue_832_paper_matrix_extended_seed_schedule.md)
  documents the S3/S5/S10/S20 schedule policy and pairwise comparison thresholds.
- [issue_1545_power_aware_seed_budget_planning.md](issue_1545_power_aware_seed_budget_planning.md)
  explains seed-budget tiers and why low-seed ranking claims remain fragile.
- [issue_691_benchmark_fallback_policy.md](issue_691_benchmark_fallback_policy.md) remains the
  fail-closed benchmark policy boundary.

## Validation

Initial implementation validation:

```bash
uv run ruff check scripts/tools/analyze_seed_sufficiency.py tests/tools/test_analyze_seed_sufficiency.py
uv run pytest tests/tools/test_analyze_seed_sufficiency.py -q
```

