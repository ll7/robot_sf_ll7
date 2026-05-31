# Issue #1845 Report Grouping Contracts

## Goal

Issue #1845 centralizes the legacy-compatible grouping contract used by benchmark report surfaces.
The implementation keeps paper/benchmark aggregation fail-closed while making human-facing reports
share the same fallback order for legacy rows:

1. `scenario_params.algo`
2. top-level `algo`
3. `algorithm_metadata.algorithm`
4. fallback grouping key, usually `scenario_id`

## Decision

The shared helper lives in `robot_sf/benchmark/grouping.py`:

* `resolve_report_group_key(...)` applies the common report fallback chain.
* `DEFAULT_REPORT_GROUP_BY` and `DEFAULT_REPORT_FALLBACK_GROUP_BY` keep CLI defaults aligned.
* `EFFECTIVE_REPORT_GROUP_KEY` documents the effective report key for metadata payloads.

`compute_aggregates(...)` still uses its stricter aggregation resolver for benchmark-strength
summaries, because silently grouping by `scenario_id` when algorithm metadata is missing can
invalidate planner comparisons. Tables, rankings, distributions, Pareto plots, SNQI ablations, and
seed-variance summaries use the shared report helper with their historical missing-row policy:

* skip missing rows for table/ranking/distribution/Pareto surfaces,
* keep an explicit `unknown` group for SNQI ablation and seed variance.

## Validation

Focused validation command:

```bash
uv run pytest tests/benchmark/test_report_grouping_contracts.py tests/test_aggregate.py tests/test_cli_table.py tests/test_ranking.py tests/benchmark/test_distributions.py tests/test_plots_pareto.py tests/test_snqi_ablation.py tests/test_seed_variance.py -q
```

Observed result on the issue branch: `42 passed`.

The regression coverage verifies:

* top-level `algo` and `algorithm_metadata.algorithm` are used before scenario fallback,
* missing group policy is explicit (`skip`, `unknown`, or `error`),
* legacy rows with only `algorithm_metadata.algorithm` group consistently across aggregate, table,
  ranking, distribution, Pareto, SNQI ablation, and seed variance,
* fully missing legacy rows still land in the `unknown` bucket for the surfaces that historically
  preserve such rows.
