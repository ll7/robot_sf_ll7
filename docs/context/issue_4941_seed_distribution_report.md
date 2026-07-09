# Seed Distribution Report (Issue #4941)

## What this report standardizes

The `seed_distribution_report.v1` schema normalizes multi-seed benchmark evidence from
different report surfaces into a common, auditable format. It lets benchmark consumers
compare seed-level outcomes across campaigns, planners, and scenarios without re-running
simulations.

## Supported inputs

The builder searches for supported artifacts in the campaign root **and** its
`reports/` subdirectory (the layout real campaigns use). It currently adapts
three existing report surfaces, reading their real on-disk shapes:

1. **`seed_variability_by_scenario.json`** (camera-ready seed-variability export)
   Provides per-scenario/planner `summary` stat dicts (`mean`, `std`, `ci_low`,
   `ci_high`, `ci_half_width`) and the per-seed means list from `per_seed`.
   Schema version `benchmark-seed-variability-by-scenario.v1`.
2. **`statistical_sufficiency.json`** (seed-sufficiency gate output)
   Provides per-metric CI half-widths (`ci_half_width`) and sufficiency status
   per scenario/planner. This surface reports precision, not point estimates, so
   `point_estimate` is null and the interval records `half_width`. Schema version
   `benchmark-seed-statistical-sufficiency.v1`.
3. **headline CI rank-stability report** (`result.json` or
   `issue_3216_headline_ci_rank_stability.json`)
   Provides a flat `cells` list; each cell carries per-metric
   `mean`/`ci_low`/`ci_high` and a `counted` flag. Cells excluded from the
   headline ranking (`counted == false`) are retained and flagged `advisory_only`.
   The generic `result.json` name is accepted only when its `schema_version`
   starts with `issue_3216`, so an unrelated `result.json` is ignored.

## Diagnostic fields

Each surface entry includes a `diagnostics` block with boolean flags:

- **`insufficient_seed_count`**: `true` when `seed_count < 10`. Low seed counts make
  point estimates unreliable for comparison. Treat these as preliminary evidence only.
- **`unstable_rank`**: `true` when rank ordering changed across seed resamples.
  Today's source artifacts do not emit per-surface rank drift, so this field is
  null (collapsed to `false`) unless a future/source row provides
  `rank_changed_across_seeds` (seed-variability) or `rank_unstable` (CI rank cell).
  Campaign-level rank stability remains available in the CI rank report's
  `rank_stability` / `decision_packet` blocks.
- **`wide_interval`**: `true` when the confidence interval width exceeds a
  metric-specific threshold. Thresholds are tuned per metric type (e.g., 0.15 for
  success, 0.10 for collisions); for sufficiency surfaces the width is taken as
  `2 * ci_half_width`. Wide intervals indicate high uncertainty in the
  point estimate.
- **`advisory_only`**: `true` when the source surface reported its evidence as
  insufficient or was excluded from the headline ranking. These entries should not
  be cited as definitive benchmark evidence.

## Evidence boundary

A normalized `seed_distribution_report.v1` supports the claim:

> Existing multi-seed benchmark outputs can be normalized into a common, auditable
> statistical-robustness report.

It does **not** support claims that:

- the benchmark scenario catalog is complete;
- the simulator is calibrated to real-world behavior;
- narrower confidence intervals imply external validity;
- a single campaign is publication-ready without its original evidence review.

Statistical repeatability is not evidence of simulator fidelity, scenario coverage,
or real-world validity. This report is analysis tooling, not a benchmark result.

## Adding new adapters

Future campaign types should add adapters without changing the v1 schema. An adapter
reads one input format and returns a list of surface dictionaries matching the v1 schema
contract. Steps:

1. Add an `_adapt_<format>(path: Path) -> list[dict[str, Any]]` function in
   `seed_distribution_report.py`.
2. Register it in the adapter loop inside `build_seed_distribution_report()`.
3. Add fixture-backed tests covering the new adapter's stable and edge cases.

Nullable fields in the schema (e.g., `scenario_id`, `raw_counts`, `seed_distribution`)
accommodate incompatible input shapes without silent data loss. Use `None` when a field
cannot be populated, and set the appropriate diagnostic flag.

## Related issues

- #4932: rare-event / accelerated stress testing
- #4933, #4934: referenced by the acceptance checklist
