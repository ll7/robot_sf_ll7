# Seed Distribution Report (Issue #4941)

## What this report standardizes

The `seed_distribution_report.v1` schema normalizes multi-seed benchmark evidence from
different report surfaces into a common, auditable format. It lets benchmark consumers
compare seed-level outcomes across campaigns, planners, and scenarios without re-running
simulations.

## Supported inputs

The builder currently adapts three existing report surfaces:

1. **seed_variability.json** (from camera-ready campaign)
   Provides per-scenario/planner seed means, across-seed CI, and raw counts.
2. **statistical_sufficiency.json** (from campaign sufficiency rows)
   Provides CI half-widths and sufficiency status per scenario/planner/metric.
3. **issue_3216_headline_ci_rank_stability.json** (from CI rank stability report)
   Provides bootstrap CI and rank-unstable flags per scenario/planner/metric.

## Diagnostic fields

Each surface entry includes a `diagnostics` block with boolean flags:

- **`insufficient_seed_count`**: `true` when `seed_count < 10`. Low seed counts make
  point estimates unreliable for comparison. Treat these as preliminary evidence only.
- **`unstable_rank`**: `true` when rank ordering changed across seed resamples.
  This means the scenario winner or planner ordering is not reproducible under the
  current seed budget.
- **`wide_interval`**: `true` when the confidence interval width exceeds a
  metric-specific threshold. Thresholds are tuned per metric type (e.g., 0.15 for
  success, 0.10 for collisions). Wide intervals indicate high uncertainty in the
  point estimate.
- **`advisory_only`**: `true` when the source surface reported its evidence as
  insufficient or diagnostic-only. These entries should not be cited as definitive
  benchmark evidence.

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
