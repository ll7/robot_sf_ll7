# Issue #4941 — Seed Distribution Report for Statistical Robustness

## Summary

Adds `seed_distribution_report.v1` — a unified reporting surface that normalizes multi-seed benchmark outputs into one schema for distributional evidence, confidence intervals, and seed/surface instability diagnostics.

## New files

| File | Purpose |
|------|---------|
| `robot_sf/benchmark/seed_distribution_report.py` | Schema, dataclass model, adapters, and builder |
| `scripts/benchmark/build_seed_distribution_report.py` | CLI entry point for report generation |
| `tests/benchmark/test_seed_distribution_report.py` | Fixture-backed tests (41 tests) |

## Schema

`SEED_DISTRIBUTION_REPORT_SCHEMA_VERSION = "seed_distribution_report.v1"`

### Key dataclasses

- `MetricSummary` — per-metric distributional summary (mean, std, cv, CI bounds)
- `RawCounts` — optional numerator/denominator for discrete outcomes
- `SeedDistributionDiagnostics` — machine-readable flags (insufficient seeds, wide CI, unstable rank, advisory-only)
- `IntervalEstimate` — confidence interval metadata
- `PerSeedValue` — single per-seed observation
- `SurfaceRecord` — normalized surface (planner x scenario x metric)
- `SourceProvenance` — report-level provenance
- `SeedDistributionReport` — top-level payload

### Adapters

- `adapt_seed_variability_report()` — consumes `benchmark-seed-variability-by-scenario.v1` payloads
- `adapt_rank_stability_report()` — consumes `issue_3216_headline_ci_rank_stability.v1` payloads

## Diagnostics

| Flag | Meaning | Default threshold |
|------|---------|-------------------|
| `insufficient_seed_count` | Too few seeds for reliable estimation | < 3 seeds |
| `wide_interval` | CI half-width exceeds threshold | > 0.15 |
| `unstable_rank` | Rank-flip rate exceeds threshold | > 0.3 |
| `advisory_only` | Set when insufficient_seed_count is true | — |

## Claim boundary

> Existing multi-seed benchmark outputs can be normalized into a common, auditable statistical-robustness report.

Does NOT support: scenario catalog completeness, simulator calibration, external validity from narrow CIs, or publication readiness without evidence review.

## Validation

```bash
uv run pytest tests/benchmark/test_seed_distribution_report.py -q
uv run python scripts/benchmark/build_seed_distribution_report.py --help
```

## Cross-links

- #4933 (virtual validation mitigation feasibility)
- PR #4934 (feasibility assessment recommending statistical robustness)
