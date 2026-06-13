# Issue #2753 Signalized Crossing Metrics Report-Row Fixture Evidence 2026-06-13

**Date:** 2026-06-13
**Commit:** (generated, see manifest)

## Scope

Fixture/report-table evidence for the four canonical signalized crossing
metric row types defined in `robot_sf/benchmark/signal_metrics.py`:

| Row type | planner_observable | benchmark_evidence | denominator | compliance_eligible |
|---|---|---|---|---|
| `red_required_stop` | true | true | 1 | true |
| `green_proceed` | true | true | 1 | true |
| `unavailable_no_claim` | false | false | 0 | false |
| `proxy_only_denominator_excluded` | false | false | 0 | false |

## Claim Boundary

This is fixture/report-table evidence only.  The script constructs
synthetic episodes and feeds them through `signal_metrics_report_rows`
to verify the report-row contract.  No simulator or runtime traces were
executed, so no compliance claim is established beyond the report-row
structure. Simulator-backed runtime evidence is deferred to issue #2799.

## Files

- `summary.json`: machine-readable fixture summary
- `report.md`: rendered Markdown report table
- `README.md`: this file

## Reproduction

```bash
uv run python scripts/tools/generate_signalized_crossing_metrics_report.py
```
