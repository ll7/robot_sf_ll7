# Issue #2903 Horizon x Timestep Denominator Health Report

## Scope

This analysis-only bundle audits the denominator coverage of the horizon x
timestep ablation from issue #2837 / PR #2887. It classifies each missing
(horizon, dt_s, trace) cell by reason, verifies the category totals sum to the
expected 180-cell matrix, spot-checks representative missing cells, and proposes
the minimum fixture additions needed to evaluate at least 90% of the matrix.

## Evidence status

- `schema`: `HorizonTimestepDenominatorHealth.v1`
- `claim_boundary`: `analysis_only_not_navigation_evidence`
- `forecast_defaults_unchanged`: `true`
- `reason`: the report only diagnoses why ablation cells are unevaluable on the
  current durable fixture set. It does not change forecast defaults, simulator
  physics, or claim navigation / closed-loop benefit.

## Files

- [denominator_report.json](denominator_report.json): full denominator-health data, category totals,
  matrix coverage, per-family missingness, spot checks, and fixture proposal
- [denominator_report.md](denominator_report.md): human-readable Markdown report
- [summary.json](summary.json): compact issue coverage summary

## Reproducible command

```bash
uv run python scripts/benchmark/build_horizon_timestep_denominator_report.py \
  --issue 2903 \
  --parent-issue 2837 \
  --generated-at-utc 2026-06-16T00:00:00+00:00 \
  --output-dir docs/context/evidence/issue_2903_horizon_denominator_health_2026-06-16
```

## Validation

```bash
uv run pytest tests/benchmark/test_horizon_timestep_denominator.py
uv run ruff check scripts/benchmark/build_horizon_timestep_denominator_report.py
uv run ruff format scripts/benchmark/build_horizon_timestep_denominator_report.py
```

## Claim boundary

This report is useful for planning fixture additions to improve ablation
denominator coverage. It does not prove navigation improvement, safety, human
realism, benchmark-strength predictor quality, or closed-loop planner benefit.
Forecast defaults must not change based on this report alone.
