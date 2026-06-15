# Issue #2868 Semantic Metadata Fixtures

## Scope

This diagnostic bundle records the Issue #2868 follow-up for metadata-bearing forecast trace
fixtures and the semantic baseline comparison they enable.

## Evidence status

- `schema`: `simulation_trace_export.v1` fixtures plus forecast comparison report
- `report`: [comparison_report.json](comparison_report.json) and
  [comparison_report.md](comparison_report.md)
- `claim_boundary`: diagnostic-only / nominal fixture evidence
- `reason`: the tracked fixtures expose signal and intent metadata consumed by signal-aware,
  goal-aware, semantic, and interaction-aware forecast baselines, while legacy rows remain
  metadata-absent for comparison.

## Reproducible command

```
uv run python scripts/benchmark/run_cv_forecast_eval.py \
  --output-dir docs/context/evidence/issue_2868_semantic_metadata_fixtures_2026-06-15 \
  --compare-all \
  --issue 2868 \
  --generated-at-utc 2026-06-15T00:00:00+00:00
```

## Validation

```
uv run pytest tests/benchmark/test_pedestrian_forecast_cv_eval.py tests/benchmark/test_pedestrian_forecast.py
```

## Claim boundary

The four Issue #2868 fixtures demonstrate that semantic metadata is present and consumed by the
baseline comparison path. They do not establish human realism, closed-loop navigation improvement,
or benchmark-strength semantic improvement claims.
