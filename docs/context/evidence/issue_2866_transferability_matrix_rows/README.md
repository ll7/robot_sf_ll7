# Issue #2866 Transferability Matrix Rows

## Scope

This diagnostic bundle records the Issue #2866 follow-up for `ForecastTransferabilityStressMatrix.v1`
row construction.

## Evidence status

- `schema`: `ForecastTransferabilityStressMatrix.v1`
- `matrix`: [transferability_matrix.json](transferability_matrix.json) and
  [transferability_matrix.md](transferability_matrix.md)
- `claim_boundary`: blocked / diagnostic-only
- `reason`: the tracked bounded inputs expose observation-tier, actor-class, scenario-family,
  horizon, and semantic-metadata splits, but one deployable cell intentionally lacks actor and
  scenario-family metadata. The matrix therefore emits explicit blocked cells and returns
  `decision: stop`.

## Reproducible command

```
uv run python scripts/benchmark/build_forecast_transferability_matrix.py \
  docs/context/evidence/issue_2866_transferability_matrix_rows/metric_oracle_corridor.json \
  docs/context/evidence/issue_2866_transferability_matrix_rows/metric_deployable_motion_rich_bicycle.json \
  docs/context/evidence/issue_2866_transferability_matrix_rows/metric_deployable_unavailable_cells.json \
  --report-id issue_2866_transferability_matrix_rows \
  --generated-at-utc 2026-06-15T00:00:00+00:00 \
  --out-json docs/context/evidence/issue_2866_transferability_matrix_rows/transferability_matrix.json \
  --out-md docs/context/evidence/issue_2866_transferability_matrix_rows/transferability_matrix.md
```

## Validation

```
uv run pytest tests/benchmark/test_forecast_transferability_stress_matrix.py
```

## Claim boundary

The complete deployable fixture cells show that the matrix can preserve row-level metadata, not that
forecast transfer is benchmark-proven. Oracle rows remain diagnostic-only, and blocked cells prevent
any transferability, safety, progress, paper-facing, or dissertation claim.
