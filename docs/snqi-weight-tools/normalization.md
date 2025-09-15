# SNQI Normalization Strategy

This note explains the current normalization approach for SNQI component metrics and its implications.

## Summary
- Per-metric normalization uses robust statistics (median/p95) to map values to approximately [0, 1].
- Post-normalization clamping ensures values lie within [0, 1].
- This supports comparability across heterogenous metrics but can hide extreme outliers.

## Details
- For each metric m, raw values are transformed using a robust scale (e.g., (x - median) / (p95 - median)), with sign conventions applied so that higher is better.
- After scaling, values are clamped to [0, 1] to avoid undue influence from extreme values.
- Some metrics may be optional; missing metrics are either skipped or assigned neutral contributions depending on flags.

## Limitations
- Clamping can reduce sensitivity to extreme poor performance; document when this is acceptable.
- Choice of median/p95 affects discriminative power; datasets with heavy tails may need alternative quantiles.
- Scaling assumes monotonic preference; metrics with non-monotonic optima require special handling.

## Future Work
- Evaluate alternative robust scalers (IQR, MAD) and adaptive quantiles.
- Explicitly document per-metric directionality and any inversions.
- Provide a configuration file to pin normalization parameters per dataset.
