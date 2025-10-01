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

## Clamping and Outliers
Clamping to the [0, 1] interval is an intentional design choice for stability and interpretability:

- Values below the baseline median floor at 0 and do not contribute negatively (we do not reward beyond “typical good” by going below 0).
- Values above the upper reference percentile (p95) saturate at 1, preventing rare outliers from dominating the weighted sum and destabilizing optimization.

Implications:

- Tail compression: Episodes with extremely poor values (≫ p95) will receive the same normalized value (1.0) and thus be indistinguishable by that metric alone. Use sensitivity analysis or additional diagnostics if tail ordering matters.
- Under-rewarding extreme excellence: Likewise, exceptionally low adverse metrics (< median) are not given extra credit beyond 0; if you need to distinguish “great” from “good,” consider asymmetric or piecewise scaling (future option).
- Degenerate baselines: When p95 ≈ median, normalization becomes ill-conditioned. Implementations guard this by using a minimum denominator and emit a warning; consider recomputing baselines or switching to IQR/MAD.

Recommended practices:

- Inspect `normalization_comparison` outputs (when enabled) to check correlation vs. the canonical median/p95 transform.
- If many values saturate at 1.0, try a lower upper percentile (e.g., p90) or switch to IQR scaling; document the choice in experiment metadata.
- Keep clamping enabled for benchmark comparability unless a study explicitly motivates a different policy.

## Future Work
- Evaluate alternative robust scalers (IQR, MAD) and adaptive quantiles.
- Explicitly document per-metric directionality and any inversions.
- Provide a configuration file to pin normalization parameters per dataset.

See also:
- Design doc section “Normalization Rationale & Limitations” (`docs/dev/issues/snqi-recomputation/DESIGN.md`).
- User guide overview for a practitioner summary (`docs/snqi-weight-tools/README.md`).
