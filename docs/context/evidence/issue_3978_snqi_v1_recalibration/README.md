# Issue #3978 SNQI-v1 Recalibration Evidence

Plain-language summary: this bundle records a small synthetic before/after decomposition for the versioned SNQI-v1 diagnostic. It is not planner evidence, benchmark evidence, or a canonical weight decision.

## Claim Boundary

- `SNQI-v0` remains the default score for historical comparability and preserves the legacy mixed raw/baseline-normalized basis.
- `SNQI-v1` is an opt-in bounded baseline-relative diagnostic.
- `SNQI-v1` scores are not numerically comparable to `SNQI-v0` scores because the normalization semantics changed.
- Constraints-first benchmark evidence remains primary.
- No weights were retuned and no new benchmark run was submitted for this bundle.

## Fixture

The fixture uses one synthetic row with unit weights and baseline median/p95 values chosen to make all SNQI-v1 penalty terms clamp to `1.0`.

- `fixture_metrics.json`: synthetic metric row.
- `fixture_weights_unity.json`: unit weights.
- `fixture_baseline_stats_v1.json`: median/p95 coverage for all SNQI-v1 penalty metrics.
- `snqi_v0_v1_decomposition.json`: score and contribution comparison.

## Reproduction

```bash
uv run pytest tests/test_snqi_normalization_comparison.py \
  tests/benchmark/test_snqi_normalization_inventory.py \
  tests/benchmark/test_snqi_governance_preflight.py -q
```

The fixture values are also covered by `test_compute_snqi_default_preserves_v0_score`,
`test_compute_snqi_v1_uses_bounded_comparable_penalty_terms`, and
`test_snqi_v1_contribution_diagnostics_are_comparable`.
