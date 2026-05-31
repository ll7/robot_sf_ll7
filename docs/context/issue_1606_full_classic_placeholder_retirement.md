# Issue #1606 Full Classic Placeholder Retirement

Date: 2026-05-30

Related issue: #1606

## Decision Applied

The Full Classic path now uses a compatibility-alias migration for the remaining
`parallel_efficiency_placeholder` manifest key.

The manifest records measured runtime fields:

- `runtime_sec`
- `episodes_per_second`
- `throughput_per_worker`

It does not claim true parallel efficiency without a measured sequential baseline. Until that
baseline exists, `parallel_efficiency` is `"not_available"` and
`parallel_efficiency_basis` records `"requires measured sequential baseline"`.

For one transition, `parallel_efficiency_placeholder` remains as a deprecated compatibility alias
with `parallel_efficiency_placeholder_deprecated: true` and a note that it is not
benchmark-strength evidence.

## Artifact Naming

The lightweight Full Classic diagnostic plot names are data-oriented:

- `path_efficiency.pdf`
- `success_collision_scatter.pdf`
- `episode_lengths.pdf`

The old documented `kde_placeholder.pdf`, `pareto_placeholder.pdf`, and
`force_heatmap_placeholder.pdf` names are retired from user-facing docs.

## Evidence Boundary

Smoke-mode Full Classic outputs are fast diagnostics. They are useful for contract checks and
artifact plumbing, but they are not benchmark-strength evidence. Smoke, fallback, degraded, skipped,
or deprecated compatibility-alias outputs must not be used as nominal benchmark claims.

## Validation

Validation run:

```bash
uv run pytest -q tests/benchmark_full tests/test_full_classic_visuals_basic.py \
  tests/test_full_classic_visuals_edge_cases.py \
  tests/test_full_classic_visuals_missing_matplotlib.py \
  tests/test_full_classic_visuals_renderer_toggle.py
```

Result: 44 passed, 1 known Numba pending-deprecation warning.

The issue success search now returns only intentional compatibility-alias references:

```bash
rg -n "placeholder|parallel_efficiency_placeholder|kde_placeholder|pareto_placeholder|force_heatmap_placeholder|fast placeholders" \
  docs/benchmark_full_classic.md scripts/classic_benchmark_full.py tests/benchmark_full \
  tests/test_full_classic_visuals*.py
```
