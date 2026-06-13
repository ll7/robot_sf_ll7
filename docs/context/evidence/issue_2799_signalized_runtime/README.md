# Issue #2799 Signalized Crossing Runtime Evidence (2026-06-13)

This bundle preserves simulator-backed runtime evidence for signalized-crossing metric
denominator and exclusion semantics.

## Reproduction

```bash
uv run python scripts/tools/run_camera_ready_benchmark.py --config configs/benchmarks/signalized_runtime_smoke_issue_2799.yaml --output-root output/benchmarks/issue_2799_signalized_runtime --label issue_2799_signalized_runtime_red --skip-publication-bundle
```

Then regenerate the bundle with:

```bash
uv run python scripts/tools/generate_signalized_runtime_metrics_report.py \
  --episodes-jsonl <campaign-root>/runs/goal__differential_drive/episodes.jsonl
```

## Files

- [summary.json](summary.json): machine-readable runtime row summary.
- [report.md](report.md): human-readable runtime row table and interpretation.
