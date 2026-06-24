# Issue #3544 Signalized Runtime Metrics

This directory preserves the compact runtime-metrics view used by the issue #3544 failure-pack
negative control. The source run is a live-execution signalized smoke run with simulation-step
trace recording enabled.

## Reproduction

```bash
uv run python scripts/tools/generate_signalized_runtime_metrics_report.py \
  --episodes-jsonl output/benchmarks/issue_3544_signalized_trace_run/episodes.jsonl \
  --output-dir docs/context/evidence/issue_3544_signalized_failure_pack_real/runtime_metrics
```

## Files

- [summary.json](summary.json): machine-readable runtime row summary.
- [report.md](report.md): human-readable runtime row table and interpretation.
