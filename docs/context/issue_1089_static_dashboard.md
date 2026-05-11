# Issue 1089: Static Benchmark Dashboard

Issue: <https://github.com/ll7/robot_sf_ll7/issues/1089>

## Decision

The initial dashboard generator supports the camera-ready campaign bundle contract:
`reports/campaign_summary.json` with `campaign`, `planner_rows`, `warnings`, and `artifacts`.
This keeps the first version deterministic and self-contained without introducing a JavaScript build
pipeline or backend service.

The generated site contains one summary page, one per-planner page per row, local CSS, normalized
JSON data, compact copied report downloads, and a manifest. It deliberately does not copy raw
episode JSONL or videos by default because those can be large and should remain in the source bundle
or publication archive.

## Artifacts

- Generator: `scripts/tools/generate_benchmark_dashboard.py`
- Tests: `tests/tools/test_generate_benchmark_dashboard.py`
- User docs: `docs/benchmark_static_dashboard.md`

Generated dashboards under `output/benchmarks/dashboards/` are reproducible local artifacts and
should remain ignored unless a compact evidence snapshot is deliberately promoted.
