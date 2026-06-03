# Issue #2172 Benchmark Worker Scaling Diagnostic

Status: current, diagnostic-only.

Issue #2172 asked for a simulator-speed fallback lane after the research queue was exhausted. This
slice adds a repeatable profiling helper and records one local worker-scaling result for the
policy-search candidate runner.

## What Changed

- Added `workers`, `parallel_execution`, and `batch_runtime_sec` to the `run_map_batch` summary so
  candidate-runner `summary.json` files identify the worker count and batch wall time directly.
- Added `scripts/tools/profile_benchmark_worker_scaling.py`, a thin wrapper around
  `scripts/validation/run_policy_search_candidate.py` that runs one candidate/stage across worker
  counts and writes a compact comparison JSON.
- Promoted compact evidence under
  `docs/context/evidence/issue_2172_benchmark_worker_scaling_2026-06-03/`; raw JSONL outputs remain
  ignored under `output/`.

## Diagnostic Result

Command:

```bash
LOGURU_LEVEL=WARNING TF_CPP_MIN_LOG_LEVEL=2 uv run python scripts/tools/profile_benchmark_worker_scaling.py \
  --candidate hybrid_rule_v3_fast_progress --stage nominal_sanity --workers 1 2 --horizon 80 \
  --output-root output/issue_2172/helper_nominal_h80
```

Result on commit `2159fd43d119d764259814d8f9cbac388fa9a0ca`:

| Workers | Jobs | Episodes | Runtime sec | Sec/job | Speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 18 | 18 | 86.314 | 4.795 | 1.000 |
| 2 | 18 | 18 | 48.140 | 2.674 | 1.793 |

Interpretation: this local 18-job h80 nominal-sanity slice showed useful parallel speedup, while
the one-episode smoke slice was dominated by startup overhead and was not a meaningful scaling
probe. The candidate gate decision stayed `revise`; this is runtime-diagnostic evidence only.

## Follow-Up Direction

The next smallest useful optimization question is whether worker counts above two improve
throughput enough to justify their CPU and memory cost on pedestrian-heavy slices. Use repeated
profiles and report median runtime before changing benchmark defaults.
