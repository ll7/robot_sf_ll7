# Issue #2172 Benchmark Worker Scaling Evidence

Status: diagnostic-only local runtime evidence.

This directory contains the compact comparison promoted from a local worker-scaling profile. Raw
episode JSONL files and per-run summaries remain under ignored `output/` paths.

## Command

```bash
LOGURU_LEVEL=WARNING TF_CPP_MIN_LOG_LEVEL=2 uv run python scripts/tools/profile_benchmark_worker_scaling.py \
  --candidate hybrid_rule_v3_fast_progress --stage nominal_sanity --workers 1 2 --horizon 80 \
  --output-root output/issue_2172/helper_nominal_h80
```

## Result

| Workers | Jobs | Episodes | Runtime sec | Sec/job | Speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 18 | 18 | 86.314 | 4.795 | 1.000 |
| 2 | 18 | 18 | 48.140 | 2.674 | 1.793 |

The planner gate decision was `revise` for both rows. This evidence only supports the simulator
runtime direction: the same local slice ran faster with two workers. It is not a planner-quality,
promotion, paper-facing, or benchmark-strength claim.
