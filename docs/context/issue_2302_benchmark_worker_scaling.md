# Issue #2302 Benchmark Worker Scaling Continuation

Status: current, diagnostic-only.

Issue #2302 continued the worker-scaling lane opened by Issue #2172. The goal was to test whether
policy-search benchmark throughput keeps improving above two workers before spending effort on a
default change or deeper simulator-speed optimization.

## Evidence

Profile command shape:

```bash
LOGURU_LEVEL=WARNING TF_CPP_MIN_LOG_LEVEL=2 uv run python scripts/tools/profile_benchmark_worker_scaling.py \
  --candidate hybrid_rule_v3_fast_progress --stage nominal_sanity --workers 1 2 4 6 --horizon 80
```

Two local repeats ran on commit `82d7819f11e3703b0c511988af99aa983098f76f`. Compact evidence is
tracked in
`docs/context/evidence/issue_2302_benchmark_worker_scaling_2026-06-05/summary.json`.

Warnings observed in both repeats:

- `map_search_paths` included a missing `configs/maps/svg_maps` entry.
- `uni_campus_big.svg` reported a self-intersection in `obstacle_13`.

Neither warning failed the diagnostic runs. All candidate rows completed with status `ok`, and all
candidate decisions remained `revise`.

## Median Timing Result

| Workers | Jobs | Episodes | Runtime sec median | Sec/job median | Speedup vs 1 worker |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 18 | 18 | 86.492 | 4.805 | 1.000 |
| 2 | 18 | 18 | 48.273 | 2.682 | 1.792 |
| 4 | 18 | 18 | 28.980 | 1.610 | 2.985 |
| 6 | 18 | 18 | 23.654 | 1.314 | 3.657 |

The median runtime reduction was 44.2% from 1 to 2 workers, 40.0% from 2 to 4 workers, and 18.4%
from 4 to 6 workers. The marginal return was smaller beyond four workers, but six workers still
cut median runtime by about 5.3 seconds for this 18-job local slice.

## Interpretation

Observed evidence: this repeated h80 nominal-sanity diagnostic shows useful local throughput
improvement above two workers through six workers. No worker count failed, and there was no observed
OOM or worker instability.

Claim boundary: this is runtime-diagnostic evidence only. The candidate result stayed `revise`, so
the run does not strengthen a planner-quality claim. It also does not prove that six workers should
become a repository default on every host.

Recommendation: use four workers as the conservative next profiling baseline when local CPU budget
allows, and use six workers for local diagnostic batches when wall time is more important than CPU
headroom. Before changing benchmark defaults, collect a heavier or pedestrian-dense profile and
confirm that the additional workers do not create memory pressure or CI-host contention.
