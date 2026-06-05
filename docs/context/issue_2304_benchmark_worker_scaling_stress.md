# Issue #2304 Stress-Slice Worker Scaling Diagnostic

Status: current, diagnostic-only.

Issue #2304 followed the Issue #2302 worker-scaling continuation with a heavier local slice. The
question was whether worker counts above two still help when the policy-search workload moves from
the h80 nominal-sanity slice to the h120 `stress_slice` stage.

## Evidence

Profile command shape:

```bash
LOGURU_LEVEL=WARNING TF_CPP_MIN_LOG_LEVEL=2 uv run python scripts/tools/profile_benchmark_worker_scaling.py \
  --candidate hybrid_rule_v3_fast_progress --stage stress_slice --workers 1 2 4 6
```

Two local repeats ran on commit `67fdb561d825267bc099f3fa32284875448407a7`. The slice contained
24 jobs per repeat: 8 stress scenarios times 3 seeds, using the funnel stage's configured 120-step
horizon. Compact evidence is tracked in
`docs/context/evidence/issue_2304_benchmark_worker_scaling_stress_2026-06-05/summary.json`.

Warnings observed in both repeats:

- `map_search_paths` included a missing `configs/maps/svg_maps` entry.
- `uni_campus_big.svg` reported a self-intersection in `obstacle_13`.

Neither warning failed the diagnostic runs. All worker counts completed with status `ok`. The
stage decision was `tracked`, which is expected because `stress_slice` has no pass/revise gate in
the policy-search funnel.

## Median Timing Result

| Workers | Jobs | Episodes | Runtime sec median | Sec/job median | Speedup vs 1 worker |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 24 | 24 | 203.681 | 8.487 | 1.000 |
| 2 | 24 | 24 | 109.006 | 4.542 | 1.869 |
| 4 | 24 | 24 | 61.414 | 2.559 | 3.317 |
| 6 | 24 | 24 | 45.503 | 1.896 | 4.476 |

Median runtime fell 46.5% from 1 to 2 workers, 43.7% from 2 to 4 workers, and 25.9% from 4 to 6
workers. The 6-worker leg saved about 15.9 seconds versus 4 workers on this local 24-job slice.

## Interpretation

Observed evidence: the heavier stress slice still scaled cleanly through six workers. No worker
count failed, and no OOM or worker instability was observed in the command output.

Claim boundary: this is runtime-diagnostic evidence only. It does not strengthen planner-quality
claims, and it does not prove that six workers should become a cross-host or CI default.

Recommendation: use six workers for local stress-slice policy-search diagnostics when the local
machine has CPU headroom. Keep global stage defaults unchanged until there is cross-host or
CI-resource evidence; for conservative local profiling where headroom matters, four workers remains
a useful lower-risk baseline.
