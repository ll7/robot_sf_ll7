# Issue #2180 One-Factor h500 Evidence

Status: complete local h500 diagnostic evidence; no failed rows.

This directory preserves the compact summary from the Issue #2180 h500 execution of the
Issue #2170 one-factor hybrid component manifest. Raw JSONL rows, generated candidate reports,
funnel files, and per-candidate summaries remain in ignored `output/` paths.

## Commands

```bash
CMAKE_BUILD_PARALLEL_LEVEL=8 uv sync --extra orca
uv run python -c 'import rvo2; print(rvo2.__name__)'
LOGURU_LEVEL=WARNING TF_CPP_MIN_LOG_LEVEL=2 uv run python scripts/tools/run_one_factor_ablation_pilot.py \
  --comparison-id static_escape_only_minus_base \
  --comparison-id static_recenter_only_minus_base \
  --comparison-id escape_recenter_pair_minus_static_escape_only \
  --comparison-id grouped_transit_minus_escape_recenter_pair \
  --comparison-id continuous_checks_minus_grouped_static \
  --comparison-id selector_only_minus_grouped_static \
  --comparison-id speed_progress_2p4_minus_base \
  --horizon 500 --workers 2 \
  --output-dir output/issue_2180/one_factor_h500_w2
```

## Result

All eight candidates completed 18/18 jobs with zero failed jobs in native local mode.

| Comparison | Success delta | Collision delta | Near-miss delta | Avg-speed delta | Runtime delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| static_escape_only_minus_base | 0.000 | 0.000 | 0.000 | 0.000 | +27.690s |
| static_recenter_only_minus_base | +0.056 | 0.000 | 0.000 | +0.075 | +17.418s |
| escape_recenter_pair_minus_static_escape_only | +0.111 | 0.000 | 0.000 | +0.116 | -12.234s |
| grouped_transit_minus_escape_recenter_pair | 0.000 | 0.000 | 0.000 | -0.000 | -0.128s |
| continuous_checks_minus_grouped_static | -0.111 | 0.000 | -0.222 | -0.071 | +11.542s |
| selector_only_minus_grouped_static | -0.056 | 0.000 | 0.000 | -0.057 | -10.289s |
| speed_progress_2p4_minus_base | -0.056 | 0.000 | +0.111 | -0.004 | +5.369s |

The strongest positive h500 signal is recentering: static recentering improves success by one row
over base, and adding recentering after static escape improves success by two rows over static
escape alone. Static escape alone is flat. Corridor-transit terms are flat after escape plus
recenter. Continuous static checks reduce near-miss rate but lose two successes versus grouped
static. Selector-only and speed/progress-2.4 are weaker than their comparators on this slice.

Treat this as local h500 diagnostic evidence for component direction. It is not a planner-promotion
claim by itself.
