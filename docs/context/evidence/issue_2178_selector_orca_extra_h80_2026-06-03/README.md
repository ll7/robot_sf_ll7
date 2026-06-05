# Issue #2178 Selector ORCA-Extra h80 Evidence

Status: diagnostic-only local evidence; complete 18/18 selector rerun.

This directory preserves the compact summary from the Issue #2178 rerun of the partial Issue #2176
selector-only comparison. The worktree was synced with the `orca` extra, and `rvo2` import was
proven before running the comparison.

## Commands

```bash
CMAKE_BUILD_PARALLEL_LEVEL=8 uv sync --extra orca
uv run python -c 'import rvo2; print(rvo2.__name__)'
LOGURU_LEVEL=WARNING TF_CPP_MIN_LOG_LEVEL=2 uv run python scripts/tools/run_one_factor_ablation_pilot.py \
  --comparison-id selector_only_minus_grouped_static --horizon 80 --workers 2 \
  --output-dir output/issue_2178/selector_h80_w2
```

## Result

| Comparison | Status | Success delta | Collision delta | Near-miss delta | Avg-speed delta | Runtime delta |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| selector_only_minus_grouped_static | ok | 0.000 | 0.000 | 0.000 | -0.096 | -9.865s |

Both candidates wrote 18/18 rows with zero failed jobs. The corrected selector-only h80 row removes
the denominator caveat from Issue #2176. It still does not move headline outcome rates at h80; the
only observed deltas are lower average speed and lower local runtime versus the grouped static
comparator.

Treat this as a setup correction plus diagnostic h80 evidence only. It is not h500 benchmark
evidence and does not support a planner-promotion or component-causality claim by itself.
