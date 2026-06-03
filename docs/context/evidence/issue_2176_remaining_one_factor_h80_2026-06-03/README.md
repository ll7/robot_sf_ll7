# Issue #2176 Remaining One-Factor h80 Evidence

Status: diagnostic-only local evidence; one selector row is partial.

This directory preserves the compact summary promoted from the remaining h80 comparisons in the
Issue #2170 one-factor hybrid component manifest. Raw JSONL rows, generated candidate reports,
temporary funnels, and seed files remain in ignored `output/` paths and are reproducible from the
command, commit, and tracked configs.

## Command

```bash
LOGURU_LEVEL=WARNING TF_CPP_MIN_LOG_LEVEL=2 uv run python scripts/tools/run_one_factor_ablation_pilot.py \
  --comparison-id static_recenter_only_minus_base \
  --comparison-id escape_recenter_pair_minus_static_escape_only \
  --comparison-id grouped_transit_minus_escape_recenter_pair \
  --comparison-id continuous_checks_minus_grouped_static \
  --comparison-id selector_only_minus_grouped_static \
  --comparison-id speed_progress_2p4_minus_base \
  --horizon 80 --workers 2 \
  --output-dir output/issue_2176/remaining_h80_w2
```

## Result

| Comparison | Status | Success delta | Collision delta | Near-miss delta | Avg-speed delta | Runtime delta |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| static_recenter_only_minus_base | ok | 0.000 | 0.000 | 0.000 | +0.090 | +16.263s |
| escape_recenter_pair_minus_static_escape_only | ok | 0.000 | 0.000 | 0.000 | +0.090 | +0.989s |
| grouped_transit_minus_escape_recenter_pair | ok | 0.000 | 0.000 | 0.000 | 0.000 | -0.120s |
| continuous_checks_minus_grouped_static | ok | 0.000 | 0.000 | 0.000 | -0.001 | +1.660s |
| selector_only_minus_grouped_static | partial | 0.000 | +0.022 | -0.144 | -0.041 | -12.594s |
| speed_progress_2p4_minus_base | ok | 0.000 | 0.000 | 0.000 | +0.011 | -0.099s |

Seven candidate rows wrote 18/18 jobs with zero failed jobs. The
`issue_2170_scenario_adaptive_orca_selector_only` row wrote 15/18 jobs with 3 failed jobs, so the
selector comparison is partial and should not be used for a clean component conclusion.

Local follow-up check: `uv run python -c 'import rvo2'` failed with `ModuleNotFoundError`, which is
consistent with the selector-only failure mode reported by the run. Treat that as an environment
dependency gap to resolve before rerunning the selector row.

The run used commit `10c6142f28d2057a80408fa7c0af9e0b49817e8e`, horizon h80, and two workers. It
is a mechanism diagnostic for local research direction only, not h500 benchmark evidence and not a
planner-promotion claim.
