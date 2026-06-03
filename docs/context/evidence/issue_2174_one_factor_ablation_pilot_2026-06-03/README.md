# Issue #2174 One-Factor Ablation Pilot Evidence

Status: diagnostic-only local pilot evidence.

This directory preserves the compact summary promoted from the first executable one-factor
comparison. Raw JSONL rows, generated candidate reports, and temporary funnel files remain under
ignored `output/` paths.

## Command

```bash
LOGURU_LEVEL=WARNING TF_CPP_MIN_LOG_LEVEL=2 uv run python scripts/tools/run_one_factor_ablation_pilot.py \
  --comparison-id static_escape_only_minus_base --horizon 80 --workers 2 \
  --output-dir output/issue_2174/static_escape_h80_w2
```

## Result

| Comparison | Success delta | Collision delta | Near-miss delta | Avg-speed delta | Runtime delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| static_escape_only_minus_base | 0.000 | 0.000 | 0.000 | 0.000 | +16.328s |

Both candidates wrote 18/18 rows with zero failed jobs on commit
`715816738a143ca4c4984ddb5dc57876d6cf7171`. The run used an h80 local pilot override, not the
manifest's full h500 horizon. Treat it as a mechanism smoke and runtime signal only.
