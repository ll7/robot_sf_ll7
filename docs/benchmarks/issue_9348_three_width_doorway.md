# Three-width doorway comparison (issue #9348)

This application freezes a within-planner comparison at 2.2, 2.8 and 3.6 m
free opening widths and 1.0 m wall depth. It uses the #6644 map generator;
the historical 2.0 m map stays unchanged and is outside the comparison.
The [protocol note](../context/issue_9348_three_width_doorway_protocol.md)
records the source-backed geometry and planner-grid distinction.

The intended campaign has `goal` and `social_force`, seeds 225–227, and a
native horizon of 400 steps at 0.1 s per step: 18 rows. The generated
scenarios differ only in their explained geometry identity and map path.
The planner-free oracle runs first. Its conservative grid search reports no
route at 2.2 and 2.8 m despite positive continuous clearance; the executed
policies do not use that route search. This remains an explicit diagnostic
finding, not a width effect or an ordinary planner failure.

## Diagnostic commands

```bash
uv run python scripts/validation/run_issue_9348_three_width_doorway_preflight.py \
  --out-json output/benchmarks/issue_9348_preflight.json \
  --variants-dir output/benchmarks/issue_9348_variants
uv run python scripts/validation/run_issue_9348_paired_reset_smoke.py \
  --out-json output/benchmarks/issue_9348_pair_smoke.json \
  --variants-dir output/benchmarks/issue_9348_pair_smoke_variants
```

The first command records geometry, oracle findings, asset SHA-256 digests,
and planner rows marked `not_run`. The second uses the real episode runner
for one step in each of the 18 cells. Its opt-in post-reset hook restores a
portable simulator snapshot and records initial actor, external random-stream,
non-width configuration, and map digests. It fails if any planner/seed pair
does not match across three distinct maps. One-step outcomes are diagnostic
only and cannot establish the 400-step comparison. Keep generated assets and
receipts in durable campaign storage before a confirmation run.

The 400-step run and paired uncertainty analysis belong to the 0.0.8
campaign. Exclude unavailable, fallback and degraded rows from success
evidence and report their counts. No physical doorway or deployment safety
claim follows from this simulator-only slice.
